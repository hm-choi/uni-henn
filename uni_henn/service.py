# from seal import *
import numpy as np
import time
import math

N = 8192    # Number of slots of ciphertext in pW_iner of 2
q = 2**32   # Scale value used in polynomial quotient ring

def re_depth(encoder, evaluator, relin_keys, C_in, count):
    """
    The function reduces the multiplication depth to meet the depth that is needed in the test.
    Even though the same operation (addition, multiplication, and rotation, etc) is used, 
    more operation time is consumed when the depth is large.

    Args:
        - encoder: CKKS Encoder in the SEAL-Python library
        - evaluator: CKKS Evaluator in the SEAL-Python library
        - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - C_in: List of the ciphertexts that want to reduce the depth
        - count: The number want to set as the depth

    Returns:
        - C_out: List of the ciphertexts after reducing the depth
    """
    C_out = []
        
    for C_in_i in C_in:
        for _ in range(count):
            Plaintext = encoder.encode([1] * N, q)
            evaluator.mod_switch_to_inplace(Plaintext, C_in_i.parms_id())
            C_out_i = evaluator.multiply_plain(C_in_i, Plaintext)
            evaluator.relinearize_inplace(C_out_i, relin_keys)
            evaluator.rescale_to_next_inplace(C_out_i)
        C_out.append(C_out_i)
    return C_out

def calculate_data_size(image_size, csps_conv_weights, csps_fc_weights, strides, paddings):
    """
    The function is used to calculate the largest layer length of the CNN model.

    Args:
        - image_size: Input image size
        - csps_conv_weights: Weight set of the convolutional layer
        - csps_fc_weights: Weight set of the fully connected layer
        - strides: Stride of convolutional layer
        - paddings: Padding size

    Returns:
        - Data_size: Maximum data size from the total layers
    """
    Data_size = image_size**2
    H_in = image_size

    len_conv_layer = len(csps_conv_weights)
    for i in range(len_conv_layer):
        len_output  = csps_conv_weights[i].shape[0]
        kernel_size = csps_conv_weights[i].shape[2]

        H_in = (H_in + paddings[i] * 2 - kernel_size)//strides[i] + 1
        output_size = H_in**2 * len_output
        if output_size > Data_size:
            Data_size = output_size

    for weight in csps_fc_weights:
        input_size = weight.shape[1]
        output_size = weight.shape[0]

        output_size = output_size * math.ceil(input_size / output_size)
        if output_size > Data_size:
            Data_size = output_size

    return Data_size

def average_pooling_layer_converter(evaluator, galois_key, C_in, K, Img, In, P, S, I_in, Const):
    """
    This function calculates the average pooling operation of the input data.
    
    Args:
        - evaluator: CKKS Evaluator in the SEAL-Python library
        - galois_key: CKKS galois key in the SEAL-Python library
        - C_in: Input ciphertexts list
        - K: Kernel size 
        - Img: Width (and height) of used image data
        - In: Size of input data that is removed the invalid values
        - P: Padding size
        - S: Stride value
        - I_in: Interval value between valid data before average pooling layer
        - Const: Value to be multiplied by C_in before layer

    Returns:
        - C_out: Output ciphertexts list
        - Out: Size of output data that is removed the invalid values
        - I_out: Interval value between valid data after average pooling layer
        - Const/(K**2): Value to be multiplied by C_out after average pooling layer
    """
    C_out = []
    Out = (In + P * 2 - K)//S + 1
    I_out = I_in * S

    C_rot = []
    CH_in = len(C_in)

    for i in range(CH_in):
        C_rot.append([])
        for p in range(K):
            C_rot[i].append([])
            for q in range(K):
                Ciphertext = evaluator.rotate_vector(C_in[i], I_in * (q + Img * p), galois_key)
                C_rot[i][p].append(Ciphertext)

    for i in range(CH_in):
        Ciphtertexts = [] 
        for p in range(K):
            for q in range(K):
                Ciphtertexts.append(C_rot[i][p][q])
        C_out_o = evaluator.add_many(Ciphtertexts)
        C_out.append(C_out_o)

    return C_out, Out, I_out, Const/(K**2)
 
def conv2d_layer_converter_(evaluator, encoder, galois_key, relin_keys, C_in, Ker, B, Img, In, P, S, I_in, Data_size, Const:int=1):
    """
    This function calculates the 2D convolution operation of the input data.
    
    Args:
        - evaluator: CKKS Evaluator in the SEAL-Python library
        - encoder: CKKS Encoder in the SEAL-Python library
        - galois_key: CKKS galois key in the SEAL-Python library
        - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - C_in: Input ciphertexts list
        - Ker: Kernel weight (shape : CH_out * CH_in * K * K)
        - B: Bias value (shape : CH_out)
        - Img: Width (and height) of used image data
        - In: Size of input data that is removed the invalid values
        - P: Padding size
        - S: Stride value
        - I_in: Interval value between valid data before conv2d layer
        - Data_size: Maximum data size from the total layers
        - Const: Value to be multiplied by C_in before layer

    Returns:
        - C_out: Output ciphertexts list
        - Out: Size of output data that is removed the invalid  values
        - I_out: Interval value between valid data after average pooling layer
        - Const: Value to be multiplied by C_out after conv2d layer (= 1)
    """
        
    CH_out  = Ker.shape[0]
    CH_in   = Ker.shape[1]
    K       = Ker.shape[2]

    C_out = []
    Out = (In + P * 2 - K)//S + 1
    I_out = I_in * S

    C_rot = []

    for i in range(CH_in):
        C_rot.append([])
        for p in range(K):
            C_rot[i].append([])
            for q in range(K):
                Ciphertext = evaluator.rotate_vector(C_in[i], I_in * (q + Img * p), galois_key)
                C_rot[i][p].append(Ciphertext)
    
    for o in range(CH_out):
        Ciphertexts = []
        for i in range(CH_in):
            for p in range(K):
                for q in range(K):
                    V_ker = [Ker.tolist()[o][i][p][q] * Const] + [0] * (I_out - 1)
                    V_ker = V_ker * Out + [0] * ((Img - Out) * I_out)
                    V_ker = V_ker * Out + [0] * (Data_size - Img * I_out * Out)
                    V_ker = V_ker * (N // Data_size)

                    Plaintext_ker = encoder.encode(V_ker, q)
                    evaluator.mod_switch_to_inplace(Plaintext_ker, C_rot[i][p][q].parms_id())

                    """
                    This try-catch part is handling exceptions for errors that occur when multiplying the vector of 0.
                    """
                    try:
                        Ciphertext = evaluator.multiply_plain(C_rot[i][p][q], Plaintext_ker)
                        evaluator.relinearize_inplace(Ciphertext, relin_keys)
                        evaluator.rescale_to_next_inplace(Ciphertext)

                        Ciphertexts.append(Ciphertext)
                    except RuntimeError as e:
                        print("Warning: An error occurred, but it's being ignored:", str(e))
        
        C_out_o = evaluator.add_many(Ciphertexts)

        V_bias = [B.tolist()[o]] + [0] * (I_out - 1)  
        V_bias = V_bias * Out + [0] * ((Img - Out) * I_out)
        V_bias = V_bias * Out + [0] * (Data_size - Img * I_out * Out)
        V_bias = V_bias * (N // Data_size) 

        Plaintext_bias = encoder.encode(V_bias, C_out_o.scale())
        evaluator.mod_switch_to_inplace(Plaintext_bias, C_out_o.parms_id())
        C_out_o = evaluator.add_plain(C_out_o, Plaintext_bias)
        C_out.append(C_out_o)

    return C_out, Out, I_out, 1

def flatten(evaluator, encoder, galois_key, relin_keys, C_in, W_in:int, H_in:int, S_total:int, In:int=28, Data_size:int=400, Const:int=1):
    """
    The function is used to concatenate between the convolution layer and the fully connected layer.

    Args:
        - evaluator: CKKS Evaluator in the SEAL-Python library
        - encoder: CKKS Encoder in the SEAL-Python library
        - galois_key: CKKS galois key in the SEAL-Python library
        - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - C_in : Input ciphertexts list
        - W_in, H_in: Width and height of input data that is removed the invalid values
        - S_total: This is used for temp params
        - In: Size of input data that is removed the invalid values
        - Data_size: The real data size 
        - Const: Value to be multiplied by C_in before layer

    Returns:
        - C_out: The output of the flattened result of the input ciphertext list
    """
    if Const != 1:
        gather_C_in = []
        for ctxt in C_in:
            tmp_list = []

            coeff = [Const] + [0]*(In * S_total - 1)
            coeff = coeff * H_in
            coeff = coeff + [0] * (Data_size - len(coeff))
            coeff = coeff * (N // Data_size)
            coeff = coeff + [0] * (N - len(coeff))

            for i in range(H_in):
                rot_coeff = np.roll(coeff, S_total * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id())
                mult_ctxt = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                evaluator.rescale_to_next_inplace(mult_ctxt)

                mult_ctxt = evaluator.rotate_vector(mult_ctxt, (S_total - 1) * i, galois_key)
                tmp_list.append(mult_ctxt)
                
            gather_C_in.append(evaluator.add_many(tmp_list))

        C_in = gather_C_in

    elif S_total != 1 and W_in != 1:
        rotated_C_in = []
        for ctxt in C_in:
            tmp_list = []
            for s in range(S_total):
                tmp_list.append(evaluator.rotate_vector(ctxt, s*(S_total-1), galois_key))
            rotated_C_in.append(evaluator.add_many(tmp_list))
        
        gather_C_in = []
        for ctxt in rotated_C_in:
            tmp_list = []

            coeff = [1]*S_total + [0]*((In - 1) * S_total)
            coeff = coeff * H_in
            coeff = coeff + [0] * (Data_size - len(coeff))
            coeff = coeff * (N // Data_size)
            
            num_rot = math.ceil(H_in / S_total)
            for i in range(num_rot):
                rot_coeff = np.roll(coeff, S_total**2 * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id())
                mult_ctxt = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                evaluator.rescale_to_next_inplace(mult_ctxt)

                mult_ctxt = evaluator.rotate_vector(mult_ctxt, S_total*(S_total-1) * i, galois_key)
                tmp_list.append(mult_ctxt)
            
            gather_C_in.append(evaluator.add_many(tmp_list))

        C_in = gather_C_in

    num_ctxt = len(C_in)
    C_outs = []
    for o in range(num_ctxt):
        ctxt = C_in[o]
        if H_in == 1:
            C_out = ctxt
        else:
            tmp_list = []
            for i in range(H_in):
                coeff = [1]*H_in + [0]*(Data_size - H_in)
                coeff = np.array(coeff * (N//len(coeff)))
                coeff = np.roll(coeff, In * S_total * i)

                encoded_coeff = encoder.encode(coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id()) 
                temp = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(temp, relin_keys)
                evaluator.rescale_to_next_inplace(temp)
                
                rot = i * (In * S_total - H_in)
                tmp_list.append(evaluator.rotate_vector(temp, rot, galois_key))

            C_out = evaluator.add_many(tmp_list)
        C_outs.append(evaluator.rotate_vector(C_out, (-1)*o*W_in*H_in,galois_key))
    return evaluator.add_many(C_outs)

# FCLayerConverter
def fc_layer_converter(evaluator, encoder, galois_key, relin_keys, C_in, M_w, bias, Data_size:int=400):
    """
    The function offers a HE-based fully connected layer operation with input ciphertext.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - C_in : Input ciphertext
        - M_w : Weight matrix (shape: DAT_out * DAT_in)
        - bias : The bias of the FC layer
        - Data_size : The data size

    Returns: 
        - C_out : The output of the FC layer of the input ciphertext
    """
    DAT_in = M_w.shape[1]
    DAT_out = M_w.shape[0]
    M_rot = []
    for o in range(DAT_out):
        M_rot.append(np.roll(M_w[o], shift=(-1)*o).tolist())

    q = DAT_in // DAT_out

    if DAT_in % DAT_out != 0:
        r = DAT_in % DAT_out
        q = q + 1
        for o in range(DAT_out):
            M_rot.append(M_rot[o][:(DAT_in-o)] + [0]*(DAT_out-r) + M_rot[o][(DAT_in-o):])
    M_rot = np.array(M_rot).transpose().tolist()
    
    I_rot = M_rot.shape[0]  # I_rot = math.ceil(DAT_in / DAT_out)
    C_outs = []

    for o in range(DAT_out):
        weight = []
        for i in range(q):
            weight = weight + M_rot[o + DAT_out * i]
        rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, C_outs, weight, C_in, o, I_rot, Data_size=Data_size)

    a = evaluator.add_many(C_outs)  
    tmp_list = []
    for i in range(q):
        tmp_list.append(evaluator.rotate_vector(a, i*DAT_out, galois_key))
    all_addition = evaluator.add_many(tmp_list)
    
    bias_list = bias.tolist() + [0]*(Data_size-len(bias.tolist()))  
    bias_list = bias_list*(N//len(bias_list))

    sss = encoder.encode(bias_list, all_addition.scale())
    evaluator.mod_switch_to_inplace(sss, all_addition.parms_id())
    return evaluator.add_plain(all_addition, sss)

def rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, C_outs:list, weight, C, rot_n:int, I_rot:int, N:int=N, Data_size:int=400):
    """
    The function that properly rotates the input ciphertext and multiplies it with the weight. 
    It is used in the FC Layer.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - C_outs : Ciphertext list that needs to be multiplied (It must be added)
        - weight : Plaintext of weight
        - C : Ciphertext that needs to be multiplied
        - rot_n : 회전해야 하는 횟수
        - I_rot : batch 크기 (interval 크기)
        - N : number of slot
        - Data_size : The data size
    """
    if rot_n > I_rot or I_rot < 1:
        raise Exception('')
    
    coeff1 = weight[:I_rot-rot_n]
    coeff1 = coeff1 + ([0]*(Data_size - len(coeff1)))
    coeff1 = coeff1 * (N//len(coeff1))
    coeff1 = coeff1 + [0]*(N - len(coeff1))

    coeff2 = [0]*(I_rot-rot_n) + weight[I_rot-rot_n:I_rot]
    coeff2 = coeff2 + ([0]*(Data_size - len(coeff2)))
    coeff2 = coeff2 * (N // len(coeff2))
    coeff2 = coeff2 + [0]*(N - len(coeff2))
        
    if any(coeff1):
        ctxt_rot_n_pos = evaluator.rotate_vector(C, rot_n, galois_key)
        encoded_coeff = encoder.encode(coeff1, q)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_pos.parms_id())
        result1 = evaluator.multiply_plain(ctxt_rot_n_pos, encoded_coeff)
        evaluator.relinearize_inplace(result1, relin_keys)
        evaluator.rescale_to_next_inplace(result1)
        C_outs.append(result1)

    if any(coeff2):
        ctxt_rot_n_neg = evaluator.rotate_vector(C, ((-1)*I_rot + rot_n), galois_key)
        encoded_coeff = encoder.encode(coeff2, q)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_neg.parms_id())
        result2 = evaluator.multiply_plain(ctxt_rot_n_neg, encoded_coeff)
        evaluator.relinearize_inplace(result2, relin_keys)
        evaluator.rescale_to_next_inplace(result2)
        C_outs.append(result2)

# ApproximateReLUConverter
def approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, C_in, _type=0, Const=1):
    """
    The function offers a HE-based ReLU operation of the input ciphertexts.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - input_size : Input data size
        - real_size : The real data size that is removed the thresh values.
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - C_in : Input ciphertexts list
        - _type : The type is used for choosing approximate ReLU type
        - Const : The constant parameter in approximate ReLU

    Returns:
        - Applied result of the approximated ReLU
    """
    coeff1 = [0.117071 * (Const**2)]*real_size + [0]*(input_size-real_size)
    coeff1 = coeff1 *(N // len(coeff1))

    coeff2 = [0.5 * Const]*real_size + [0]*(input_size-real_size)
    coeff2 = coeff2 *(N // len(coeff2))

    coeff3 = [0.375373]*real_size + [0]*(input_size-real_size)
    coeff3 = coeff3 *(N // len(coeff3))

    if _type == 0:
        if type(C_in) == list:
            tmp_list = []
            for C in C_in:
                tmp_list.append(approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, C, _type, Const)[0])
            return tmp_list, 1

        else:
            encoded_coeff1 = encoder.encode(coeff1, q)
            evaluator.mod_switch_to_inplace(encoded_coeff1, C_in.parms_id())

            temp = evaluator.multiply_plain(C_in, encoded_coeff1)
            evaluator.relinearize_inplace(temp, relin_keys)

            encoded_coeff2 = encoder.encode(coeff2 , temp.scale())
            evaluator.mod_switch_to_inplace(encoded_coeff2, temp.parms_id())
            temp = evaluator.add_plain(temp, encoded_coeff2)
            temp = evaluator.multiply(C_in, temp)  
            evaluator.relinearize_inplace(temp, relin_keys)
            encoded_coeff3 = encoder.encode(coeff3 , temp.scale())
            evaluator.mod_switch_to_inplace(encoded_coeff3, temp.parms_id())
            temp = evaluator.add_plain(temp, encoded_coeff3)

            evaluator.rescale_to_next_inplace(temp)
            evaluator.rescale_to_next_inplace(temp)
            return temp, 1

    elif _type == 1:
        sq_ctxt = C_in * C_in
        qd_ctxt = sq_ctxt * sq_ctxt
        return qd_ctxt * (-0.0063896) + sq_ctxt * 0.204875 + C_in * 0.5 + 0.234606
    else :
        raise Exception('The type is not appropriated')

# Square
def square(evaluator, relin_keys, C_in, Const):
    """
    The function offers a HE-based square operation of the input ciphertexts.
    
    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - C_in : Input ciphertexts list
        - Const : The constant parameter in square function

    Returns:
        - C_out : Squared result ciphertexts list
        - Const**2 : Square of the Const
    """
    if type(C_in) == list:
        C_out = []
        for C in C_in:
            C = evaluator.square(C)
            evaluator.relinearize_inplace(C, relin_keys)
            evaluator.rescale_to_next_inplace(C)
            C_out.append(C)
        
        return C_out, Const**2
    else:
        C_out = evaluator.square(C_in) 
        evaluator.relinearize_inplace(C_out, relin_keys)
        evaluator.rescale_to_next_inplace(C_out)
        return C_out, Const**2