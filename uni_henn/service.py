from seal import *
import numpy as np
import time
import math

N = 8192    # Number of slots of ciphertext in power of 2
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
        - data_size: Maximum data size from the total layers
    """
    data_size = image_size**2
    OH = image_size

    len_conv_layer = len(csps_conv_weights)
    for i in range(len_conv_layer):
        len_output  = csps_conv_weights[i].shape[0]
        kernel_size = csps_conv_weights[i].shape[2]

        OH = (OH + paddings[i] * 2 - kernel_size)//strides[i] + 1
        output_size = OH**2 * len_output
        if output_size > data_size:
            data_size = output_size

    for weight in csps_fc_weights:
        input_size = weight.shape[1]
        output_size = weight.shape[0]

        output_size = output_size * math.ceil(input_size / output_size)
        if output_size > data_size:
            data_size = output_size

    return data_size

def average_pooling_layer_converter(evaluator, galois_key, C_in, K, Img, In, P, S, I_in, const):
    """
    This function calculates the average pooling operation of the input data.
    
    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - C_in : Input ciphertexts list
        - K : Kernel size 
        - Img : Size of used image data (flattened length)
        - In : Size of input data that is removed the invalid values
        - P : Padding size
        - S : Stride value
        - I_in : Interval value between valid data before average pooling layer
        - const : Value to be multiplied by C_in before average pooling layer

    Returns:
        - C_out : Output ciphertexts list
        - Out : Size of output data that is removed the invalid  values
        - I_out : Interval value between valid data after average pooling layer
        - const/(K**2) : Value to be multiplied by C_out after average pooling layer
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

    return C_out, Out, I_out, const/(K**2)
 
def conv2d_layer_converter_(evaluator, encoder, galois_key, relin_keys, C_in, Ker, B, Img, In, P, S, I_in, data_size, const:int=1):
    """
    This function calculates the 2D convolution operation of the input data.
    
    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - C_in : Input ciphertexts list
        - Ker : Kernel weight (shape : CH_out * CH_in * K * K)
        - B : Bias value (shape : CH_out)
        - Img : Size of used image data (flattened length)
        - In : Size of input data that is removed the invalid values
        - P : Padding size
        - S : Stride value
        - I_in : Interval value between valid data before conv2d layer
        - data_size : Maximum data size from the total layers
        - const : Value to be multiplied by C_in before conv2d layer

    Returns:
        - C_out : Output ciphertexts list
        - Out : Size of output data that is removed the invalid  values
        - I_out : Interval value between valid data after average pooling layer
        - 1 : Value to be multiplied by C_out after conv2d layer (= 1)
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
                    V_ker = [Ker.tolist()[o][i][p][q] * const] + [0] * (I_out - 1)
                    V_ker = V_ker * Out + [0] * ((Img - Out) * I_out)
                    V_ker = V_ker * Out + [0] * (data_size - Img * I_out * Out)
                    V_ker = V_ker * (N // data_size)

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
        V_bias = V_bias * Out + [0] * (data_size - Img * I_out * Out)
        V_bias = V_bias * (N // data_size) 

        Plaintext_bias = encoder.encode(V_bias, C_out_o.scale())
        evaluator.mod_switch_to_inplace(Plaintext_bias, C_out_o.parms_id())
        C_out_o = evaluator.add_plain(C_out_o, Plaintext_bias)
        C_out.append(C_out_o)

    return C_out, Out, I_out, 1

def flatten(evaluator, encoder, galois_key, relin_keys, ctxt_list, OW:int, OH:int, tmp:int, input_size:int=28, data_size:int=400, const_param:int=1):
    """
    The function is used to concatenate between the convolution layer and the fully connected layer.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - ctxt_list : The input ciphertexts list
        - OW : Width of the input data
        - OH : Height of the input data
        - tmp : This is used for temp params
        - input_size : The input data size
        - data_size : The real data size 
        - const_param : The parameter in flatten layer
    Returns:
        - result : The output of the flattened result of the input ciphertexts list.
    """
    if const_param != 1:
        gather_ctxt_list = []
        for ctxt in ctxt_list:
            tmp_list = []

            coeff = [const_param] + [0]*(input_size * tmp - 1)
            coeff = coeff * OH
            coeff = coeff + [0] * (data_size - len(coeff))
            coeff = coeff * (N // data_size)
            coeff = coeff + [0] * (N - len(coeff))

            for i in range(OH):
                rot_coeff = np.roll(coeff, tmp * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id())
                mult_ctxt = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                evaluator.rescale_to_next_inplace(mult_ctxt)

                mult_ctxt = evaluator.rotate_vector(mult_ctxt, (tmp - 1) * i, galois_key)
                tmp_list.append(mult_ctxt)
                
            gather_ctxt_list.append(evaluator.add_many(tmp_list))

        ctxt_list = gather_ctxt_list

    elif tmp != 1 and OW != 1:
        rotated_ctxt_list = []
        for ctxt in ctxt_list:
            tmp_list = []
            for s in range(tmp):
                tmp_list.append(evaluator.rotate_vector(ctxt, s*(tmp-1), galois_key))
            rotated_ctxt_list.append(evaluator.add_many(tmp_list))
        
        gather_ctxt_list = []
        for ctxt in rotated_ctxt_list:
            tmp_list = []

            coeff = [1]*tmp + [0]*((input_size - 1) * tmp)
            coeff = coeff * OH
            coeff = coeff + [0] * (data_size - len(coeff))
            coeff = coeff * (N // data_size)
            
            num_rot = math.ceil(OH / tmp)
            for i in range(num_rot):
                rot_coeff = np.roll(coeff, tmp**2 * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id())
                mult_ctxt = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                evaluator.rescale_to_next_inplace(mult_ctxt)

                mult_ctxt = evaluator.rotate_vector(mult_ctxt, tmp*(tmp-1) * i, galois_key)
                tmp_list.append(mult_ctxt)
            
            gather_ctxt_list.append(evaluator.add_many(tmp_list))

        ctxt_list = gather_ctxt_list

    num_ctxt = len(ctxt_list)
    result_list = []
    for o in range(num_ctxt):
        ctxt = ctxt_list[o]
        if OH == 1:
            result = ctxt
        else:
            tmp_list = []
            for i in range(OH):
                coeff = [1]*OH + [0]*(data_size - OH)
                coeff = np.array(coeff * (N//len(coeff)))
                coeff = np.roll(coeff, input_size * tmp * i)

                encoded_coeff = encoder.encode(coeff, q)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id()) 
                temp = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(temp, relin_keys)
                evaluator.rescale_to_next_inplace(temp)
                
                rot = i * (input_size * tmp - OH)
                tmp_list.append(evaluator.rotate_vector(temp, rot, galois_key))

            result = evaluator.add_many(tmp_list)
        result_list.append(evaluator.rotate_vector(result, (-1)*o*OW*OH,galois_key))
    return evaluator.add_many(result_list)

# FCLayerConverter
def fc_layer_converter(evaluator, encoder, galois_key, relin_keys, ctxt, weights, bias, data_size:int=400):
    """
    The function offers a HE-based fully connected layer operation with input ciphertext.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - galois_key : CKKS galois key in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - ctxt : Input ciphertext
        - weights : The weight of the FC layer
        - bias : The bias of the FC layer
        - data_size : The data size
    Returns:
        - result : The out of the FC layer of input ctxt
    """
    input_size = weights.shape[1] 
    output_size = weights.shape[0] 
    weights_origin = []
    for idx in range(input_size):
        if idx < output_size:
            weights_origin.append(np.roll(weights[idx], shift=(-1)*idx).tolist())

    tmp = input_size // output_size

    if input_size % output_size != 0:
        aa = input_size % output_size
        tmp_list = []
        tmp = tmp+1
        for idx in range(len(weights_origin)):
            length = len(weights_origin[idx])
            tmp_list.append(weights_origin[idx][:(length-idx)] + [0]*(output_size-aa) + weights_origin[idx][(length-idx):])
        weights_origin = tmp_list
    weights_origin = np.array(weights_origin).transpose().tolist()
    
    rot_interval = np.array(weights_origin).transpose().shape[1]
    result_list = []

    for idx in range(output_size):
        weight = []
        for i in range(tmp):
            weight = weight + weights_origin[idx+output_size*i]
        rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, result_list, weight, ctxt, idx, rot_interval, data_size=data_size)

    a = evaluator.add_many(result_list)  
    tmp_list = []
    for i in range(tmp):
        tmp_list.append(evaluator.rotate_vector(a, i*output_size, galois_key))
    all_addition = evaluator.add_many(tmp_list)
    
    bias_list = bias.tolist() + [0]*(data_size-len(bias.tolist()))  
    bias_list = bias_list*(N//len(bias_list))

    sss = encoder.encode(bias_list, all_addition.scale())
    evaluator.mod_switch_to_inplace(sss, all_addition.parms_id())
    return evaluator.add_plain(all_addition, sss)

# ApproximateReLUConverter
def approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, ctxt_list, _type=0, const_param=1):
    """
    The function offers a HE-based ReLU operation of the input ciphertexts.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - input_size : Input data size
        - real_size : The real data size that is removed the thresh values.
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - ctxt_list : Input ciphertexts list
        - _type : The type is used for choosing approximate ReLU type
        - const_param : The constant parameter in approximate ReLU
    Returns:
        - Applied result of the approximated ReLU
    """
    coeff1 = [0.117071 * (const_param**2)]*real_size + [0]*(input_size-real_size)
    coeff1 = coeff1 *(N//len(coeff1))

    coeff2 = [0.5 * const_param]*real_size + [0]*(input_size-real_size)
    coeff2 = coeff2 *(N//len(coeff2))

    coeff3 = [0.375373]*real_size + [0]*(input_size-real_size)
    coeff3 = coeff3 *(N//len(coeff3))

    if _type == 0:
        if type(ctxt_list) == list:
            tmp_list = []
            for ctxt in ctxt_list:
                tmp_list.append(approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, ctxt, _type, const_param)[0])
            return tmp_list, 1

        else:
            ctxt = ctxt_list
            encoded_coeff1 = encoder.encode(coeff1, q)
            evaluator.mod_switch_to_inplace(encoded_coeff1, ctxt.parms_id())

            temp = evaluator.multiply_plain(ctxt, encoded_coeff1)
            evaluator.relinearize_inplace(temp, relin_keys)

            encoded_coeff2 = encoder.encode(coeff2 , temp.scale())
            evaluator.mod_switch_to_inplace(encoded_coeff2, temp.parms_id())
            temp = evaluator.add_plain(temp, encoded_coeff2)
            temp = evaluator.multiply(ctxt, temp)  
            evaluator.relinearize_inplace(temp, relin_keys)
            encoded_coeff3 = encoder.encode(coeff3 , temp.scale())
            evaluator.mod_switch_to_inplace(encoded_coeff3, temp.parms_id())
            temp = evaluator.add_plain(temp, encoded_coeff3)

            evaluator.rescale_to_next_inplace(temp)
            evaluator.rescale_to_next_inplace(temp)
            return temp, 1

    elif _type == 1:
        sq_ctxt = ctxt * ctxt
        qd_ctxt = sq_ctxt * sq_ctxt
        return qd_ctxt * (-0.0063896) + sq_ctxt * 0.204875 + ctxt * 0.5 + 0.234606
    else :
        raise Exception('The type is not appropriated')

def rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, result_list:list, weight, ctxt, rot_n:int, interval_len:int, N:int=N, data_size:int=400):
    """
    The function is used in the FC Layer
    """
    if rot_n > interval_len or interval_len < 1:
        raise Exception('')
    
    coeff1 = weight[:interval_len-rot_n]
    coeff1 = coeff1 + ([0]*(data_size - len(coeff1)))
    coeff1 = coeff1 * (N//len(coeff1))
    coeff1 = coeff1 + [0]*(N - len(coeff1))

    coeff2 = [0]*(interval_len-rot_n) + weight[interval_len-rot_n:interval_len]
    coeff2 = coeff2 + ([0]*(data_size - len(coeff2)))
    coeff2 = coeff2 * (N // len(coeff2))
    coeff2 = coeff2 + [0]*(N - len(coeff2))
        
    if any(coeff1):
        ctxt_rot_n_pos = evaluator.rotate_vector(ctxt, rot_n, galois_key)
        encoded_coeff = encoder.encode(coeff1, q)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_pos.parms_id())
        result1 = evaluator.multiply_plain(ctxt_rot_n_pos, encoded_coeff)
        evaluator.relinearize_inplace(result1, relin_keys)
        evaluator.rescale_to_next_inplace(result1)
        result_list.append(result1)

    if any(coeff2):
        ctxt_rot_n_neg = evaluator.rotate_vector(ctxt, ((-1)*interval_len + rot_n), galois_key)
        encoded_coeff = encoder.encode(coeff2, q)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_neg.parms_id())
        result2 = evaluator.multiply_plain(ctxt_rot_n_neg, encoded_coeff)
        evaluator.relinearize_inplace(result2, relin_keys)
        evaluator.rescale_to_next_inplace(result2)
        result_list.append(result2)

# Square
def square(evaluator, relin_keys, ctxt_list, const_param):
    """
    The function offers a HE-based square operation of the input ciphertexts.
    
    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - ctxt_list : Input ciphertexts list
        - const_param : The constant parameter in square function

    Returns:
        - result : Squared result ciphertexts list
        - const_param**2 : Square of the const_param
    """
    if type(ctxt_list) == list:
        result_list = []
        for ctxt in ctxt_list:
            result = evaluator.square(ctxt)
            evaluator.relinearize_inplace(result, relin_keys)
            evaluator.rescale_to_next_inplace(result)
            result_list.append(result)
        
        return result_list, const_param**2
    else:
        ctxt = ctxt_list
        result = evaluator.square(ctxt) 
        evaluator.relinearize_inplace(result, relin_keys)
        evaluator.rescale_to_next_inplace(result)
        return result, const_param**2