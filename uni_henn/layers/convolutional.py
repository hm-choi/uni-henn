from seal import *

from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output, Cuboid, Rectangle
from uni_henn.utils.functional import copy_ciphertext

def conv2d_layer_converter_(context: Context, In: Output, Img: Cuboid, layer, data_size):
    """
    This function calculates the 2D convolution operation of the input data.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - In: This is containing the information below
            - ciphertexts: Input ciphertexts list
            - size: Size of input data that is removed the invalid values
            - interval: Interval value between valid data before Conv2d layer
            - const: Value to be multiplied by ciphertext before layer
        - Img: Width (and height) of used image data
        - layer: Convolutional 2D layer that is containing the information below
            - in_channels: Number of input channels
            - out_channels: Number of output channels
            - weight: Kernel weight (shape: CH_out * CH_in * K.h * K.w)
            - bias: Bias value (shape: CH_out)
            - padding: Padding size
            - stride: Stride value
        - data_size: Maximum data size from the total layers

    Returns:
        - Out: This is containing the information below
            - ciphertexts: Output ciphertexts list
            - size: Size of output data that is removed the invalid  values
            - interval: Interval value between valid data after Conv2d layer
            - const: Value to be multiplied by C_out after Conv2d layer (= 1)
    """ 
    CH_in   = layer.in_channels
    CH_out  = layer.out_channels
    K       = Rectangle(layer.kernel_size[0], layer.kernel_size[1])
    S       = Rectangle(layer.stride[0], layer.stride[1])
    P       = Rectangle(layer.padding[0], layer.padding[1])
    
    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_out,
            height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ), 
        interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
        const = 1
    )

    C_rot = []

    for i in range(CH_in):
        C_rot.append([])
        for p in range(K.h):
            C_rot[i].append([])
            for q in range(K.w):
                ciphertext = context.evaluator.rotate_vector(
                    In.ciphertexts[i],
                    In.interval.h * Img.w * p + In.interval.w * q, 
                    context.galois_key
                )
                C_rot[i][p].append(ciphertext)
    
    for o in range(CH_out):
        C_outs = []
        for i in range(CH_in):
            for p in range(K.h):
                for q in range(K.w):
                    """Vector of kernel"""
                    V_ker = [layer.weight.detach().tolist()[o][i][p][q] * In.const] + [0] * (Out.interval.w - 1)
                    V_ker = V_ker * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
                    V_ker = V_ker * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
                    V_ker = V_ker * (context.number_of_slots // data_size)

                    Plaintext_ker = context.encoder.encode(V_ker, context.scale)
                    context.evaluator.mod_switch_to_inplace(Plaintext_ker, C_rot[i][p][q].parms_id())

                    """
                    This try-catch part is handling exceptions for errors that occur when multiplying the vector of 0.
                    """
                    try:
                        ciphertext = context.evaluator.multiply_plain(C_rot[i][p][q], Plaintext_ker)
                        context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
                        context.evaluator.rescale_to_next_inplace(ciphertext)

                        C_outs.append(ciphertext)
                    except RuntimeError as e:
                        print("Warning: An error occurred, but it's being ignored:", str(e))
        
        ciphertext = context.evaluator.add_many(C_outs)

        """Vector of bias"""            
        V_bias = [layer.bias.detach().tolist()[o]] + [0] * (Out.interval.w - 1)  
        V_bias = V_bias * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
        V_bias = V_bias * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
        V_bias = V_bias * (context.number_of_slots // data_size) 

        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
        Out.ciphertexts.append(ciphertext)

    return Out

def conv2d_layer_converter_one_data(context: Context, In: Output, Img: Cuboid, layer, data_size):
    """
    This function calculates the 2D convolution operation of the input data.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - In: This is containing the information below
            - ciphertexts: Input ciphertexts list
            - size: Size of input data that is removed the invalid values
            - interval: Interval value between valid data before Conv2d layer
            - const: Value to be multiplied by ciphertext before layer
        - Img: Width (and height) of used image data
        - layer: Convolutional 2D layer that is containing the information below
            - in_channels: Number of input channels
            - out_channels: Number of output channels
            - weight: Kernel weight (shape: CH_out * CH_in * K.h * K.w)
            - bias: Bias value (shape: CH_out)
            - padding: Padding size
            - stride: Stride value
        - data_size: Maximum data size from the total layers

    Returns:
        - Out: This is containing the information below
            - ciphertexts: Output ciphertexts list
            - size: Size of output data that is removed the invalid  values
            - interval: Interval value between valid data after Conv2d layer
            - const: Value to be multiplied by C_out after Conv2d layer (= 1)
    """ 
    CH_in   = layer.in_channels
    CH_out  = layer.out_channels
    K       = Rectangle(layer.kernel_size[0], layer.kernel_size[1])
    S       = Rectangle(layer.stride[0], layer.stride[1])
    P       = Rectangle(layer.padding[0], layer.padding[1])
    data_num = context.number_of_slots // data_size
    # model 2 -> data_num = 10
    # model 4 -> data_num = 7
    # 즉, kernel size가 5x5이므로 각각 3, 4번의 연산을 수행해야 함.
    
    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_out,
            height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ), 
        interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
        const = 1
    )

    C_copy = []
    for i in range(CH_in):
        C_copy.append(copy_ciphertext(context, In.ciphertexts[i], data_size))
    
    for o in range(CH_out):
        C_outs = []
        for i in range(CH_in):
            for step in range(K.h * K.w // data_num):
                """Vector of kernel"""
                V_ker = []
                for idx in range(data_num):
                    j = step * data_num + idx
                    if j >= K.h * K.w:
                        break
                        
                    p = j // K.w
                    q = j % K.w
                    
                    V_ker_part = [layer.weight.detach().tolist()[o][i][p][q] * In.const] + [0] * (Out.interval.w - 1)
                    V_ker_part = V_ker_part * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
                    V_ker_part = [0] * (In.interval.h * Img.w * p + In.interval.w * q) + V_ker_part * Out.size.h 
                    V_ker_part = V_ker_part + [0] * (data_size - len(V_ker_part))
                    V_ker = V_ker + V_ker_part
                    
                plaintext_ker = context.encoder.encode(V_ker, context.scale)
                context.evaluator.mod_switch_to_inplace(plaintext_ker, C_copy[i].parms_id())
                
                try:
                    ciphertext = context.evaluator.multiply_plain(C_copy[i], plaintext_ker)
                    context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
                    context.evaluator.rescale_to_next_inplace(ciphertext)
                except RuntimeError as e:
                    print("Warning: An error occurred, but it's being ignored:", str(e))

                for idx in range(data_num):
                    j = step * data_num + idx
                    if j >= K.h * K.w:
                        break
                        
                    p = j // K.w
                    q = j % K.w
                    
                    ciphertext_part = context.evaluator.rotate_vector(
                        ciphertext,
                        In.interval.h * Img.w * p + In.interval.w * q + idx * data_size,
                        context.galois_key)
                    
                    if idx == 0:
                        ciphertext_final = ciphertext_part
                    else:
                        ciphertext_final = context.evaluator.add(ciphertext_final, ciphertext_part)
                C_outs.append(ciphertext_final)              
        
        ciphertext = context.evaluator.add_many(C_outs)
        
        V_mask = [1] * data_size
        plaintext_ker = context.encoder.encode(V_mask, context.scale)
        context.evaluator.mod_switch_to_inplace(plaintext_ker, ciphertext.parms_id())
        ciphertext = context.evaluator.multiply_plain(ciphertext, plaintext_ker)
        context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
        context.evaluator.rescale_to_next_inplace(ciphertext)

        """Vector of bias"""            
        V_bias = [layer.bias.detach().tolist()[o]] + [0] * (Out.interval.w - 1)  
        V_bias = V_bias * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
        V_bias = V_bias * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
        V_bias = V_bias * (context.number_of_slots // data_size) 

        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
        Out.ciphertexts.append(ciphertext)

    return Out

def conv1d_layer_converter_(context: Context, In: Output, layer, data_size):
    """
    This function calculates the 1D convolution operation of the input data.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - In: This is containing the information below
            - ciphertexts: Input ciphertexts list
            - size: Size of input data that is removed the invalid values
            - interval: Interval value between valid data before Conv2d layer
            - const: Value to be multiplied by ciphertext before layer
        - layer: Convolutional 1D layer that is containing the information below
            - in_channels: Number of input channels
            - out_channels: Number of output channels
            - weight: Kernel weight (shape: CH_out * CH_in * K.w)
            - bias: Bias value (shape: CH_out)
            - padding: Padding size
            - stride: Stride value
        - data_size: Maximum data size from the total layers

    Returns:
        - Out: This is containing the information below
            - ciphertexts: Output ciphertexts list
            - size: Size of output data that is removed the invalid  values
            - interval: Interval value between valid data after Conv1d layer
            - const: Value to be multiplied by C_out after Conv1d layer (= 1)
    """ 
    CH_in   = layer.in_channels
    CH_out  = layer.out_channels
    K       = Rectangle(1, layer.kernel_size[0])
    S       = Rectangle(1, layer.stride[0])
    P       = Rectangle(0, layer.padding[0])
    
    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_out,
            height = 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ), 
        interval = Rectangle(1, In.interval.w * S.w), 
        const = 1
    )

    C_rot = []

    for i in range(CH_in):
        C_rot.append([])
        for q in range(K.w):
            ciphertext = context.evaluator.rotate_vector(
                In.ciphertexts[i],
                In.interval.w * q,
                context.galois_key
            )
            C_rot[i].append(ciphertext)
    
    for o in range(CH_out):
        C_outs = []
        for i in range(CH_in):
            for q in range(K.w):
                """Vector of kernel"""
                V_ker = [layer.weight.detach().tolist()[o][i][q] * In.const] + [0] * (Out.interval.w - 1)
                V_ker = V_ker * Out.size.w + [0] * (data_size - Out.interval.w * Out.size.w)
                V_ker = V_ker * (context.number_of_slots // data_size)

                Plaintext_ker = context.encoder.encode(V_ker, context.scale)
                context.evaluator.mod_switch_to_inplace(Plaintext_ker, C_rot[i][q].parms_id())

                """
                This try-catch part is handling exceptions for errors that occur when multiplying the vector of 0.
                """
                try:
                    ciphertext = context.evaluator.multiply_plain(C_rot[i][q], Plaintext_ker)
                    context.evaluator.relinearize_inplace(ciphertext, context.relin_keys)
                    context.evaluator.rescale_to_next_inplace(ciphertext)

                    C_outs.append(ciphertext)
                except RuntimeError as e:
                    print("Warning: An error occurred, but it's being ignored:", str(e))
        
        ciphertext = context.evaluator.add_many(C_outs)

        """Vector of bias"""            
        V_bias = [layer.bias.detach().tolist()[o]] + [0] * (Out.interval.w - 1)  
        V_bias = V_bias * Out.size.w + [0] * (data_size - Out.interval.w * Out.size.w)
        V_bias = V_bias * (context.number_of_slots // data_size) 

        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
        Out.ciphertexts.append(ciphertext)

    return Out