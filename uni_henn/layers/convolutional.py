from seal import *

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from constants import NUMBER_OF_SLOTS, SCALE
from utils.module import Context
from utils.structure import Output, Cuboid, Rectangle

def conv2d_layer_converter_(context: Context, In: Output, Img: Cuboid, layer, data_size):
    """
    This function calculates the 2D convolution operation of the input data.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: Re-linearization key of CKKS scheme in the SEAL-Python library
        - In: blah
            - ciphertexts: Input ciphertexts list
            - size: Size of input data that is removed the invalid values
            - interval: Interval value between valid data before Conv2d layer
            - const: Value to be multiplied by ciphertext before layer
        - Img: Width (and height) of used image data
        - layer: blah
            - in_channels: blah
            - out_channels: blah
            - weight: Kernel weight (shape : CH_out * CH_in * K * K)
            - bias: Bias value (shape : CH_out)
            - padding: Padding size
            - stride: Stride value
        - data_size: Maximum data size from the total layers

    Returns:
        - Out: blah
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
            z = CH_out,
            h = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            w = (In.size.w + 2 * P.w - K.w) // S.w + 1
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
                    V_ker = [list(layer.weight)[o][i][p][q] * In.const] + [0] * (Out.interval.w - 1)
                    V_ker = V_ker * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
                    V_ker = V_ker * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
                    V_ker = V_ker * (NUMBER_OF_SLOTS // data_size)

                    Plaintext_ker = context.encoder.encode(V_ker, SCALE)
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
        V_bias = [list(layer.bias)[o]] + [0] * (Out.interval.w - 1)  
        V_bias = V_bias * Out.size.w + [0] * (Img.w * Out.interval.h - Out.size.w * Out.interval.w)
        V_bias = V_bias * Out.size.h + [0] * (data_size - Img.w * Out.interval.h * Out.size.h)
        V_bias = V_bias * (NUMBER_OF_SLOTS // data_size) 

        Plaintext_bias = context.encoder.encode(V_bias, ciphertext.scale())
        context.evaluator.mod_switch_to_inplace(Plaintext_bias, ciphertext.parms_id())
        ciphertext = context.evaluator.add_plain(ciphertext, Plaintext_bias)
        Out.ciphertexts.append(ciphertext)

    return Out

