from seal import *
import math
import numpy as np

from uni_henn.utils.context import Context

def rotate_in_subspace(context: Context, C_outs: list, weight, ciphertext, rot_n, I_rot, data_size):
    """
    The function that properly rotates the input ciphertext and multiplies it with the weight. 
    It is used in the FC Layer.

    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - C_outs: Ciphertext list that needs to be multiplied (It must be added)
        - weight: Plaintext of weight
        - ciphertext: Ciphertext that needs to be multiplied
        - rot_n: Number of rotations required
        - I_rot: batch size (interval size)
        - data_size: Maximum data size from the total layers
    """
    if rot_n > I_rot or I_rot < 1:
        raise Exception('')
    
    coeff1 = weight[:I_rot-rot_n]
    coeff1 = coeff1 + ([0]*(data_size - len(coeff1)))
    coeff1 = coeff1 * (context.number_of_slots//len(coeff1))
    coeff1 = coeff1 + [0]*(context.number_of_slots - len(coeff1))

    coeff2 = [0]*(I_rot-rot_n) + weight[I_rot-rot_n:I_rot]
    coeff2 = coeff2 + ([0]*(data_size - len(coeff2)))
    coeff2 = coeff2 * (context.number_of_slots // len(coeff2))
    coeff2 = coeff2 + [0]*(context.number_of_slots - len(coeff2))
        
    if any(coeff1):
        ctxt_rot_n_pos = context.evaluator.rotate_vector(ciphertext, rot_n, context.galois_key)
        encoded_coeff = context.encoder.encode(coeff1, context.scale)
        context.evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_pos.parms_id())
        result1 = context.evaluator.multiply_plain(ctxt_rot_n_pos, encoded_coeff)
        context.evaluator.relinearize_inplace(result1, context.relin_keys)
        context.evaluator.rescale_to_next_inplace(result1)
        C_outs.append(result1)

    if any(coeff2):
        ctxt_rot_n_neg = context.evaluator.rotate_vector(ciphertext, ((-1)*I_rot + rot_n), context.galois_key)
        encoded_coeff = context.encoder.encode(coeff2, context.scale)
        context.evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_neg.parms_id())
        result2 = context.evaluator.multiply_plain(ctxt_rot_n_neg, encoded_coeff)
        context.evaluator.relinearize_inplace(result2, context.relin_keys)
        context.evaluator.rescale_to_next_inplace(result2)
        C_outs.append(result2)

def fc_layer_converter(context: Context, C_in, layer, data_size):
    """
    The function offers a HE-based fully connected layer operation with input ciphertext.

    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - C_in: Input ciphertext
        - layer: FC layer that contains weight and bias parameter
            - weight: Weight matrix (shape: DAT_out * DAT_in)
            - bias: The bias of the FC layer
        - data_size: Maximum data size from the total layers

    Returns: 
        - C_out: The output of the FC layer of the input ciphertext
    """
    DAT_in = layer.in_features
    DAT_out = layer.out_features
    M_rot = []
    for o in range(DAT_out):
        M_rot.append(np.roll(layer.weight.detach()[o], shift=(-1)*o).tolist())

    q = math.ceil(DAT_in / DAT_out)

    if DAT_in % DAT_out != 0:
        M_rot_transform = []
        r = DAT_in % DAT_out
        for o in range(DAT_out):
            M_rot_transform.append(M_rot[o][:(DAT_in-o)] + [0]*(DAT_out-r) + M_rot[o][(DAT_in-o):])
        M_rot = M_rot_transform

    M_rot = np.array(M_rot).transpose().tolist()
    
    I_rot = q * DAT_out
    C_outs = []

    for o in range(DAT_out):
        weight = []
        for i in range(q):
            weight = weight + M_rot[o + DAT_out * i]
        rotate_in_subspace(context, C_outs, weight, C_in, o, I_rot, data_size)

    a = context.evaluator.add_many(C_outs)  
    tmp_list = []
    for i in range(q):
        tmp_list.append(context.evaluator.rotate_vector(a, i*DAT_out, context.galois_key))
    all_addition = context.evaluator.add_many(tmp_list)
    
    bias_list = layer.bias.detach().tolist() + [0]*(data_size-len(layer.bias.tolist()))  
    bias_list = bias_list*(context.number_of_slots // len(bias_list))

    sss = context.encoder.encode(bias_list, all_addition.scale())
    context.evaluator.mod_switch_to_inplace(sss, all_addition.parms_id())
    return context.evaluator.add_plain(all_addition, sss)