from seal import *
import math
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from constants import NUMBER_OF_SLOTS, SCALE

def rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, C_outs, weight, C, rot_n, I_rot, Data_size):
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
        - Data_size : The data size
    """
    if rot_n > I_rot or I_rot < 1:
        raise Exception('')
    
    coeff1 = weight[:I_rot-rot_n]
    coeff1 = coeff1 + ([0]*(Data_size - len(coeff1)))
    coeff1 = coeff1 * (NUMBER_OF_SLOTS//len(coeff1))
    coeff1 = coeff1 + [0]*(NUMBER_OF_SLOTS - len(coeff1))

    coeff2 = [0]*(I_rot-rot_n) + weight[I_rot-rot_n:I_rot]
    coeff2 = coeff2 + ([0]*(Data_size - len(coeff2)))
    coeff2 = coeff2 * (NUMBER_OF_SLOTS // len(coeff2))
    coeff2 = coeff2 + [0]*(NUMBER_OF_SLOTS - len(coeff2))
        
    if any(coeff1):
        ctxt_rot_n_pos = evaluator.rotate_vector(C, rot_n, galois_key)
        encoded_coeff = encoder.encode(coeff1, SCALE)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_pos.parms_id())
        result1 = evaluator.multiply_plain(ctxt_rot_n_pos, encoded_coeff)
        evaluator.relinearize_inplace(result1, relin_keys)
        evaluator.rescale_to_next_inplace(result1)
        C_outs.append(result1)

    if any(coeff2):
        ctxt_rot_n_neg = evaluator.rotate_vector(C, ((-1)*I_rot + rot_n), galois_key)
        encoded_coeff = encoder.encode(coeff2, SCALE)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_neg.parms_id())
        result2 = evaluator.multiply_plain(ctxt_rot_n_neg, encoded_coeff)
        evaluator.relinearize_inplace(result2, relin_keys)
        evaluator.rescale_to_next_inplace(result2)
        C_outs.append(result2)

def fc_layer_converter(evaluator, encoder, galois_key, relin_keys, C_in, M_w, bias, Data_size):
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
        rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, C_outs, weight, C_in, o, I_rot, Data_size)

    a = evaluator.add_many(C_outs)  
    tmp_list = []
    for i in range(q):
        tmp_list.append(evaluator.rotate_vector(a, i*DAT_out, galois_key))
    all_addition = evaluator.add_many(tmp_list)
    
    bias_list = bias.tolist() + [0]*(Data_size-len(bias.tolist()))  
    bias_list = bias_list*(NUMBER_OF_SLOTS//len(bias_list))

    sss = encoder.encode(bias_list, all_addition.scale())
    evaluator.mod_switch_to_inplace(sss, all_addition.parms_id())
    return evaluator.add_plain(all_addition, sss)