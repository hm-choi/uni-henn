from seal import *
import math
import numpy as np

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from constants import NUMBER_OF_SLOTS, SCALE

def flatten(evaluator, encoder, galois_key, relin_keys, C_in, W_in, H_in, S_total, Img, Data_size, Const):
    """
    The function is used to concatenate between the convolution layer and the fully connected layer.

    Args:
        - evaluator: CKKS Evaluator in the SEAL-Python library
        - encoder: CKKS Encoder in the SEAL-Python library
        - galois_key: CKKS galois key in the SEAL-Python library
        - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - C_in : Input ciphertexts list
        - W_in, H_in: Width and height of input data that is removed the invalid values
        - S_total: The value is the product of each convolutional layer’s stride and the kernel sizes of all average pooling layers
        - Img: Size of input data that is removed the invalid values
        - Data_size: The real data size 
        - Const: Value to be multiplied by C_in before layer

    Returns:
        - C_out: The output of the flattened result of the input ciphertext list
    """
    if Const != 1:
        gather_C_in = []
        for ctxt in C_in:
            tmp_list = []

            coeff = [Const] + [0]*(Img * S_total - 1)
            coeff = coeff * H_in
            coeff = coeff + [0] * (Data_size - len(coeff))
            coeff = coeff * (NUMBER_OF_SLOTS // Data_size)
            coeff = coeff + [0] * (NUMBER_OF_SLOTS - len(coeff))

            for i in range(H_in):
                rot_coeff = np.roll(coeff, S_total * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, SCALE)
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

            coeff = [1]*S_total + [0]*((Img - 1) * S_total)
            coeff = coeff * H_in
            coeff = coeff + [0] * (Data_size - len(coeff))
            coeff = coeff * (NUMBER_OF_SLOTS // Data_size)
            
            num_rot = math.ceil(H_in / S_total)
            for i in range(num_rot):
                rot_coeff = np.roll(coeff, S_total**2 * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, SCALE)
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
                coeff = np.array(coeff * (NUMBER_OF_SLOTS//len(coeff)))
                coeff = np.roll(coeff, Img * S_total * i)

                encoded_coeff = encoder.encode(coeff, SCALE)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id()) 
                temp = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(temp, relin_keys)
                evaluator.rescale_to_next_inplace(temp)
                
                rot = i * (Img * S_total - H_in)
                tmp_list.append(evaluator.rotate_vector(temp, rot, galois_key))

            C_out = evaluator.add_many(tmp_list)
        C_outs.append(evaluator.rotate_vector(C_out, (-1)*o*W_in*H_in,galois_key))
    return evaluator.add_many(C_outs)