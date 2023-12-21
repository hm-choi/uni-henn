from seal import *

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from constants import NUMBER_OF_SLOTS, SCALE

def approximated_ReLU_converter(evaluator, encoder, relin_keys, Data_size, C_in, Const=1):
    """
    The function offers a HE-based ReLU operation of the input ciphertexts.

    Args:
        - evaluator : CKKS Evaluator in the SEAL-Python library
        - encoder : CKKS Encoder in the SEAL-Python library
        - relin_keys : CKKS re-linearlization key in the SEAL-Python library
        - Data_size : Maximum data size from the total layers
        - C_in : Input ciphertexts list
        - Const : The constant parameter in approximate ReLU

    Returns:
        - Applied result of the approximated ReLU
    """
    coeff1 = [0.117071 * (Const**2)] * NUMBER_OF_SLOTS
    coeff2 = [0.5 * Const] * NUMBER_OF_SLOTS
    coeff3 = [0.375373] * NUMBER_OF_SLOTS

    if type(C_in) == list:
        result_list = []
        for C in C_in:
            result_list.append(approximated_ReLU_converter(evaluator, encoder, relin_keys, Data_size, C, Const)[0])
        return result_list, 1

    else:
        encoded_coeff1 = encoder.encode(coeff1, SCALE)
        evaluator.mod_switch_to_inplace(encoded_coeff1, C_in.parms_id())

        result = evaluator.multiply_plain(C_in, encoded_coeff1)
        evaluator.relinearize_inplace(result, relin_keys)

        encoded_coeff2 = encoder.encode(coeff2 , result.scale())
        evaluator.mod_switch_to_inplace(encoded_coeff2, result.parms_id())
        result = evaluator.add_plain(result, encoded_coeff2)
        
        result = evaluator.multiply(C_in, result)
        evaluator.relinearize_inplace(result, relin_keys)
        
        encoded_coeff3 = encoder.encode(coeff3 , result.scale())
        evaluator.mod_switch_to_inplace(encoded_coeff3, result.parms_id())
        result = evaluator.add_plain(result, encoded_coeff3)

        evaluator.rescale_to_next_inplace(result)
        evaluator.rescale_to_next_inplace(result)
        return result, 1

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