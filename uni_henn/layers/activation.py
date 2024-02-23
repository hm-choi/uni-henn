from seal import *

from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output

def approximated_ReLU_converter(context:Context, In: Output):
    """
    The function offers a HE-based ReLU operation of the input ciphertexts.

    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - relin_keys: CKKS re-linearlization key in the SEAL-Python library
         In: This is containing the information below
            - ciphertexts: Input ciphertexts list
            - const: Value to be multiplied by ciphertext before layer

    Returns:
        - Applied result of the approximated ReLU
    """
    coeff1 = [0.117071 * (In.const**2)] * context.number_of_slots
    coeff2 = [0.5 * In.const] * context.number_of_slots
    coeff3 = [0.375373] * context.number_of_slots

    Out = Output(
        ciphertexts = [],
        size = In.size,
        interval = In.interval,
        const = 1
    )
    for C in In.ciphertexts:
        encoded_coeff1 = context.encoder.encode(coeff1, context.scale)
        context.evaluator.mod_switch_to_inplace(encoded_coeff1, C.parms_id())
        C_out = context.evaluator.multiply_plain(C, encoded_coeff1)
        context.evaluator.relinearize_inplace(C_out, context.relin_keys)

        encoded_coeff2 = context.encoder.encode(coeff2 , C_out.scale())
        context.evaluator.mod_switch_to_inplace(encoded_coeff2, C_out.parms_id())
        C_out = context.evaluator.add_plain(C_out, encoded_coeff2)
        
        C_out = context.evaluator.multiply(C, C_out)
        context.evaluator.relinearize_inplace(C_out, context.relin_keys)
        
        encoded_coeff3 = context.encoder.encode(coeff3 , C_out.scale())
        context.evaluator.mod_switch_to_inplace(encoded_coeff3, C_out.parms_id())
        C_out = context.evaluator.add_plain(C_out, encoded_coeff3)

        context.evaluator.rescale_to_next_inplace(C_out)
        context.evaluator.rescale_to_next_inplace(C_out)

        Out.ciphertexts.append(C_out)
    return Out

def square(context: Context, In: Output):
    """
    The function offers a HE-based square operation of the input ciphertexts.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - C_in: Input ciphertexts list
        - Const: The constant parameter in square function

    Returns:
        - C_out: Squared result ciphertexts list
        - Const**2: Square of the Const
    """
    Out = Output(
        ciphertexts = [],
        size = In.size,
        interval = In.interval,
        const = In.const ** 2
    )

    for C in In.ciphertexts:
        C = context.evaluator.square(C)
        context.evaluator.relinearize_inplace(C, context.relin_keys)
        context.evaluator.rescale_to_next_inplace(C)
        Out.ciphertexts.append(C)
    return Out