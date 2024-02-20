from .context import Context

def re_depth(context: Context, C_in: list, count):
    """
    The function reduces the multiplication depth to meet the depth that is needed in the test.
    Even though the same operation (addition, multiplication, and rotation, etc) is used, more operation time is consumed when the depth is large.

    Args:
        - context: Context that has all the necessary keys
        - C_in: List of the ciphertexts that want to reduce the depth
        - count: The number want to set as the depth

    Returns:
        - C_out: List of the ciphertexts after reducing the depth
    """
    C_out = []
    
    for C in C_in:
        for _ in range(count):
            Plaintext = context.encoder.encode([1] * context.number_of_slots, context.scale)
            context.evaluator.mod_switch_to_inplace(Plaintext, C.parms_id())
            C = context.evaluator.multiply_plain(C, Plaintext)
            context.evaluator.relinearize_inplace(C, context.relin_keys)
            context.evaluator.rescale_to_next_inplace(C)
        C_out.append(C)
    return C_out