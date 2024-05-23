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

def copy_ciphertext(context: Context, Ciphertext, data_size):
    # C_1 = Ciphertext.copy()
    data_num = context.number_of_slots // data_size
    
    # # 만약 data_num이 10이라면 10 -> 5 -> 4 -> 2 -> 1이므로
    # # C_1 -> C_2 -> C_4 -> C_5 -> C_10 순으로 만들기
    # counts = [data_num]
    # while data_num > 1:
    #     # counts의 맨 앞에 data_num을 추가        
    #     if data_num % 2 == 0:
    #         data_num = data_num // 2
    #     else:
    #         data_num = data_num - 1
            
    #     counts = [data_num] + counts
    
    for idx in range(1, data_num):
        ciphertext_temp = context.evaluator.rotate_vector(
            Ciphertext, (-1) * idx * data_size, context.galois_key)
        Ciphertext = context.evaluator.add(Ciphertext, ciphertext_temp)
    return Ciphertext  
        
        