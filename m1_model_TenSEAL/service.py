from seal import *
import numpy as np
import time
import math

num_of_slot = 8192
scale = 2**32

def re_depth(encoder, evaluator, relin_keys, ctxt_list, count, _scale):
    global scale

    scale = _scale

    result_list = []
    coeff = [1] * num_of_slot
        
    for ctxt in ctxt_list:
        for _ in range(count):
            encoded_coeff = encoder.encode(coeff, scale)
            evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id())
            ctxt = evaluator.multiply_plain(ctxt, encoded_coeff)
            evaluator.relinearize_inplace(ctxt, relin_keys)
            evaluator.rescale_to_next_inplace(ctxt)
        result_list.append(ctxt)
    return result_list

def calculate_data_size(image_size, csps_conv_weights, csps_fc_weights, strides, paddings):
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
    
def input_layer_converter(encryptor, encoder, plain_vector:np.array):
    if type(plain_vector) != np.array:
        plain_vector = np.array(plain_vector) 
    len_of_pv = len(plain_vector.flatten())
    
    num_of_ctxt = len_of_pv // slot_count + 1

    result_list = []
    flattend = plain_vector.flatten()
    for i in range(num_of_ctxt):
        result_list.append(encryptor.encrypt(encoder.encode(flattend[slot_count*i: slot_count*(i+1)], scale)))
    
    return result_list

def average_pooling_layer_converter(evaluator, encoder, galois_key, relin_keys, ctxt_list, kernel_size, input_size, real_input_size, padding, stride, tmp_param, data_size, const_param):
    result_list = []
    rotated_ctxt_list = []
    len_ctxt = len(ctxt_list)
    OH = (real_input_size + padding * 2 - kernel_size)//stride + 1

    for i in range(len_ctxt):
        ctxt = ctxt_list[i]
        tmp_list = []
        for k1 in range(kernel_size):
            for k2 in range(kernel_size):
                rot_val = tmp_param * (k2 + input_size * k1)
                rotated_ctxt = evaluator.rotate_vector(ctxt, rot_val, galois_key)
                tmp_list.append(rotated_ctxt)
        rotated_ctxt_list.append(tmp_list)
        # result = evaluator.add_many(tmp_list)
        # result_list.append(result)

    # weight = 1/(kernel_size**2)
    # weight = 1
    # weight_list = [weight] + [0] * (stride * tmp_param - 1)
    # weight_list = weight_list * OH
    # weight_list = weight_list + [0]*(input_size * stride * tmp_param - len(weight_list))
    # weight_list = weight_list * OH
    # weight_list = weight_list + [0]*(data_size - len(weight_list))
    # weight_list = weight_list * (num_of_slot//len(weight_list))
    # encoded_coeff = encoder.encode(weight_list, scale)

    for i in range(len_ctxt):
        added_tmp_list = [] 
        for j in range(kernel_size**2):
            rotated_ctxt = rotated_ctxt_list[i][j]
            # evaluator.mod_switch_to_inplace(encoded_coeff, rotated_ctxt.parms_id())
            # mult_ctxt = evaluator.multiply_plain(rotated_ctxt, encoded_coeff)
            # evaluator.relinearize_inplace(mult_ctxt, relin_keys)
            # evaluator.rescale_to_next_inplace(mult_ctxt)

            # added_tmp_list.append(mult_ctxt)
            added_tmp_list.append(rotated_ctxt)
        result = evaluator.add_many(added_tmp_list)
        result_list.append(result)
    return result_list, OH, tmp_param*stride, const_param/(kernel_size**2)

def conv2d_layer_converter_(evaluator, encoder, galois_key, relin_keys, ctxt_list, kernel_list, bias_list, input_size:int=28, real_input_size:int=28, padding:int=0, stride:int=2, tmp_param:int=1, data_size:int=400, const_param:int=1):
    len_output  = kernel_list.shape[0]
    len_input   = kernel_list.shape[1]
    kernel_size = kernel_list.shape[2]
    result_list = []
    rotated_ctxt_list = []
    OH = (real_input_size + padding * 2 - kernel_size)//stride + 1

    for i in range(len_input):
        ctxt = ctxt_list[i]
        tmp_list = []
        for k1 in range(kernel_size):
            for k2 in range(kernel_size):
                rot_val = tmp_param * (k2 + input_size * k1)
                rotated_ctxt = evaluator.rotate_vector(ctxt, rot_val, galois_key)
                tmp_list.append(rotated_ctxt)
        rotated_ctxt_list.append(tmp_list)
    
    for o in range(len_output):
        added_tmp_list = []
        for i in range(len_input):
            for j in range(len(rotated_ctxt_list[i])):
                weight = kernel_list[o][i].flatten()[j].tolist()
                weight = weight * const_param
                rotated_ctxt = rotated_ctxt_list[i][j]
                # if padding > 0 and padding > j:
                #     weight = [0] * (padding-j) + [weight] * (num_of_slot - (padding-j))
                # else:
                weight_list = [weight] + [0] * (stride * tmp_param - 1)
                weight_list = weight_list * OH
                weight_list = weight_list + [0]*(input_size * stride * tmp_param - len(weight_list))
                weight_list = weight_list * OH
                weight_list = weight_list + [0]*(data_size - len(weight_list))
                weight_list = weight_list * (num_of_slot//len(weight_list))
                
                encoded_coeff = encoder.encode(weight_list, scale) #
                evaluator.mod_switch_to_inplace(encoded_coeff, rotated_ctxt.parms_id()) #

                # scale is too small
                try:
                    mult_ctxt = evaluator.multiply_plain(rotated_ctxt, encoded_coeff)
                    evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                    evaluator.rescale_to_next_inplace(mult_ctxt)

                    added_tmp_list.append(mult_ctxt)
                except RuntimeError as e:
                    print("Warning: An error occurred, but it's being ignored:", str(e))

                # mult_ctxt = evaluator.multiply_plain(rotated_ctxt, encoded_coeff)
        result = evaluator.add_many(added_tmp_list)

        bias = [bias_list.tolist()[o]]+[0]*(stride * tmp_param - 1)  
        bias = bias * OH
        bias = bias + [0] * (input_size * stride * tmp_param - len(bias))
        bias = bias * OH
        bias = bias + [0] * (data_size - len(bias))
        bias = bias * (num_of_slot//len(bias)) 

        encoded_bias = encoder.encode(bias, result.scale())
        evaluator.mod_switch_to_inplace(encoded_bias, result.parms_id())
        result = evaluator.add_plain(result, encoded_bias)
        result_list.append(result)

    return result_list, OH, tmp_param*stride, 1

def flatten(evaluator, encoder, galois_key, relin_keys, ctxt_list, OW:int, OH:int, tmp:int, input_size:int=28, data_size:int=400, const_param:int=1):
    if const_param != 1:
        gather_ctxt_list = []
        for ctxt in ctxt_list:
            tmp_list = []

            coeff = [const_param] + [0]*(input_size * tmp - 1)
            coeff = coeff * OH
            coeff = coeff + [0] * (data_size - len(coeff))
            coeff = coeff * (num_of_slot // data_size)
            coeff = coeff + [0] * (num_of_slot - len(coeff))

            for i in range(OH):
                rot_coeff = np.roll(coeff, tmp * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, scale)
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
            coeff = coeff * (num_of_slot // data_size)
            
            num_rot = math.ceil(OH / tmp)
            for i in range(num_rot):
                rot_coeff = np.roll(coeff, tmp**2 * i).tolist()
                encoded_coeff = encoder.encode(rot_coeff, scale)
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
                coeff = np.array(coeff * (num_of_slot//len(coeff)))
                coeff = np.roll(coeff, input_size * tmp * i)

                encoded_coeff = encoder.encode(coeff, scale)
                evaluator.mod_switch_to_inplace(encoded_coeff, ctxt.parms_id()) 
                temp = evaluator.multiply_plain(ctxt, encoded_coeff)
                evaluator.relinearize_inplace(temp, relin_keys)
                evaluator.rescale_to_next_inplace(temp)
                
                rot = i * (input_size * tmp - OH)
                tmp_list.append(evaluator.rotate_vector(temp, rot, galois_key))

            result = evaluator.add_many(tmp_list)
        result_list.append(evaluator.rotate_vector(result, (-1)*o*OW*OH,galois_key))
    return evaluator.add_many(result_list)

def fc_layer_converter(evaluator, encoder, galois_key, relin_keys, ctxt, weights, bias, data_size:int=400):
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
        rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, result_list, weight, ctxt, idx, rot_interval, num_of_slots=num_of_slot, data_size=data_size)

    a = evaluator.add_many(result_list)  
    tmp_list = []
    for i in range(tmp):
        tmp_list.append(evaluator.rotate_vector(a, i*output_size, galois_key))
    all_addition = evaluator.add_many(tmp_list)
    
    bias_list = bias.tolist() + [0]*(data_size-len(bias.tolist()))  # 변경
    bias_list = bias_list*(num_of_slot//len(bias_list))

    sss = encoder.encode(bias_list, all_addition.scale())
    evaluator.mod_switch_to_inplace(sss, all_addition.parms_id())
    return evaluator.add_plain(all_addition, sss)

def batch_norm_converter(evaluator, encoder, ctxt, running_mean, input_size, running_var, weight, beta, epsilon:float=0.001):
    param = 1/np.sqrt(running_var + epsilon)
    # if weight is not None:
    #     coeff = weight * (num_of_slot//)


    temp = evaluator.sub_plain(ctxt, encoder.encode(running_mean * scale_val, scale))
    temp = evaluator.multiply_plain(temp, encoder.encode(param * scale_val, scale))
    temp = evaluator.multiply_plain(temp, encoder.encode(gamma * scale_val, scale))
    return evaluator.add_plain(temp, encoder.encode(beta * scale_val, scale))

# def batch_norm_converter(evaluator, encoder, ctxt, gamma, beta, moving_mean, scale_val, moving_variance, epsilon:float=0.001):
#     param = 1/np.sqrt(moving_variance + epsilon)
#     temp = evaluator.sub_plain(ctxt, encoder.encode(moving_mean * scale_val, scale))
#     temp = evaluator.multiply_plain(temp, encoder.encode(param * scale_val, scale))
#     temp = evaluator.multiply_plain(temp, encoder.encode(gamma * scale_val, scale))
#     return evaluator.add_plain(temp, encoder.encode(beta * scale_val, scale))

def approximated_Swish_converter(evaluator, encoder, relin_keys, ctxt_list, type=0):
    result_list = []
    if type == 0:
        for ctxt in ctxt_list:
            temp = evaluator.multiply_plain(ctxt, encoder.encode([0.145276] * scale_val, scale))
            evaluator.relinearize_inplace(temp, relin_keys)
            temp = evaluator.add_plain(temp, encoder.encode([0.5] * scale_val , temp.scale()))
            evaluator.mod_switch_to_inplace(ctxt, temp.parms_id())
            temp = evaluator.multiply(ctxt, temp)  
            evaluator.relinearize_inplace(temp, relin_keys)
            temp = evaluator.add_plain(temp, encoder.encode([0.12592] * scale_val , temp.scale()))
            result_list.append(temp)
        return result_list 

    elif type == 1:
        for ctxt in ctxt_list:
            sq_ctxt = ctxt * ctxt
            qd_ctxt = sq_ctxt * sq_ctxt
            result_list.append(qd_ctxt * (-0.005075) + sq_ctxt * 0.19566 + ctxt * 0.5 + 0.03347)
        return result_list
    else :
        raise Exception('The type is not appropriated')

def approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, ctxt_list, _type=0, const_param=1):
    coeff1 = [0.117071 * (const_param**2)]*real_size + [0]*(input_size-real_size)
    coeff1 = coeff1 *(num_of_slot//len(coeff1))

    coeff2 = [0.5 * const_param]*real_size + [0]*(input_size-real_size)
    coeff2 = coeff2 *(num_of_slot//len(coeff2))

    coeff3 = [0.375373]*real_size + [0]*(input_size-real_size)
    coeff3 = coeff3 *(num_of_slot//len(coeff3))

    if _type == 0:
        if type(ctxt_list) == list:
            tmp_list = []
            for ctxt in ctxt_list:
                tmp_list.append(approximated_ReLU_converter(evaluator, encoder, input_size, real_size, relin_keys, ctxt, _type, const_param)[0])
            return tmp_list, 1

        else:
            ctxt = ctxt_list
            encoded_coeff1 = encoder.encode(coeff1, scale)
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

def getBinaryNum(n, lists):
    a, b = divmod(n, 2)
    lists.append(b)
    if a == 0 :
        return lists
    else :
        return getBinaryNum(a, lists)

def rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, result_list:list, weight, ctxt, rot_n:int, interval_len:int, num_of_slots:int=num_of_slot, data_size:int=400):
    if rot_n > interval_len or interval_len < 1:
        raise Exception('')
    
    coeff1 = weight[:interval_len-rot_n]
    coeff1 = coeff1 + ([0]*(data_size - len(coeff1)))
    coeff1 = coeff1 * (num_of_slots//len(coeff1))
    coeff1 = coeff1 + [0]*(num_of_slots - len(coeff1))

    coeff2 = [0]*(interval_len-rot_n) + weight[interval_len-rot_n:interval_len]
    coeff2 = coeff2 + ([0]*(data_size - len(coeff2)))
    coeff2 = coeff2 * (num_of_slots//len(coeff2))
    coeff2 = coeff2 + [0]*(num_of_slots - len(coeff2))
        
    if any(coeff1):
        ctxt_rot_n_pos = evaluator.rotate_vector(ctxt, rot_n, galois_key)
        encoded_coeff = encoder.encode(coeff1, scale)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_pos.parms_id())
        result1 = evaluator.multiply_plain(ctxt_rot_n_pos, encoded_coeff)
        evaluator.relinearize_inplace(result1, relin_keys)
        evaluator.rescale_to_next_inplace(result1)
        result_list.append(result1)

    if any(coeff2):
        ctxt_rot_n_neg = evaluator.rotate_vector(ctxt, ((-1)*interval_len + rot_n), galois_key)
        encoded_coeff = encoder.encode(coeff2, scale)
        evaluator.mod_switch_to_inplace(encoded_coeff, ctxt_rot_n_neg.parms_id())
        result2 = evaluator.multiply_plain(ctxt_rot_n_neg, encoded_coeff)
        evaluator.relinearize_inplace(result2, relin_keys)
        evaluator.rescale_to_next_inplace(result2)
        result_list.append(result2)

def square(evaluator, relin_keys, ctxt_list, const_param):
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
        result = evaluator.square(ctxt) # ValueError: scale out of bounds
        evaluator.relinearize_inplace(result, relin_keys)
        evaluator.rescale_to_next_inplace(result)
        return result, const_param**2

def conv1d_layer_converter(evaluator, encoder, galois_key, relin_keys, ctxt_list, weights, bias_list, stride, padding):
    len_output  = weights.shape[0]
    len_input   = weights.shape[1]
    kernel_size = weights.shape[2]
    result_list = []
    for i in range(len_input):
        ctxt = ctxt_list[i]
        rotated_ctxt_list = []
        for k in range(kernel_size):
            rot_val = ((-1) * padding + k * stride)
            rotated_ctxt = evaluator.rotate_vector(ctxt, rot_val, galois_key)
            rotated_ctxt_list.append(rotated_ctxt)

        for o in range(len_output): 
            tmp_list = []
            for c_idx in range(len(rotated_ctxt_list)):
                weight = weights[o][0][c_idx].tolist()
                bias = [bias_list.tolist()[o]]*num_of_slot
                rotated_ctxt = rotated_ctxt_list[c_idx]
                if padding > 0 and padding > c_idx:
                    weight = [0] * (padding-c_idx) + [weight] * (num_of_slot - (padding-c_idx))
                else:
                    weight = [weight] * num_of_slot

                mult_ctxt = evaluator.multiply_plain(rotated_ctxt, encoder.encode(weight, scale))
                evaluator.relinearize_inplace(mult_ctxt, relin_keys)
                tmp_list.append(mult_ctxt)

            result = evaluator.add_many(tmp_list)
            encoded_bais = encoder.encode(bias, result.scale())
            result = evaluator.add_plain(result, encoded_bais)
            result_list.append(result)
    return result_list

def logistic_regression(evaluator, encoder, galois_key, relin_keys, ctxt, weights, bias_list, rot_interval):
    weights = weights.transpose()
    
    len_input  = weights.shape[1] # 784
    len_output   = weights.shape[0] # 10
 
    weights_origin = []
    for idx in range(len_input):
        if idx < len_output:
            weights_origin.append(np.roll(weights[idx], shift=(-1)*idx).tolist())
 
    result_list = []

    tmp = len_input // len_output
    check = False

    if len_input % len_output != 0:
        aa = len_input % len_output
        check = True
        tmp_list = []
        tmp = tmp+1
        for idx in range(len(weights_origin)):
            length = len(weights_origin[idx])
            tmp_list.append(weights_origin[idx][:(length-idx)] + [0]*(len_output-aa) + weights_origin[idx][(length-idx):])
        weights_origin = tmp_list
    weights_origin = np.array(weights_origin).transpose().tolist()
    # rot_interval = np.array(weights_origin).transpose().shape[1]
    for idx in range(len_output):
        weight = []
        for i in range(tmp):
            weight = weight + weights_origin[idx+len_output*i]
        rotate_in_subspace(evaluator, encoder, galois_key, relin_keys, result_list, weight, ctxt, idx, rot_interval, num_of_slots=num_of_slot)

    tmp_added_ctxt = evaluator.add_many(result_list) 


    divisor_set = find_divisor(rot_interval)
    print('divisor_set', divisor_set)
    tmp_list1 = []
    divisor_set = [79]
    M = 79
    for d in divisor_set:
        tmp_list2 = []
        for i in range(d):
            # print(int(i*(M//d)))
            tmp_list2.append(evaluator.rotate_vector(tmp_added_ctxt, i * 10, galois_key))
        tmp_added_ctxt = evaluator.add_many(tmp_list2)
        tmp_list1.append(tmp_added_ctxt)
        M = M/d
    added = evaluator.add_many(tmp_list2)
    bias_list = bias_list.tolist() + [0]*(790 - len(bias_list))
    bias_list = bias_list * (8192//len(bias_list))

    sss = encoder.encode(bias_list, added.scale())
    result = evaluator.mod_switch_to_inplace(sss, added.parms_id())

    result = evaluator.add_plain(added, sss)
    return result

def find_divisor(N):
    divisor_set = []
    m = 2
    while N != 1:  # N과 m을 나눴을 때 몫이 1이 되면 멈춤.
        if N % m == 0: 
            divisor_set.append(m)
            N = N // m
        else:
            m += 1

    return divisor_set

