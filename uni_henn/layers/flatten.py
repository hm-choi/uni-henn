from seal import *
import math
import numpy as np

from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output, Cuboid, Rectangle

def flatten(context: Context, In: Output, Img: Cuboid, data_size):
    """
    The function is used to concatenate between the convolution layer and the fully connected layer.

    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - encoder: CKKS Encoder in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
            - relin_keys: CKKS re-linearlization key in the SEAL-Python library
        - In: This is containing the information below
            - C_in: Input ciphertexts list
            - W_in, H_in: Width and height of input data that is removed the invalid values
            - S_total: The value is the product of each convolutional layer’s stride and the kernel sizes of all average pooling layers
            - Const: Value to be multiplied by C_in before layer
        - Img: Width (and height) of used image data
        - data_size: Maximum data size from the total layers 

    Returns:
        - C_out: The output of the flattened result of the input ciphertext list
    """
    C_in = In.ciphertexts

    # Removing invalid data and row interval
    if In.const != 1:
        gather_C_in = []
        for C in C_in:
            tmp_list = []

            coeff = [In.const] + [0]*(Img.w * In.interval.h - 1)
            coeff = coeff * In.size.h
            coeff = coeff + [0] * (data_size - len(coeff))
            coeff = coeff * (context.number_of_slots // data_size)
            coeff = coeff + [0] * (context.number_of_slots - len(coeff))

            for i in range(In.size.h):
                rot_coeff = np.roll(coeff, In.interval.w * i).tolist()
                encoded_coeff = context.encoder.encode(rot_coeff, context.scale)
                context.evaluator.mod_switch_to_inplace(encoded_coeff, C.parms_id())
                mult_C = context.evaluator.multiply_plain(C, encoded_coeff)
                context.evaluator.relinearize_inplace(mult_C, context.relin_keys)
                context.evaluator.rescale_to_next_inplace(mult_C)

                mult_C = context.evaluator.rotate_vector(mult_C, (In.interval.w - 1) * i, context.galois_key)
                tmp_list.append(mult_C)
                
            gather_C_in.append(context.evaluator.add_many(tmp_list))

        C_in = gather_C_in

    # Removing the row interval
    elif In.interval.w != 1 and In.size.w != 1:
        rotated_C_in = []
        for C in C_in:
            tmp_list = []
            for s in range(In.interval.w):
                tmp_list.append(context.evaluator.rotate_vector(C, s*(In.interval.w - 1), context.galois_key))
            rotated_C_in.append(context.evaluator.add_many(tmp_list))
        
        gather_C_in = []
        for C in rotated_C_in:
            tmp_list = []

            coeff = [1]*In.interval.w + [0]*(Img.w * In.interval.h - In.interval.w)
            coeff = coeff * In.size.h
            coeff = coeff + [0] * (data_size - len(coeff))
            coeff = coeff * (context.number_of_slots // data_size)
            
            num_rot = math.ceil(In.size.w / In.interval.w)
            for i in range(num_rot):
                rot_coeff = np.roll(coeff, In.interval.w**2 * i).tolist()
                encoded_coeff = context.encoder.encode(rot_coeff, context.scale)
                context.evaluator.mod_switch_to_inplace(encoded_coeff, C.parms_id())
                mult_C = context.evaluator.multiply_plain(C, encoded_coeff)
                context.evaluator.relinearize_inplace(mult_C, context.relin_keys)
                context.evaluator.rescale_to_next_inplace(mult_C)

                mult_C = context.evaluator.rotate_vector(mult_C, In.interval.w * (In.interval.w - 1) * i, context.galois_key)
                tmp_list.append(mult_C)
            
            gather_C_in.append(context.evaluator.add_many(tmp_list))

        C_in = gather_C_in

    # Removing the column interval
    CH_in = len(C_in)
    C_outs = []
    for o in range(CH_in):
        C = C_in[o]
        if In.size.h == 1:
            C_out = C
        else:
            tmp_list = []
            for i in range(In.size.h):
                coeff = [1]*In.size.w + [0]*(data_size - In.size.w)
                coeff = np.array(coeff * (context.number_of_slots//len(coeff)))
                coeff = np.roll(coeff, Img.w * In.interval.h * i)

                encoded_coeff = context.encoder.encode(coeff, context.scale)
                context.evaluator.mod_switch_to_inplace(encoded_coeff, C.parms_id()) 
                temp = context.evaluator.multiply_plain(C, encoded_coeff)
                context.evaluator.relinearize_inplace(temp, context.relin_keys)
                context.evaluator.rescale_to_next_inplace(temp)
                
                tmp_list.append(
                    context.evaluator.rotate_vector(
                        temp, 
                        i * (Img.w * In.interval.h - In.size.w), 
                        context.galois_key
                    )
                )

            C_out = context.evaluator.add_many(tmp_list)
        C_outs.append(
            context.evaluator.rotate_vector(
                C_out, 
                (-1) * o * In.size.w * In.size.h,
                context.galois_key
            )
        )

    Out = Output(
        ciphertexts = [context.evaluator.add_many(C_outs)],
        size = Cuboid(1, 1, In.size.size3d()),
        interval = Rectangle(1, 1),
        const = 1
    )
    return Out