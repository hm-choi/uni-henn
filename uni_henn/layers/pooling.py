from seal import *
from uni_henn.utils.context import Context
from uni_henn.utils.structure import Output, Cuboid, Rectangle

def average_pooling_layer_converter(context: Context, In: Output, Img: Cuboid, layer):
    """
    This function calculates the average pooling operation of the input data.
    
    Args:
        - context: Context that has all the necessary keys
            - evaluator: CKKS Evaluator in the SEAL-Python library
            - galois_key: CKKS galois key in the SEAL-Python library
        - In: This is containing the information below
            - ciphertexts: Input ciphertexts list
            - size: Size of input data that is removed the invalid values
            - interval: Interval value between valid data before AvgPool2d layer
            - const: Value to be multiplied by ciphertext before layer
        - Img: Width (and height) of used image data
        - layer: Average pooling layer that is containing the information below
            - kernel_size: Kernel size 
            - padding: Padding size
            - stride: Stride value

    Returns:
        - Out: This is containing the information below
            - ciphertexts: Output ciphertexts list
            - size: Size of output data that is removed the invalid  values
            - interval: Interval value between valid data after AvgPool2d layer
            - const: Value to be multiplied by C_out after AvgPool2d layer
    """
    CH_in = In.size.z
    K     = Rectangle(layer.kernel_size, layer.kernel_size)
    S     = Rectangle(layer.stride, layer.stride)
    P     = Rectangle(layer.padding, layer.padding)
    
    Out = Output(
        ciphertexts = [], 
        size = Cuboid(
            length = CH_in,
            height = (In.size.h + 2 * P.h - K.h) // S.h + 1,
            width = (In.size.w + 2 * P.w - K.w) // S.w + 1
        ), 
        interval = Rectangle(In.interval.h * S.h, In.interval.w * S.w), 
        const = In.const / (K.h * K.w)
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
                    context.galois_key)
                C_rot[i][p].append(ciphertext)

    for i in range(CH_in):
        Ciphtertexts = [] 
        for p in range(K.h):
            for q in range(K.w):
                Ciphtertexts.append(C_rot[i][p][q])
        C_out_o = context.evaluator.add_many(Ciphtertexts)
        Out.ciphertexts.append(C_out_o)

    return Out