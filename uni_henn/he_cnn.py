from .layers import *
from .utils import *
from .constants import *

import torch
import math
import time
import sys

class HE_CNN(torch.nn.Module):
    """
    Simple CNN model with Homomorphic Encryption:
    Conv2d | AvgPool2d | Flatten | FC | Sqaure | Approx ReLU
    """
    def __init__(self, model, Img: Cuboid, context: Context):
        super().__init__()
        self.model = model
        self.Img = Img
        self.context = context
        self.data_size = self.calculate_data_size()

    def calculate_depth(self):
        req_depth = 0
        _const = 1
        Out = self.Img.CopyNew()

        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)
            
            if layer.__class__.__name__ == 'Conv2d':
                req_depth += 1
                _const = 1
                Out.z = layer_params.out_channels
                Out.h = (Out.h + 2 * layer_params.padding[0] - layer_params.kernel_size[0]) // layer_params.stride[0] + 1
                Out.w = (Out.w + 2 * layer_params.padding[1] - layer_params.kernel_size[1]) // layer_params.stride[1] + 1

            elif layer.__class__.__name__ == 'Conv1d':
                req_depth += 1
                _const = 1
                Out.z = layer_params.out_channels
                Out.w = (Out.w + 2 * layer_params.padding[0] - layer_params.kernel_size[0]) // layer_params.stride[0] + 1
                
            elif layer.__class__.__name__ == 'AvgPool2d':
                _const = -1
                Out.h = (Out.h + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride + 1
                Out.w = (Out.w + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride + 1
            
            elif layer.__class__.__name__ == 'Square':
                req_depth += 1

            elif layer.__class__.__name__ == 'ApproxReLU':
                req_depth += 2
                _const = 1

            elif layer.__class__.__name__ == 'Flatten':
                req_depth += 1
                if Out.w != 1 or _const != 1:
                    req_depth += 1
                if Out.h != 1:
                    req_depth += 1

            elif layer.__class__.__name__ == 'Linear':
                req_depth += 1
            
        return req_depth

    def calculate_data_size(self):
        data_size = self.Img.size2d()

        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)

            if layer.__class__.__name__ == 'AvgPool2d':
                req_size = self.Img.size2d() + (self.Img.w + 1) * (layer_params.stride - 1)
                if data_size < req_size:
                    data_size = req_size
                
            elif layer.__class__.__name__ == 'Linear':
                layer = getattr(self.model, layer_name)
                _size = layer.out_features * math.ceil(layer.in_features / layer.out_features)
                if data_size < _size:
                    data_size = _size
            
        return smallest_power_of_two_geq(data_size)
    
    def encrypt(self, plaintext):
        if type(plaintext) != list:
            return self.context.encryptor.encrypt(
                        self.context.encoder.encode(plaintext, self.context.scale)
                    )
        else:
            ciphertext_list = []
            for plain in plaintext:
                ciphertext_list.append(
                    self.context.encryptor.encrypt(
                        self.context.encoder.encode(plain, self.context.scale)
                    )
                )
            return ciphertext_list
    
    def decrypt(self, ciphertext):
        if type(ciphertext) != list:
            return self.context.encoder.decode(
                        self.context.decryptor.decrypt(ciphertext)
                    ).tolist()
        else:
            plaintext_list = []
            for cipher in ciphertext:
                plaintext_list.append(
                    self.context.encoder.decode(
                        self.context.decryptor.decrypt(cipher)
                    ).tolist()
                )
            return plaintext_list
    
    def forward(self, C_in: list, _time=False):
        req_depth = self.calculate_depth()
        
        if req_depth > self.context.depth:
            raise ValueError("There is not enough depth to infer the current model.")

        if _time:
            START_TIME = time.time()
        C_out = re_depth(self.context, C_in, self.context.depth - req_depth)
        Out = Output(C_out, self.Img)
        copy_count = 1

        if _time:
            _order = 0
            CHECK_TIME = []
            CHECK_TIME.append(time.time())
            print('Drop depth TIME\t %.3f sec' %(CHECK_TIME[_order] - START_TIME))
        
        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)
            
            if layer.__class__.__name__ == 'Conv2d':
                Out, copy_count = conv2d_layer_converter_one_data(
                    self.context, Out, self.Img, layer_params, self.data_size, copy_count
                )
                # if copy_count == 4:
                # return Out.ciphertexts[0]

            # if layer.__class__.__name__ == 'Conv2d':
            #     Out = conv2d_layer_converter_(
            #         self.context, Out, self.Img, layer_params, self.data_size
            #     )
            #     if len(Out.ciphertexts) == 12:
            #         return Out.ciphertexts[0]


            elif layer.__class__.__name__ == 'Conv1d':
                Out = conv1d_layer_converter_(
                    self.context, Out, layer_params, self.data_size
                )

            elif layer.__class__.__name__ == 'AvgPool2d':
                Out = average_pooling_layer_converter(
                    self.context, Out, self.Img, layer_params
                )
            
            elif layer.__class__.__name__ == 'Square':
                Out = square(
                    self.context, Out
                )

            elif layer.__class__.__name__ == 'ApproxReLU':
                Out = approximated_ReLU_converter(
                    self.context, Out
                )

            elif layer.__class__.__name__ == 'Flatten':
                Out = flatten_one_data(self.context, Out, self.Img, self.data_size, copy_count)
                # Out = flatten(self.context, Out, self.Img, self.data_size, copy_count)
                # return Out.ciphertexts[0]

            elif layer.__class__.__name__ == 'Linear':
                Out.ciphertexts[0] = fc_layer_converter(self.context, Out.ciphertexts[0], layer_params, self.data_size)

            if _time:
                CHECK_TIME.append(time.time())
                _order += 1
                if layer.__class__.__name__ == 'ApproxReLU':
                    print('%s TIME %.3f sec' %(layer_name, CHECK_TIME[_order] - CHECK_TIME[_order - 1]))
                else:
                    print('%s TIME\t %.3f sec' %(layer_name, CHECK_TIME[_order] - CHECK_TIME[_order - 1]))

        if _time:
            END_TIME = time.time()
            print('Total Time\t %.3f sec' %(END_TIME - START_TIME))
            print('='*50)

        return Out.ciphertexts[0]
                
    def __str__(self):
        return self.model.__str__()