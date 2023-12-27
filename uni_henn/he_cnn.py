from layers import *
from utils import *
from constants import *

import torch
import math

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from models.model_structures import *

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
                Out.h = (Out.h + 2 * layer_params.padding[0] - layer_params.kernel_size[0]) // layer_params.stride[0] - 1
                Out.w = (Out.w + 2 * layer_params.padding[1] - layer_params.kernel_size[1]) // layer_params.stride[1] - 1
                
            elif layer.__class__.__name__ == 'AvgPool2d':
                _const = -1
                Out.h = (Out.h + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride - 1
                Out.w = (Out.w + 2 * layer_params.padding - layer_params.kernel_size) // layer_params.stride - 1
            
            elif layer.__class__.__name__ == 'Square':
                req_depth += 1

            elif layer.__class__.__name__ == 'ApproxReLU':
                req_depth += 2
                _const = 1

            elif layer.__class__.__name__ == 'Flatten':
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
            
        return data_size
    
    def encrypt(self, plaintext_list: list):
        ciphertext_list = []
        for plaintext in plaintext_list:
            ciphertext_list.append(
                self.context.encryptor.encrypt(
                    self.context.encoder.encode(plaintext, SCALE)
                )
            )
        return ciphertext_list
    
    def decrypt(self, ciphertext):
        return self.context.encoder.decode(
                    self.context.decryptor.decrypt(ciphertext)
                )
    
    def forward(self, C_in: list):
        C_out = re_depth(self.context, C_in, DEPTH - self.calculate_depth())
        Out = Output(C_out, self.Img)

        for layer_name, layer in self.model.named_children():
            layer_params = getattr(self.model, layer_name)
            
            if layer.__class__.__name__ == 'Conv2d':
                Out = conv2d_layer_converter_(
                    self.context, Out, self.Img, layer_params, self.data_size
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
                    self.context, Out, self.data_size
                )

            elif layer.__class__.__name__ == 'Flatten':
                Out = flatten(self.context, Out, self.Img, self.data_size)

            elif layer.__class__.__name__ == 'Linear':
                Out.ciphertexts[0] = fc_layer_converter(self.context, Out.ciphertexts[0], layer_params, self.data_size)

        return Out.ciphertexts[0]
                
    def __str__(self):
        return self.model.__str__()