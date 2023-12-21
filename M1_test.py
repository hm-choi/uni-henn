from uni_henn import *
from models.model_structures import M1

from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import time

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

HE_CNN = torch.load(current_dir + '/models/M1_model.pth', map_location=torch.device('cpu'))

client = M1()


conv2d_client = CNN()
conv2d_client.load_state_dict(model_cnn)

csps_conv_weights, csps_conv_biases, csps_fc_weights, csps_fc_biases = [], [], [], []
csps_conv_weights.append(model_cnn['Conv1.weight'])
csps_conv_biases.append(model_cnn['Conv1.bias'])
csps_fc_weights.append(model_cnn['FC1.weight'])
csps_fc_biases.append(model_cnn['FC1.bias'])
csps_fc_weights.append(model_cnn['FC2.weight'])
csps_fc_biases.append(model_cnn['FC2.bias'])
strides = [3]
paddings = [0]

def enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, label, scale):
    START_TIME = time.time()

    result = re_depth(ckks_encoder, evaluator, relin_keys, [csps_ctxt], 4)
    DEPTH_TIME = time.time()
    print('DROP DEPTH TIME\t%.3f' %(DEPTH_TIME - START_TIME))

    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_conv_weights[0], csps_conv_biases[0], image_size, image_size, paddings[0], strides[0], 1, data_size, 1)
    CHECK_TIME1 = time.time()
    print('CONV2D 1 TIME\t%.3f' %(CHECK_TIME1-DEPTH_TIME))

    result, const_param = square(evaluator, relin_keys, result, const_param)
    CHECK_TIME2 = time.time()
    print('SQ 1 TIME\t%.3f' %(CHECK_TIME2-CHECK_TIME1))

    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, OH, S, image_size, data_size, const_param)
    CHECK_TIME3 = time.time()
    print('FLATTEN TIME\t%.3f' %(CHECK_TIME3-CHECK_TIME2))

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_fc_weights[0], csps_fc_biases[0], data_size)
    CHECK_TIME4 = time.time()
    print('FC1 TIME\t%.3f' %(CHECK_TIME4-CHECK_TIME3))

    result, const_param = square(evaluator, relin_keys, result, 1)
    CHECK_TIME5 = time.time()
    print('SQ 2 TIME\t%.3f' %(CHECK_TIME5-CHECK_TIME4))

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_fc_weights[1], csps_fc_biases[1], data_size)
    END_TIME = time.time()
    print('FC2 TIME\t%.3f' %(END_TIME-CHECK_TIME5))

    print('Total Time\t%.3f sec' %(END_TIME-START_TIME))
    print()
    
    for i in range(num_of_data):
        max_data_idx = -1
        dataList = conv2d_client(data)[i].flatten().tolist()
        max_data_idx = dataList.index(max(dataList))

        max_ctxt_idx = -1
        max_ctxt = -1e10
        sum = 0
        for j in range(10):
            ctxt_data = ckks_encoder.decode(decryptor.decrypt(result))[j+data_size*i]

            sum = sum + np.abs(dataList[j] - ctxt_data)

            if(max_ctxt < ctxt_data):
                max_ctxt = ctxt_data
                max_ctxt_idx = j
        
        print(i+1, 'th result')
        print("Error          |", sum)
        print("original label |", max_data_idx)
        print("HE label       |", max_ctxt_idx)
        print("real label     |", label[i])
        print("="*30)

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 8192*2
parms.set_poly_modulus_degree(poly_modulus_degree)
bits_scale1 = 40
bits_scale2 = 32
depth = 11
coeff_mod_bit_sizes = [bits_scale1] + [bits_scale2]*depth + [bits_scale1]
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, coeff_mod_bit_sizes))

scale = 2.0**bits_scale2
context = SEALContext(parms)
ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()

keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_key = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

image_size = 28 # Suppose that image shape is sqaure
data_size = calculate_data_size(image_size, csps_conv_weights, csps_fc_weights, strides, paddings)
num_of_data = int(slot_count//data_size)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./Data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

data, _label = next(iter(test_loader))
data, _label = np.array(data), _label.tolist()

new_data = []
for i in range(num_of_data):
    new_data.extend(data[i].flatten())
    new_data.extend([0] * (data_size - image_size**2))
data = torch.Tensor(data)
new_data = torch.Tensor(new_data)

ctxt = encryptor.encrypt(ckks_encoder.encode(new_data, scale))
# ctxt.save('ctxt/mnist_ctxt')

enc_test(evaluator, ckks_encoder, galois_key, relin_keys, ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, _label, scale)
