from service import *
from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import h5py, os
import time

parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 8192*2
parms.set_poly_modulus_degree(poly_modulus_degree)
bits_scale1 = 40
bits_scale2 = 32

coeff_mod_bit_sizes = [bits_scale1] + [bits_scale2]*7 + [bits_scale1]
parms.set_coeff_modulus(CoeffModulus.Create(poly_modulus_degree, coeff_mod_bit_sizes))

scale = 2.0**bits_scale2
context = SEALContext(parms)
ckks_encoder = CKKSEncoder(context)
slot_count = ckks_encoder.slot_count()
print(f'Number of slots: {slot_count}') # 8192
 
keygen = KeyGenerator(context)
public_key = keygen.create_public_key()
secret_key = keygen.secret_key()
galois_key = keygen.create_galois_keys()
relin_keys = keygen.create_relin_keys()
encryptor = Encryptor(context, public_key)
evaluator = Evaluator(context)
decryptor = Decryptor(context, secret_key)

public_key.save('key/public_key')
secret_key.save('key/secret_key')
galois_key.save('key/galois_key')
relin_keys.save('key/relin_keys')

import torch.nn.functional as F
import torch.optim as optim

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 16, 16, 1)
        #    Conv     -> (?, 14, 14, 6)
        #    Pool     -> (?, 7, 7, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.FC1 = torch.nn.Linear(294, 64)
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = self.AvgPool1(x)
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

model_cnn = torch.load('./CNN_USPS.pth', map_location=torch.device('cpu'))

conv2d_client = CNN()
conv2d_client.load_state_dict(model_cnn)

csps_conv_weights, csps_conv_biases, csps_fc_weights, csps_fc_biases = [], [], [], []
csps_conv_weights.append(model_cnn['Conv1.weight'])
csps_conv_biases.append(model_cnn['Conv1.bias'])
csps_fc_weights.append(model_cnn['FC1.weight'])
csps_fc_biases.append(model_cnn['FC1.bias'])
csps_fc_weights.append(model_cnn['FC2.weight'])
csps_fc_biases.append(model_cnn['FC2.bias'])
strides = [1]
paddings = [0]

image_size = 16 # Suppose that image shape is sqaure
data_size = 320
num_of_data = int((poly_modulus_degree/2)//data_size)
print(data_size, num_of_data)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.USPS(root='./../Data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

def enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, label):
    START_TIME = time.time()
    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, [csps_ctxt], csps_conv_weights[0], csps_conv_biases[0], input_size=image_size, real_input_size=image_size, padding=paddings[0], stride=strides[0], data_size=data_size, const_param=1)
    CHECK_TIME1 = time.time()
    print(CHECK_TIME1-START_TIME)

    result, const_param = square(evaluator, relin_keys, result, const_param)
    CHECK_TIME2 = time.time()
    print(CHECK_TIME2-CHECK_TIME1)

    result, OH, S, const_param = average_pooling_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, kernel_size=2, input_size=image_size, real_input_size=OH, padding=0, stride=2, tmp_param=S, data_size=data_size, const_param = const_param)
    CHECK_TIME3 = time.time()
    print(CHECK_TIME3-CHECK_TIME2)


    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, S, input_size=image_size, data_size=data_size, const_param=const_param)
    CHECK_TIME4 = time.time()
    print(CHECK_TIME4-CHECK_TIME3)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key,relin_keys, result, csps_fc_weights[0], csps_fc_biases[0], data_size=data_size)
    CHECK_TIME5 = time.time()
    print(CHECK_TIME5-CHECK_TIME4)

    result, const_param = square(evaluator, relin_keys, result, const_param=1)
    CHECK_TIME6 = time.time()
    print(CHECK_TIME6-CHECK_TIME5)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key,relin_keys, result, csps_fc_weights[1], csps_fc_biases[1], data_size=data_size)
    END_TIME = time.time()
    print(END_TIME-CHECK_TIME6)

    count_correct = 0
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
        if max_data_idx == max_ctxt_idx:
            count_correct += 1
        
        # print(i+1, 'th result')
        # print("Error          |", sum)
        # print("original label |", max_data_idx)
        # print("HE label       |", max_ctxt_idx)
        # print("real label     |", label[i])
        # print("="*30)

    # print('Test Accuracy (Overall): {0}% ({1}/{2})'.format(count_correct/num_of_data*100, count_correct, num_of_data))
    # print('Total Time', END_TIME-START_TIME)
    # print()
    print(END_TIME-START_TIME)

for i in range(100):
    data, label = next(iter(test_loader))
    data, label = np.array(data), label.tolist()

    new_data = []
    for i in range(num_of_data):
        new_data.extend(data[i].flatten())
        new_data.extend([0] * (data_size - image_size**2))
    data = torch.Tensor(data)
    new_data = torch.Tensor(new_data)

    csps_ctxt = encryptor.encrypt(ckks_encoder.encode(new_data, scale))
    csps_ctxt.save('ctxt/usps_ctxt')

    enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, label)