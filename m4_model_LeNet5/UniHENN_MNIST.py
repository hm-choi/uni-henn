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
coeff_mod_bit_sizes = [bits_scale1] + [bits_scale2]*11 + [bits_scale1]
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

# public_key.save('key/public_key')
# secret_key.save('key/secret_key')
# galois_key.save('key/galois_key')
# relin_keys.save('key/relin_keys')

import torch.nn.functional as F
import torch.optim as optim

class CNN(torch.nn.Module):
    def __init__(self, hidden=84, output=10):
        super(CNN, self).__init__()
        # L1 Image shape=(?, 32, 32, 1)
        #    Conv     -> (?, 14, 14, 6)
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.AvgPool1 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 14, 14, 6)
        #    Conv     -> (?, 5, 5, 16)
        self.Conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.AvgPool2 = torch.nn.AvgPool2d(kernel_size = 2)
        # L2 Image shape=(?, 5, 5, 16)
        #    Conv     -> (?, 1, 1, 120)
        self.Conv3 = torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.FC1 = torch.nn.Linear(120, hidden)
        self.FC2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = self.AvgPool1(x)

        x = self.Conv2(x)
        x = x * x
        x = self.AvgPool2(x)
        
        x = self.Conv3(x)
        x = x * x
        
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

strides = [1,1,1]
paddings = [0,0,0]

model_cnn = torch.load('./MNIST_LeNet5.pth', map_location=torch.device('cpu'))
# print(model_cnn)

conv2d_client = CNN()
conv2d_client.Conv1.weight.data = model_cnn['Conv1.weight']
conv2d_client.Conv1.bias.data = model_cnn['Conv1.bias']
conv2d_client.Conv2.weight.data = model_cnn['Conv2.weight']
conv2d_client.Conv2.bias.data = model_cnn['Conv2.bias']
conv2d_client.Conv3.weight.data = model_cnn['Conv3.weight']
conv2d_client.Conv3.bias.data = model_cnn['Conv3.bias']

conv2d_client.FC1.weight.data = model_cnn['FC1.weight']
conv2d_client.FC1.bias.data = model_cnn['FC1.bias']
conv2d_client.FC2.weight.data = model_cnn['FC2.weight']
conv2d_client.FC2.bias.data = model_cnn['FC2.bias']

csps_conv_weights, csps_conv_biases, csps_fc_weights, csps_fc_biases = [], [], [], []
csps_conv_weights.append(model_cnn['Conv1.weight'])
csps_conv_biases.append(model_cnn['Conv1.bias'])
csps_conv_weights.append(model_cnn['Conv2.weight'])
csps_conv_biases.append(model_cnn['Conv2.bias'])
csps_conv_weights.append(model_cnn['Conv3.weight'])
csps_conv_biases.append(model_cnn['Conv3.bias'])

csps_fc_weights.append(model_cnn['FC1.weight'])
csps_fc_biases.append(model_cnn['FC1.bias'])
csps_fc_weights.append(model_cnn['FC2.weight'])
csps_fc_biases.append(model_cnn['FC2.bias'])

# print()
# print("Conv1.weight:\t", csps_conv_weights[0].shape)
# print("Conv1.bias:\t", csps_conv_biases[0].shape)
# print("Conv2.weight:\t", csps_conv_weights[1].shape)
# print("Conv2.bias:\t", csps_conv_biases[1].shape)
# print("Conv3.weight:\t", csps_conv_weights[2].shape)
# print("Conv3.bias:\t", csps_conv_biases[2].shape)
# print("FC1.weight:\t", csps_fc_weights[0].shape)
# print("FC1.bias:\t", csps_fc_biases[0].shape)
# print("FC2.weight:\t", csps_fc_weights[1].shape)
# print("FC2.bias:\t", csps_fc_biases[1].shape)
# print()

image_size = 32 # Suppose that image shape is sqaure
data_size = 32*32 + 32 + 1
num_of_data = int((poly_modulus_degree/2)//data_size)
# num_of_data = 1

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])
test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

def enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, label):
    global labels
    global convs
    global sqs
    global avgpools
    global flattens
    global fcs
    global totals
    global errors
    global originals
    global hes
    global real_labels

    START_TIME = time.time()
    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, [csps_ctxt], csps_conv_weights[0], csps_conv_biases[0], input_size=image_size, real_input_size=image_size, padding=paddings[0], stride=strides[0], data_size=data_size, const_param =1)
    CHECK_TIME1 = time.time()
    print('CONV2D 1 TIME', CHECK_TIME1-START_TIME)
    convs[0].append(CHECK_TIME1-START_TIME)

    result, const_param = square(evaluator, relin_keys, result, const_param)
    CHECK_TIME2 = time.time()
    print('SQ 1 TIME', CHECK_TIME2-CHECK_TIME1)
    sqs[0].append(CHECK_TIME2-CHECK_TIME1)

    result, OH, S, const_param = average_pooling_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, kernel_size=2, input_size=image_size, real_input_size=OH, padding=0, stride=2, tmp_param=S, data_size=data_size, const_param=const_param)
    CHECK_TIME3 = time.time()
    print('AvgPool 1 TIME', CHECK_TIME3-CHECK_TIME2)
    avgpools[0].append(CHECK_TIME3-CHECK_TIME2)


    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_conv_weights[1], csps_conv_biases[1], input_size=image_size, real_input_size=OH, padding=paddings[1], stride=strides[1], tmp_param = S, data_size=data_size, const_param=const_param)
    CHECK_TIME4 = time.time()
    print('CONV2D 2 TIME', CHECK_TIME4-CHECK_TIME3)
    convs[1].append(CHECK_TIME4-CHECK_TIME3)

    result, const_param = square(evaluator, relin_keys, result, const_param)
    CHECK_TIME5 = time.time()
    print('SQ 2 TIME', CHECK_TIME5-CHECK_TIME4)
    sqs[1].append(CHECK_TIME5-CHECK_TIME4)

    result, OH, S, const_param = average_pooling_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, kernel_size=2, input_size=image_size, real_input_size=OH, padding=0, stride=2, tmp_param=S, data_size=data_size, const_param=const_param)
    CHECK_TIME6 = time.time()
    print('AvgPool 2 TIME', CHECK_TIME6-CHECK_TIME5)
    avgpools[1].append(CHECK_TIME6-CHECK_TIME5)


    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_conv_weights[2], csps_conv_biases[2], input_size=image_size, real_input_size=OH, padding=paddings[2], stride=strides[2], tmp_param = S, data_size=data_size, const_param=const_param)
    CHECK_TIME7 = time.time()
    print('CONV2D 3 TIME', CHECK_TIME7-CHECK_TIME6)
    convs[2].append(CHECK_TIME7-CHECK_TIME6)

    result, const_param = square(evaluator, relin_keys, result, const_param)
    CHECK_TIME8 = time.time()
    print('SQ 3 TIME', CHECK_TIME8-CHECK_TIME7)
    sqs[2].append(CHECK_TIME8-CHECK_TIME7)

    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, S, input_size=image_size, data_size=data_size, const_param=const_param)
    CHECK_TIME9 = time.time()
    print('FLATTEN TIME', CHECK_TIME9-CHECK_TIME8)
    flattens.append(CHECK_TIME9-CHECK_TIME8)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key,relin_keys, result, csps_fc_weights[0], csps_fc_biases[0], data_size=data_size)
    CHECK_TIME10 = time.time()
    print('FC1 TIME', CHECK_TIME10-CHECK_TIME9)
    fcs[0].append(CHECK_TIME10-CHECK_TIME9)

    result, const_param = square(evaluator, relin_keys, result, 1)
    CHECK_TIME11 = time.time()
    print('SQ 4 TIME', CHECK_TIME11-CHECK_TIME10)
    sqs[3].append(CHECK_TIME11-CHECK_TIME10)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key,relin_keys, result, csps_fc_weights[1], csps_fc_biases[1], data_size=data_size)
    END_TIME = time.time()
    print('FC2 TIME', END_TIME-CHECK_TIME11)
    fcs[1].append(END_TIME-CHECK_TIME11)

    count_correct = 0
    for i in range(num_of_data):
        max_data_idx = 0
        dataList = conv2d_client(data)[i].flatten().tolist()
        max_data_idx = 1 + dataList.index(max(dataList))

        max_ctxt_idx = 0
        max_ctxt = -1e10 
        for j in range(10):
            ctxt_data = ckks_encoder.decode(decryptor.decrypt(result))[j+data_size*i]
            if(max_ctxt < ctxt_data):
                max_ctxt = ctxt_data
                max_ctxt_idx = 1 + j
        
        if max_data_idx == max_ctxt_idx:
            count_correct += 1

    print('Test Accuracy (Overall): {0}% ({1}/{2})'.format(count_correct/num_of_data*100, count_correct, num_of_data))
    print('Total Time', END_TIME-START_TIME)
    print()
    totals.append(END_TIME-START_TIME)

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

        errors[i].append(sum)
        originals[i].append(max_data_idx)
        hes[i].append(max_ctxt_idx)
        real_labels[i].append(label[i])

labels = []
convs = [[], [], []]
sqs = [[], [], [], []]
avgpools = [[], []]
flattens = []
fcs = [[], []]
totals = []

errors = []
originals = []
hes = []
real_labels = []
for _ in range(num_of_data):
    errors.append([])
    originals.append([])
    hes.append([])
    real_labels.append([])

import pandas as pd
for index in range(300):
    data, _label = next(iter(test_loader))
    data, _label = np.array(data), _label.tolist()
    
    new_data = []
    for i in range(num_of_data):
        new_data.extend(data[i].flatten())
        new_data.extend([0] * (data_size - image_size**2))
    data = torch.Tensor(data)
    new_data = torch.Tensor(new_data)

    csps_ctxt = encryptor.encrypt(ckks_encoder.encode(new_data, scale))
    csps_ctxt.save('ctxt/mnist_ctxt')

    labels.append(index+1)
    enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, _label)

    df = pd.DataFrame(labels, columns=["label"])
    df["CONV1"] = convs[0]
    df["SQ1"] = sqs[0]
    df["AvgPool1"] = avgpools[0]
    df["CONV2"] = convs[1]
    df["SQ2"] = sqs[1]
    df["AvgPool2"] = avgpools[1]
    df["CONV3"] = convs[2]
    df["SQ3"] = sqs[2]
    df["Flatten"] = flattens
    df["FC1"] = fcs[0]
    df["SQ4"] = sqs[3]
    df["FC2"] = fcs[1]
    df["Total"] = totals

    for i in range(num_of_data):
        error_name = "Error" + str(i+1)
        original_name = "Original" + str(i+1)
        HE_name = "HE" + str(i+1)
        real_name = "real_label" + str(i+1)

        df[error_name] = errors[i]
        df[original_name] = originals[i]
        df[HE_name] = hes[i]
        df[real_name] = real_labels[i]

    df.to_csv("MNIST_LeNet5.csv", index = False)