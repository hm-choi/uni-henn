from service import *
from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch
import h5py, os
import time
import math

# public_key.save('key/public_key')
# secret_key.save('key/secret_key')
# galois_key.save('key/galois_key')
# relin_keys.save('key/relin_keys')

import torch.nn.functional as F
import torch.optim as optim

class CNN(torch.nn.Module):
    def __init__(self, hidden=64, output=10):
        super(CNN, self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, stride=3, padding=0)
        self.FC1 = torch.nn.Linear(9 * 9 * 8, 64)
        self.FC2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = self.Conv1(x)
        x = x * x
        x = torch.flatten(x, 1)
        x = self.FC1(x)
        x = x * x
        x = self.FC2(x)
        return x

model_cnn = torch.load('./MNIST_test1.pth', map_location=torch.device('cpu'))

conv2d_client = CNN()
conv2d_client.Conv1.weight.data = model_cnn['Conv1.weight']
conv2d_client.Conv1.bias.data = model_cnn['Conv1.bias']
conv2d_client.FC1.weight.data = model_cnn['FC1.weight']
conv2d_client.FC1.bias.data = model_cnn['FC1.bias']
conv2d_client.FC2.weight.data = model_cnn['FC2.weight']
conv2d_client.FC2.bias.data = model_cnn['FC2.bias']

csps_conv_weights, csps_conv_biases, csps_fc_weights, csps_fc_biases = [], [], [], []
csps_conv_weights.append(model_cnn['Conv1.weight'])
csps_conv_biases.append(model_cnn['Conv1.bias'])
csps_fc_weights.append(model_cnn['FC1.weight'])
csps_fc_biases.append(model_cnn['FC1.bias'])
csps_fc_weights.append(model_cnn['FC2.weight'])
csps_fc_biases.append(model_cnn['FC2.bias'])
strides = [3]
paddings = [0]

# print()
# print("Conv1.weight:\t", csps_conv_weights[0].shape)
# print("Conv1.bias:\t", csps_conv_biases[0].shape)
# print("FC1.weight:\t", csps_fc_weights[0].shape)
# print("FC1.bias:\t", csps_fc_biases[0].shape)
# print("FC2.weight:\t", csps_fc_weights[1].shape)
# print("FC2.bias:\t", csps_fc_biases[1].shape)
# print()

def enc_test(evaluator, ckks_encoder, galois_key, relin_keys, csps_ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, label, scale):
    global labels
    global depth_time
    global convs
    global sqs
    # global avgpools
    global flattens
    global fcs
    global totals
    global errors
    global originals
    global hes
    # global origin_values
    # global he_values
    # global error_values
    global real_labels

    START_TIME = time.time()

    result = re_depth(ckks_encoder, evaluator, relin_keys, [csps_ctxt], 4)
    DEPTH_TIME = time.time()
    print('DROP DEPTH TIME', DEPTH_TIME - START_TIME)
    depth_time.append(DEPTH_TIME - START_TIME)

    result, OH, S, const_param = conv2d_layer_converter_(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_conv_weights[0], csps_conv_biases[0], input_size=image_size, real_input_size=image_size, padding=paddings[0], stride=strides[0], data_size=data_size, const_param=1)
    CHECK_TIME1 = time.time()
    print('CONV2D 1 TIME', CHECK_TIME1-START_TIME)
    convs[0].append(CHECK_TIME1-DEPTH_TIME)

    result, const_param = square(evaluator, relin_keys, result, const_param=const_param)
    CHECK_TIME2 = time.time()
    print('SQ 1 TIME', CHECK_TIME2-CHECK_TIME1)
    sqs[0].append(CHECK_TIME2-CHECK_TIME1)

    result = flatten(evaluator, ckks_encoder, galois_key, relin_keys, result, OH, OH, S, input_size=image_size, data_size=data_size, const_param=const_param)
    CHECK_TIME3 = time.time()
    print('FLATTEN TIME', CHECK_TIME3-CHECK_TIME2)
    flattens.append(CHECK_TIME3-CHECK_TIME2)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key, relin_keys, result, csps_fc_weights[0], csps_fc_biases[0], data_size=data_size)
    CHECK_TIME4 = time.time()
    print('FC1 TIME', CHECK_TIME4-CHECK_TIME3)
    fcs[0].append(CHECK_TIME4-CHECK_TIME3)

    result, const_param = square(evaluator, relin_keys, result, const_param=1)
    CHECK_TIME5 = time.time()
    print('SQ 2 TIME', CHECK_TIME5-CHECK_TIME4)
    sqs[1].append(CHECK_TIME5-CHECK_TIME4)

    result = fc_layer_converter(evaluator, ckks_encoder, galois_key,relin_keys, result, csps_fc_weights[1], csps_fc_biases[1], data_size=data_size)
    END_TIME = time.time()
    print('FC2 TIME', END_TIME-CHECK_TIME5)
    fcs[1].append(END_TIME-CHECK_TIME5)

    # count_correct = 0
    # for i in range(num_of_data):
    #     max_data_idx = 0
    #     dataList = conv2d_client(data)[i].flatten().tolist()
    #     max_data_idx = 1 + dataList.index(max(dataList))

    #     max_ctxt_idx = 0
    #     max_ctxt = -1e10
    #     for j in range(10):
    #         ctxt_data = ckks_encoder.decode(decryptor.decrypt(result))[j+data_size*i]
    #         if(max_ctxt < ctxt_data):
    #             max_ctxt = ctxt_data
    #             max_ctxt_idx = 1 + j
        
    #     if max_data_idx == max_ctxt_idx:
    #         count_correct += 1

    # print('Test Accuracy (Overall): {0}% ({1}/{2})'.format(count_correct/num_of_data*100, count_correct, num_of_data))
    print('Total Time', END_TIME-START_TIME)
    # print(END_TIME-START_TIME)
    totals.append(END_TIME-START_TIME)
    # print()
    
    for i in range(num_of_data):
        max_data_idx = -1
        dataList = conv2d_client(data)[i].flatten().tolist()
        max_data_idx = dataList.index(max(dataList))

        max_ctxt_idx = -1
        max_ctxt = -1e10
        sum = 0
        for j in range(10):
            ctxt_data = ckks_encoder.decode(decryptor.decrypt(result))[j+data_size*i]

            # origin_values[j].append(dataList[j])
            # he_values[j].append(ctxt_data)
            # error_values[j].append(np.abs(dataList[j] - ctxt_data))

            sum = sum + np.abs(dataList[j] - ctxt_data)

            if(max_ctxt < ctxt_data):
                max_ctxt = ctxt_data
                max_ctxt_idx = j
        
        # print(i+1, 'th result')
        # print("Error          |", sum)
        # print("original label |", max_data_idx)
        # print("HE label       |", max_ctxt_idx)
        # print("real label     |", label[i])
        # print("="*30)

        errors[i].append(sum)
        originals[i].append(max_data_idx)
        hes[i].append(max_ctxt_idx)
        real_labels[i].append(label[i])


parms = EncryptionParameters(scheme_type.ckks)
poly_modulus_degree = 8192*2
parms.set_poly_modulus_degree(poly_modulus_degree)
bits_scale1 = 40
# bits_scale2 = 32
for bits_scale2 in [32]:
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
    test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    depth_time = []
    labels = []
    convs = [[]]
    sqs = [[], []]
    # avgpools = [[], [], []]
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

    # origin_values = []
    # he_values = []
    # error_values = []
    # for _ in range(10):
    #     origin_values.append([])
    #     he_values.append([])
    #     error_values.append([])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./../Data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    import pandas as pd
    for index in range(5):
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

        print('result', index+1)
        enc_test(evaluator, ckks_encoder, galois_key, relin_keys, ctxt, csps_conv_weights, csps_conv_biases, image_size, paddings, strides, data_size, _label, scale)
        print()
        
        labels.append(index+1)
        df = pd.DataFrame(labels, columns=["label"])
        df["DEPTH"] = depth_time
        df["CONV1"] = convs[0]
        df["SQ1"] = sqs[0]
        df["Flatten"] = flattens
        df["FC1"] = fcs[0]
        df["SQ2"] = sqs[1]
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

        # for j in range(10):
        #     origin_value_name = "Origin " + str(j)
        #     df[origin_value_name] = origin_values[j]
        # for j in range(10):
        #     he_value_name = "HE " +str(j)
        #     df[he_value_name] = he_values[j]
        # for j in range(10):
        #     error_value_name = "Error "+str(j)
        #     df[error_value_name] = error_values[j]
        
        df.to_csv("M1_scale_"+str(bits_scale2)+".csv", index = False)