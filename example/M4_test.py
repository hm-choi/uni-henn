from uni_henn import *
from uni_henn.he_cnn import HE_CNN
from models.model_structures import M4

from seal import *
from torchvision import datasets, transforms
import numpy as np
import torch

import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    m4_model = M4()
    m4_model = torch.load(current_dir + '/models/M4_model.pth', map_location=torch.device('cpu'))

    MNIST_Img = Cuboid(1, 32, 32)
    context = Context()

    HE_m4 = HE_CNN(m4_model, MNIST_Img, context)
    # print(HE_m4)
    # print('='*50)

    num_of_data = int(NUMBER_OF_SLOTS // HE_m4.data_size)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.MNIST(
        root=current_dir + '/Data', 
        train=False, 
        transform=transform,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), MNIST_Img, num_of_data, HE_m4.data_size)

    ciphertext_list = HE_m4.encrypt(ppData)
 
    result_ciphertext = HE_m4(ciphertext_list, _time=True)

    result_plaintext = HE_m4.decrypt(result_ciphertext)

    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = m4_model(data)[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))

        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum = 0
        for idx in range(10):
            he_output = result_plaintext[idx + HE_m4.data_size*i]

            sum = sum + np.abs(origin_results[idx] - he_output)

            if(MIN_VALUE < he_output):
                MIN_VALUE = he_output
                he_result = idx

        """
        After calculating the sum of errors between the results of the original model and the model with homomorphic encryption applied, Outputting whether it matches the original results.
        """        
        print('%sth result Error: %.8f\t| Result is %s' %(str(i+1), sum, "Correct" if origin_result == he_result else "Wrong"))

        # print(i+1, 'th result')
        # print("Error          |", sum)
        # print("original label |", max_data_idx)
        # print("HE label       |", max_ctxt_idx)
        # print("real label     |", _label[i])
        # print("="*30)
