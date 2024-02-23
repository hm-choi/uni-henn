from uni_henn import *
from models.model_structures import M1

from seal import *
from torchvision import datasets
import numpy as np
import torch

import sys, os

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    m1_model = M1()
    m1_model = torch.load(root_dir + '/models/M1_model.pth', map_location=torch.device('cpu'))

    MNIST_Img = Cuboid(1, 28, 28)

    context = sys.argv[1]

    HE_m1 = HE_CNN(m1_model, MNIST_Img, context)
    # print(HE_m1)
    # print('='*50)

    num_of_data = int(context.number_of_slots // HE_m1.data_size)
   
    test_dataset = datasets.MNIST(
        root=root_dir + '/Data', 
        train=False, 
        transform=TRANSFORM,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), MNIST_Img, num_of_data, HE_m1.data_size)

    ciphertext_list = HE_m1.encrypt(ppData)
 
    result_ciphertext = HE_m1(ciphertext_list, _time=True)

    result_plaintext = HE_m1.decrypt(result_ciphertext)

    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = m1_model(data)[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))

        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum = 0
        for idx in range(10):
            he_output = result_plaintext[idx + HE_m1.data_size*i]

            sum = sum + np.abs(origin_results[idx] - he_output)

            if(MIN_VALUE < he_output):
                MIN_VALUE = he_output
                he_result = idx

        """
        After calculating the sum of errors between the results of the original model and the model with homomorphic encryption applied, Outputting whether it matches the original results.
        """        
        print('%sth result Error: %.8f\t| Result is %s' %(str(i+1), sum, "Correct" if origin_result == he_result else "Wrong"))
