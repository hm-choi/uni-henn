from uni_henn import *
from uni_henn.he_cnn import HE_CNN
from models.model_structures import M6

from seal import *
from torchvision import datasets
import numpy as np
import torch

import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    m6_model = M6()
    m6_model = torch.load(current_dir + '/models/M6_model.pth', map_location=torch.device('cpu'))

    USPS_Img = Cuboid(1, 16, 16)
    context = Context()

    HE_m6 = HE_CNN(m6_model, USPS_Img, context)
    # print(HE_m6)
    # print('='*50)

    num_of_data = int(NUMBER_OF_SLOTS // HE_m6.data_size)
   
    test_dataset = datasets.USPS(
        root=current_dir + '/Data', 
        train=False, 
        transform=TRANSFORM,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), USPS_Img, num_of_data, HE_m6.data_size)

    ciphertext_list = HE_m6.encrypt(ppData)
 
    result_ciphertext = HE_m6(ciphertext_list, _time=True)

    result_plaintext = HE_m6.decrypt(result_ciphertext)

    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = m6_model(data)[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))

        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum = 0
        for idx in range(10):
            he_output = result_plaintext[idx + HE_m6.data_size*i]

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