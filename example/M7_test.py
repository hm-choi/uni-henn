from uni_henn import *
from models.model_structures import M7

from seal import *
import numpy as np
import torch

import os
import h5py

class ECG(torch.utils.data.Dataset):
    def __init__(self, mode='test'):
        if mode == 'test':
            with h5py.File('./Data/test_ecg.hdf5', 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    m7_model = M7()
    m7_model = torch.load(current_dir + '/models/M7_model.pth', map_location=torch.device('cpu'))

    ECG_Img = Cuboid(1, 1, 128)

    context = sys.argv[1]

    HE_m7 = HE_CNN(m7_model, ECG_Img, context)
    # print(HE_m7)
    # print('='*50)

    num_of_data = int(context.number_of_slots // HE_m7.data_size)
   
    test_dataset = ECG(mode = 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=num_of_data, shuffle=True)

    data, _label = next(iter(test_loader))
    _label = _label.tolist()

    ppData = preprocessing(np.array(data), ECG_Img, num_of_data, HE_m7.data_size)

    ciphertext_list = HE_m7.encrypt(ppData)
 
    result_ciphertext = HE_m7(ciphertext_list, _time=True)

    result_plaintext = HE_m7.decrypt(result_ciphertext)

    for i in range(num_of_data):
        """Model result without homomorphic encryption"""
        origin_results = m7_model(data)[i].flatten().tolist()
        origin_result = origin_results.index(max(origin_results))

        """Model result with homomorphic encryption"""
        he_result = -1
        MIN_VALUE = -1e10
        sum = 0
        for idx in range(5):
            he_output = result_plaintext[idx + HE_m7.data_size*i]

            sum = sum + np.abs(origin_results[idx] - he_output)

            if(MIN_VALUE < he_output):
                MIN_VALUE = he_output
                he_result = idx

        """
        After calculating the sum of errors between the results of the original model and the model with homomorphic encryption applied, Outputting whether it matches the original results.
        """        
        print('%sth result Error: %.8f\t| Result is %s' %(str(i+1), sum, "Correct" if origin_result == he_result else "Wrong"))
