root_path = '.'
data_dir = '../Data'
train_name = 'train_ecg.hdf5'
test_name = 'test_ecg.hdf5'
all_name = 'all_ecg.hdf5'

model_dir = '../models'
model_name = 'M7'
model_ext = '.pth'

csv_dir = 'csv'
csv_ext = '.csv'

csv_name = 'conv2'
csv_accs_name = 'accs_conv2'

import os
import sys

current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from models.model_structures import M7

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

class ECG(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            with h5py.File(os.path.join(root_path, data_dir, train_name), 'r') as hdf:
                self.x = hdf['x_train'][:]
                self.y = hdf['y_train'][:]
        elif mode == 'test':
            with h5py.File(os.path.join(root_path, data_dir, test_name), 'r') as hdf:
                self.x = hdf['x_test'][:]
                self.y = hdf['y_test'][:]
        elif mode == 'all':
            with h5py.File(os.path.join(root_path, data_dir, all_name), 'r') as hdf:
                self.x = hdf['x'][:]
                self.y = hdf['y'][:]
        else:
            raise ValueError('Argument of mode should be train, test, or all.')
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float), torch.tensor(self.y[idx])
    
batch_size = 32

train_dataset = ECG(mode='train')
test_dataset = ECG(mode='test')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

x_train, y_train = next(iter(train_loader))

total_batch = len(train_loader)
        
run = 1
epoch = 30
lr = 0.001

def train(nrun, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = list()
    train_accs = list()

    test_losses = list()
    test_accs = list()

    best_test_acc = 0  # best test accuracy 

    for e in range(epoch):
        print("Epoch {} - ".format(e+1), end='')

        # train
        train_loss = 0.0
        correct, total = 0, 0
        for _, batch in enumerate(train_loader):
            x, label = batch  # get feature and label from a batch
            x, label = x.to(device), label.to(device)  # send to device
            optimizer.zero_grad()  # init all grads to zero
            output = model(x)  # forward propagation
            loss = criterion(output, label)  # calculate loss
            loss.backward()  # backward propagation
            optimizer.step()  # weight update

            train_loss += loss.item()
            correct += torch.sum(output.argmax(dim=1) == label).item()
            total += len(label)
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(correct / total)
        print("loss: {:.4f}, acc: {:.2f}%".format(train_losses[-1], train_accs[-1]*100), end=' / ')
        
        # test
        with torch.no_grad():
            test_loss = 0.0
            correct, total = 0, 0
            for _, batch in enumerate(test_loader):
                x, label = batch
                x, label = x.to(device), label.to(device)
                output = model(x)
                loss = criterion(output, label)
                
                test_loss += loss.item()
                correct += torch.sum(output.argmax(dim=1) == label).item()
                total += len(label)
            test_losses.append(test_loss / len(test_loader))
            test_accs.append(correct / total)
        print("test_loss: {:.4f}, test_acc: {:.2f}%".format(test_losses[-1], test_accs[-1]*100))

        # save model that has best validation accuracy
        if test_accs[-1] > best_test_acc:
            best_test_acc = test_accs[-1]
            torch.save(model, os.path.join(root_path, model_dir, '_'.join([model_name, 'model']) + model_ext))
    
    return train_losses, train_accs, test_losses, test_accs

best_test_accs = list()

for i in range(run):
    print('Run', i+1)
    
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    ecgnet = M7()  # init new model
    # torch.save(ecgnet.state_dict(), 'init_weight.pth')  # save init weights
    train_losses, train_accs, test_losses, test_accs = train(i, ecgnet.to(device))  # train

    best_test_accs.append(max(test_accs))  # get best test accuracy
    best_test_acc_epoch = np.array(test_accs).argmax() + 1
    print('Best test accuracy {:.2f}% in epoch {}.'.format(best_test_accs[-1]*100, best_test_acc_epoch))
    print('-' * 100)

    df = pd.DataFrame({  # save model training process into csv file
        'loss': train_losses,
        'test_loss': test_losses,
        'acc': train_accs,
        'test_acc': test_accs
    })
    # df.to_csv(os.path.join(root_path, csv_dir, '_'.join([csv_name, str(i+1)]) + csv_ext))

# df = pd.DataFrame({'best_test_acc': best_test_accs})  # save best test accuracy of each run
# # df.to_csv(os.path.join(root_path, csv_dir, csv_accs_name + csv_ext))


# for i, a in enumerate(best_test_accs):
#     print('Run {}: {:.2f}%'.format(i+1, a*100))

# df = pd.read_csv(os.path.join('csv', 'conv2_1.csv'))
# test_accs = df['test_acc']
# train_accs = df['acc']
# test_losses = df['test_loss']
# train_losses = df['loss']

# fig, ax = plt.subplots(1, 2, figsize=(15, 4))

# ax[0].plot(train_losses)
# ax[0].plot(test_losses)
# ax[0].set_xticks([0, 5, 10, 15, 20, 25, 30])
# ax[0].set_xlabel('Epoch', size=16)
# ax[0].set_ylabel('Loss', size=16)
# ax[0].set_ylim(0.9, 1.1)
# ax[0].set_yticks([0.9, 1.0, 1.1, 1.2])
# ax[0].grid(alpha=0.5)
# ax[0].tick_params(labelsize=16)
# ax[0].legend(['Train loss', 'Test loss'], loc='right', fontsize=16)

# ax[1].set_ylim(0.7, 1.0)
# ax[1].set_yticks([0.7, 0.8, 0.9, 1.0])
# ax[1].plot(train_accs)
# ax[1].plot(test_accs)
# yt = ax[1].get_yticks()
# ax[1].set_yticklabels(['{:,.0%}'.format(x) for x in yt])
# ax[1].set_xticks([0, 5, 10, 15, 20, 25, 30])
# ax[1].set_xlabel('Epoch', size=16)
# ax[1].set_ylabel('Accuracy', size=16, labelpad=-5)
# ax[1].grid(alpha=0.5)
# ax[1].tick_params(labelsize=16)
# ax[1].legend(['Train accuracy', 'Test accuracy'], loc='right', fontsize=16)

# fig.savefig('loss_acc_conv2_seed.pdf', bbox_inches='tight')