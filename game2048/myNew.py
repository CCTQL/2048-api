import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import time
import pandas as pd
import numpy as np
import csv


batch_size = 128
NUM_EPOCHS = 30
LR = 0.001
TIME_STEP = 4


class CCRNN(nn.Module):
    def __init__(self):
        # 继承RNN
        super(CCRNN, self).__init__()

        self.ccLSTM = nn.LSTM(
            input_size=4,
            hidden_size=128,
            num_layers=4,
            bidirectional=True,
            batch_first=True

        )

        self.ccCNN22 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            stride=2,
            padding=0
        )

        self.ccCNN14 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 4),
            stride=1,
            padding=0
        )

        self.ccCNN41 = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(4, 1),
            stride=1,
            padding=0
        )

        self.CNN22toFC = nn.Linear(4, 64)
        self.CNN41toFC = nn.Linear(4, 32)
        self.CNN14toFC = nn.Linear(4, 32)
        self.LSTMtoFC = nn.Linear(256, 128)
        self.FCtoOut = nn.Linear(256, 4)

    def forward(self, x):
        LSTM_out, (h_n, c_n) = self.ccLSTM(x, None)
        CNN_in = torch.unsqueeze(x[:, 0:4, :], 1)
        CNN_out22 = self.ccCNN22(CNN_in)
        CNN_out41 = self.ccCNN41(CNN_in)
        CNN_out14 = self.ccCNN14(CNN_in)

        CNN22_reshape = CNN_out22.view(-1, 4)
        CNN14_reshape = CNN_out41.view(-1, 4)
        CNN41_reshape = CNN_out14.view(-1, 4)
        CNN22toFC = self.CNN22toFC(CNN22_reshape)
        CNN14toFC = self.CNN14toFC(CNN14_reshape)
        CNN41toFC = self.CNN41toFC(CNN41_reshape)
        LSTMtoFC = self.LSTMtoFC(LSTM_out[:, -1, :])
        CNNandLSTM = torch.cat((CNN22toFC, CNN41toFC, CNN14toFC, LSTMtoFC), 1)
        out = self.FCtoOut(CNNandLSTM)
        return out

#------------------读入数据-----------------------------
csv_data = pd.read_csv('./drive/My Drive/DATA.csv')
csv_data = csv_data.values
A = csv_data.shape[0]
board_data = csv_data[:,0:16]
# X = np.log2(X)
X = torch.FloatTensor(board_data)
X = np.int64(board_data)

# 转置后拼接
X = np.reshape(X, (-1,4,4))
XT = X.transpose(0,2,1)

X = np.concatenate((X,XT),axis=1)
print(X.shape)

direction_data = csv_data[:,16]
Y = np.int64(direction_data)


#-------------------------------------------------------


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,shuffle=False)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(X_train,Y_train)
# test_dataset = torch.utils.data.TensorDataset(X_test,Y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                      batch_size=batch_size,
                      shuffle=True
)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                       batch_size=batch_size,
#                       shuffle=False
# )

batch_size = 128
NUM_EPOCHS = 30
LR = 0.001
TIME_STEP = 4


#------------------读入数据-----------------------------
csv_data = pd.read_csv('./drive/My Drive/DATA.csv')
csv_data = csv_data.values
A = csv_data.shape[0]
board_data = csv_data[:,0:16]
# X = np.log2(X)
X = torch.FloatTensor(board_data)
X = np.int64(board_data)

# 转置后拼接
X = np.reshape(X, (-1,4,4))
XT = X.transpose(0,2,1)

X = np.concatenate((X,XT),axis=1)
print(X.shape)

direction_data = csv_data[:,16]
Y = np.int64(direction_data)



model = CCRNN()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        data = data/11.0
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(self.model, 'rnn_model_' + str(epoch) + '.pkl')


if __name__ == '__main__':
    for epoch in range(0, NUM_EPOCHS):
        train(epoch)