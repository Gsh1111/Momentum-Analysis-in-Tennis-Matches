import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import datetime
import time
import itertools
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


lr        = 10  # 学习率
epochs    = 1000  # 训练轮数
n_feature = 20    # 输入特征(20个特征)
n_hidden  = 128   # 隐含层
n_output  = 1     # 输出(二分类)

# 1.准备数据
data = pd.read_excel('相关性分析.xlsx')  # 下载数据集

data['elapsed_time'] = pd.to_datetime(data['elapsed_time'], format='%H:%M:%S').dt.hour * 60 + pd.to_datetime(data['elapsed_time'], format='%H:%M:%S').dt.minute

features = data[['elapsed_time', 'set_no', 'game_no', 'server', 'serve_no', 'set_flag',
                  'p1_ACE', 'p2_ace', 'p1_W', 'p2_winner', 'p1_DF', 'p2_DF', 'p1_UE', 'p2_UE',
                  'p1_TPS', 'p2_three_points_streak', 'p1_NP', 'p2_NP', 'p1_NPW', 'p2_NPW']]

data['GF'] = data['GF'].replace(2,0)
target = data[['GF']]

# 数据归一化
# scaler = MinMaxScaler()
# features = scaler.fit_transform(features)

# 设置训练集数据80%，测试集20%
x_train, x_test, y_train, y_test = train_test_split(features.values, target.values, test_size=0.2, random_state=22)

# 将数据类型转换为tensor方便pytorch使用
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

#2.定义BP神经网络
# class BPNetModel(torch.nn.Module):
#     def __init__(self,n_feature,n_hidden,n_output):
#         super(BPNetModel, self).__init__()
#         self.hiddden=torch.nn.Linear(n_feature,n_hidden)#定义隐层网络
#         self.out=torch.nn.Linear(n_hidden,n_output)#定义输出层网络
#     def forward(self,x):
#         x = F.relu(self.hiddden(x))        #隐层激活函数采用relu()函数
#         out = torch.sigmoid(self.out(x))
#         # out = F.softmax(self.out(x),dim=1) #输出层采用softmax函数
#         return out

# 定义一个名为NeuralNetwork的类，它继承了PyTorch框架的nn.Module类，用于创建神经网络。
# class NeuralNetwork(nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output, embedding_dim=10):
#         super(NeuralNetwork, self).__init__()
#         self.embedding = nn.Embedding(2, embedding_dim)  # 二元变量的 Embedding 层
#         self.flatten = nn.Flatten()
#         self.hidden1 = nn.Linear(n_feature, n_hidden)
#         self.bn1 = nn.BatchNorm1d(n_hidden)  # Batch Normalization
#         # self.hidden2 = nn.Linear(n_hidden, n_hidden)
#         # self.bn2 = nn.BatchNorm1d(n_hidden)  # Batch Normalization
#         self.hidden2 = nn.Linear(n_hidden, n_hidden//2)
#         self.bn2 = nn.BatchNorm1d(n_hidden//2)  # Batch Normalization
#         self.out = nn.Linear(n_hidden//2, n_output)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.hidden1(x)
#         x = self.bn1(x)  # Batch Normalization
#         x = F.relu(x)
#         x = self.hidden2(x)
#         x = self.bn2(x)  # Batch Normalization
#         # x = F.relu(x)
#         # x = self.hidden3(x)
#         # x = self.bn3(x)  # Batch Normalization
#         x = F.relu(x)
#         x = self.out(x)
#         x = torch.sigmoid(x)
#         return x

# model=NeuralNetwork().cuda()

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(NeuralNetwork, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:,-1,:]  # 取最后一个时间步的输出
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output

# # model.parameters()用于获取模型的所有参数。这些参数包括权重和偏差等
# # 它们都是Tensor类型的，是神经网络的重要组成部分
# optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

#3.定义优化器和损失函数
net = NeuralNetwork(input_size=n_feature, hidden_size=n_hidden, output_size=n_output) #调用网络
optimizer= torch.optim.SGD(net.parameters(),lr=lr) #使用Adam优化器，并设置学习率
# loss_fun=torch.nn.CrossEntropyLoss() #对于多分类一般使用交叉熵损失函数
loss_fun=torch.nn.BCELoss()            #对于二分类一般使用交叉熵损失函数

#4.训练数据
loss_steps=np.zeros(epochs) #构造一个array([ 0., 0., 0., 0., 0.])里面有epochs个0
accuracy_steps=np.zeros(epochs)
y_train = y_train.view(-1).float()

for epoch in range(epochs):
    y_pred=net(x_train) #前向传播
    y_pred=y_pred.view(-1)
    loss=loss_fun(y_pred,y_train)#预测值和真实值对比
    optimizer.zero_grad() #梯度清零
    loss.backward() #反向传播
    optimizer.step() #更新梯度
    loss_steps[epoch]=loss.item()#保存loss
    running_loss = loss.item()

    if epoch % 100 == 0:
        print(f"第{epoch}次训练，loss={running_loss}".format(epoch,running_loss))
        with torch.no_grad(): #下面是没有梯度的计算,主要是测试集使用，不需要再计算梯度了
            y_pred=net(x_test)
            correct=(torch.argmax(y_pred,dim=1)==y_test).type(torch.FloatTensor)
            accuracy_steps[epoch]=correct.mean()
            print("预测准确率", accuracy_steps[epoch])

#5.绘制损失函数和精度
fig_name="Iris_dataset_classify_BPNet"
fontsize=15
fig,(ax1,ax2)=plt.subplots(2,figsize=(15,12), sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy",fontsize=fontsize)
ax1.set_title(fig_name,fontsize="xx-large")
ax2.plot(loss_steps)
ax2.set_ylabel("train loss",fontsize=fontsize)
ax2.set_xlabel("epochs",fontsize=fontsize)
plt.tight_layout()
plt.savefig(fig_name+'.png')
plt.show()
