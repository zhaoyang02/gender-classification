# coding=gbk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
from Dataloader import Dataload
from tqdm import tqdm

# 设置超参数
BATCH_SIZE = 10 # 如果是笔记本电脑跑或者显卡显存较小，可以减小此值
LR = 0.1 # 学习率
MM = 0.9 # 随机梯度下降法中momentum参数
EPOCH = 30 # 训练轮数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#既能获取样本，也能实现输出样本总数和提取某个样本的功能
class myDatasets(Dataset):
    def __init__(self,dataType,dataLoader) :
        if dataType == 'train':
            self.img,self.lab,_ = dataLoader.getTrainData()
        elif dataType == "validation":
            self.img,self.lab,_ = dataLoader.getValidationData()
        elif dataType == "test":
            self.img,self.lab,_ = dataLoader.getTestData()
        else:
            print('DataType is wrong!')
            exit()
    def __len__(self):                        #输出样本总数
        return len(self.img)
    def __getitem__(self,i):                  #提取样本
        return self.img[i],self.lab[i]

dataLoad = Dataload()
dataLoad.LoadLFWData()

def get_data(dataType):
    return myDatasets(dataType,dataLoad)

#加载数据集
train_data = get_data("train")
validation_data = get_data("validation")
test_data = get_data("test")

# 构建dataloader，pytorch输入神经网络的数据需要通过dataloader来实现
train_loader = DataLoader(
                    train_data, 
                    batch_size=BATCH_SIZE)

validation_loader = DataLoader(
                    validation_data, 
                    batch_size=BATCH_SIZE)

test_loader = DataLoader(
                    test_data, 
                    batch_size=BATCH_SIZE)

#定义网络结构
class CNN(nn.Module):
    def __init__(self) :
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(3,6,5,1)
        self.conv2=nn.Conv2d(6,16,5,1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*53*53,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,2)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=F.relu(x)
        x=self.pool(x)
        x=F.normalize(x)
        x=torch.flatten(x,1)    #按照x的第一个维度进行拼接
        x = self.fc1(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc3(x)
        return x
model = CNN().to(device)

# 定义损失函数，分类问题采用交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

# 定义优化方法，此处使用随机梯度下降法
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)

# 定义每5个epoch，学习率变为之前的0.1 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# 训练神经网络 
def train_model(model, criterion, optimizer, scheduler):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    scheduler.step()

    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects.double() /len(train_data)

    print('train Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

    return model

# 测试神经网络
def test_model(model, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(test_data)
    epoch_acc = running_corrects.double() / len(test_data)

    print('test Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))
    return epoch_acc

#验证神经网络
def validation(model):
    model.eval()
    Accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for inputs,labels in tqdm(validation_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs,1)

            correct += torch.sum(preds == labels.data)
    Accuracy = correct.double() / len(validation_data)
    #输出验证结果
    print(f"Validation accuracy: {(100*Accuracy):>0.1f}%\n")

#开始训练、测试和验证
since=time.time()
best_acc = 0
#开始训练
for epoch in range(EPOCH):
    print(f'Epoch {epoch}/{EPOCH - 1}')
    print('-' * 10)

    model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
    epoch_acc = test_model(model, loss_func)
#输出时间
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
#开始验证
validation(model)
#保存训练好的模型权重
torch.save(model.state_dict(), "model.pt")
print("Saved PyTorch Model State to model.pt")
