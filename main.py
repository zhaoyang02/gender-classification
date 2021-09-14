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

# ���ó�����
BATCH_SIZE = 10 # ����ǱʼǱ������ܻ����Կ��Դ��С�����Լ�С��ֵ
LR = 0.1 # ѧϰ��
MM = 0.9 # ����ݶ��½�����momentum����
EPOCH = 30 # ѵ������

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#���ܻ�ȡ������Ҳ��ʵ�����������������ȡĳ�������Ĺ���
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
    def __len__(self):                        #�����������
        return len(self.img)
    def __getitem__(self,i):                  #��ȡ����
        return self.img[i],self.lab[i]

dataLoad = Dataload()
dataLoad.LoadLFWData()

def get_data(dataType):
    return myDatasets(dataType,dataLoad)

#�������ݼ�
train_data = get_data("train")
validation_data = get_data("validation")
test_data = get_data("test")

# ����dataloader��pytorch�����������������Ҫͨ��dataloader��ʵ��
train_loader = DataLoader(
                    train_data, 
                    batch_size=BATCH_SIZE)

validation_loader = DataLoader(
                    validation_data, 
                    batch_size=BATCH_SIZE)

test_loader = DataLoader(
                    test_data, 
                    batch_size=BATCH_SIZE)

#��������ṹ
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
        x=torch.flatten(x,1)    #����x�ĵ�һ��ά�Ƚ���ƴ��
        x = self.fc1(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.normalize(x)
        x = self.fc3(x)
        return x
model = CNN().to(device)

# ������ʧ����������������ý�������ʧ����
loss_func = nn.CrossEntropyLoss()

# �����Ż��������˴�ʹ������ݶ��½���
optimizer_ft = optim.SGD(model.parameters(), lr=LR, momentum=MM)

# ����ÿ5��epoch��ѧϰ�ʱ�Ϊ֮ǰ��0.1 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

# ѵ�������� 
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

# ����������
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

#��֤������
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
    #�����֤���
    print(f"Validation accuracy: {(100*Accuracy):>0.1f}%\n")

#��ʼѵ�������Ժ���֤
since=time.time()
best_acc = 0
#��ʼѵ��
for epoch in range(EPOCH):
    print(f'Epoch {epoch}/{EPOCH - 1}')
    print('-' * 10)

    model = train_model(model, loss_func, optimizer_ft, exp_lr_scheduler)
    epoch_acc = test_model(model, loss_func)
#���ʱ��
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
#��ʼ��֤
validation(model)
#����ѵ���õ�ģ��Ȩ��
torch.save(model.state_dict(), "model.pt")
print("Saved PyTorch Model State to model.pt")
