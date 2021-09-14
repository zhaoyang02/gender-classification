from PIL import Image
import torch
from torchvision import transforms
import os
import numpy as np
from tqdm import tqdm

data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

class Dataload:
    images = []
    labels = []
    labels_onehot = []

    dataPath = 'Dataset/'

    train_images = []
    train_labels = []
    train_labels_onehot = []
    
    validation_images = []
    validation_labels = []
    validation_labels_onehot = []

    test_images = []
    test_labels = []
    test_labels_onehot = []
    
    dataCount = 0

    #����LFW����
    def LoadLFWData(self):
        for gender in os.listdir('Dataset/'):
            self.dataCount=self.dataCount+len(os.listdir('Dataset/'+gender))         #��¼��������
            print(f"has loaded {len(os.listdir('Dataset/'+gender))} data of {gender}")   #����Ѿ����ص����������Ա�
            for picFile in tqdm(os.listdir('Dataset/'+gender)):                      #��tqdm��ʾ�������
                picPath=self.dataPath + gender +'/'+ picFile                         #��ȡͼƬ��ַ
                image = self.loadPicArray(picPath)
                label = 1 if gender == 'male' else 0                                 #���Ա�ǩ��Ϊ1��Ů����Ϊ0
                self.images.append(image)
                self.labels.append(label)
                self.labels_onehot = np.eye(2)[self.labels]                          #�����������ݵı�ǩֱֵ�ӵõ��������ݵı�ǩ��onehot��ʽ
        print("Total data count: ",self.dataCount)

        #�������ݣ�ʹ����ͬ�Ĵ������images��labels��labels_onehot����֤������Ȼ��Ӧ
        state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(state)
        np.random.shuffle(self.labels)
        np.random.set_state(state)
        np.random.shuffle(self.labels_onehot)

        #�������и����ݣ���Ϊѵ��������֤���Ͳ��Լ�
        trainIndex = int(self.dataCount * 0.4)
        validationIndex = int(self.dataCount * 0.5)
        self.train_images = self.images[0 : trainIndex]
        self.train_labels = self.labels[0 : trainIndex]
        self.train_labels_onehot = self.labels_onehot[0 : trainIndex]
        self.validation_images = self.images[trainIndex : validationIndex]
        self.validation_labels = self.labels[trainIndex : validationIndex]
        self.validation_labels_onehot = self.labels_onehot[trainIndex : validationIndex]
        self.test_images = self.images[validationIndex : ]
        self.test_labels = self.labels[validationIndex : ]
        self.test_labels_onehot = self.labels_onehot[validationIndex : ]


    #��ȡͼƬ���ݣ��õ�ͼƬ��Ӧ������ֵ�����飬��һ����0-1֮ǰ
    def loadPicArray(self, picFilePath):
        picData = Image.open(picFilePath)
        picArray = data_transform(picData)         #ת��ͼƬ��Ϣ
        return picArray

    def getTrainData(self):
        return self.train_images, self.train_labels, self.train_labels_onehot

    def getValidationData(self):
        return self.validation_images, self.validation_labels, self.validation_labels_onehot

    def getTestData(self):
        return self.test_images, self.test_labels, self.test_labels_onehot



