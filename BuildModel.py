import torch
import torch.nn as nn
from torchvision.models import resnet101, resnet50


class Resnet50(nn.Module):
    def __init__(self, num_classes, embedding_vector):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.embedding_vector = embedding_vector
        self.num_classes = num_classes
        for param in self.model.parameters():
            param.requires_grad_(False)
        top_head = self.create_head(num_features , self.num_classes, self.embedding_vector)
        self.model.fc = top_head
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        
    def create_head(self, num_features, number_classes, embedding_vector, dropout_prob=0.5 ,activation_func =nn.ReLU):
        features_lst = [num_features ,embedding_vector]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)


    def get_model(self):
        return self.model

class Resnet101(nn.Module):
    def __init__(self, num_classes, embedding_vector):
        super().__init__()
        self.model = resnet101(pretrained=True)
        self.num_features = self.model.fc.in_features
        self.embedding_vector = embedding_vector
        self.num_classes = num_classes
        for param in self.model.parameters():
            param.requires_grad_(False)
        top_head = self.create_head(self.num_features , self.num_classes, self.embedding_vector)
        self.model.fc = top_head
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
        
    def create_head(self, num_features, number_classes, embedding_vector, dropout_prob=0.5 ,activation_func =nn.ReLU):
        features_lst = [num_features , num_features//2 , num_features//4, embedding_vector]
        layers = []
        for in_f ,out_f in zip(features_lst[:-1] , features_lst[1:]):
            layers.append(nn.Linear(in_f , out_f))
            layers.append(activation_func())
            layers.append(nn.BatchNorm1d(out_f))
            if dropout_prob !=0 : layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.Linear(features_lst[-1] , number_classes))
        return nn.Sequential(*layers)


    def get_model(self):
        return self.model
        
    