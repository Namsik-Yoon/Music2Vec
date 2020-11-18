import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import Dataset
import BuildModel


def select_optimizer(model, args):
    lr = args['lr']
    beta1 = args['beta1']
    beta2 = args['beta2']
    eps = args['eps']
    weight_decay = args['weight_decay']
    amsgrad = args['amsgrad']

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2),
                                 eps=eps, weight_decay=weight_decay,
                                 amsgrad=amsgrad)

    return optimizer


def select_scheduler(optimizer, config):
    T_max = config['T_max']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return scheduler

def train(model, loader, criterion, optimizer):
    model.cuda()
    model.train()
    train_losses = []

    targets = []
    outputs = []
    train_cnt = 0
    for idx,data in enumerate(loader):
        print(idx)
        x, y = data
        x ,y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach())
        train_cnt += len(data)

        targets.append(y.detach())
        outputs.append(y_pred.detach())
    return model, train_losses/(idx+1), f1_score(outputs, targets, average='samples'), (targets==outputs).sum()/train_cnt


def evaluate(model, loader, criterion):
    model.cuda()
    model.eval()
    eval_losses = []

    targets = []
    outputs = []
    eval_cnt = 0
    for idx, data in enumerate(loader):
        x, y = data
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)

        loss = criterion(y_pred, y)

        eval_losses.append(loss.detach())
        eval_cnt += len(data)

        targets.append(y.detach())
        outputs.append(y_pred.detach())

    return model, eval_losses / (idx + 1), f1_score(outputs, targets, average='samples'), (targets == outputs).sum() / eval_cnt




if __name__ == '__main__':
    args: object = json.load(open("config.json"))

    ## DATASET
    if args['augmentation']:
        DataSet = Dataset.AugDset()
    else:
        DataSet = Dataset.DSet()

    ## DATALOADER
    ratio = [int(len(DataSet) * args['train_ratio']), len(DataSet) - int(len(DataSet) * args['train_ratio'])]
    train_set, val_set = torch.utils.data.random_split(DataSet, ratio)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=4)

    ## MODEL
    num_classes = len(os.listdir('data/genres'))
    model = BuildModel.resnet101(num_classes)

    ## CRITERION & OPTIMIZER & SECHEDULER
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(model, args['Optim'])
    scheduler = select_scheduler(optimizer, args['Scheduler'])

    ## RUN
    for i in range(args['Scheduler']['T_max']):
        model, train_loss, train_f1, train_acc = train(model, loader=train_loader, criterion=criterion, optimizer=optimizer)
        model, val_loss, val_f1, val_acc = train(model, loader=val_loader, criterion=criterion)
        print(f'epoch {i+1} train : train_loss = {train_loss}, train_f1 = {train_f1}, train_acc = {train_acc}')
        print(f'epoch {i+1} validation : val_loss = {val_loss}, val_f1 = {val_f1}, val_acc = {val_acc}')
        print('-'*50)