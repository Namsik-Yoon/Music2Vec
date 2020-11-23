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
    train_corr = 0
    for idx, data in enumerate(loader):
        x, y = data
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_pred = model(x.float())
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.detach())
        train_cnt += len(x)

        train_corr += (y == torch.argmax(y_pred, 1)).sum()
        targets += torch.argmax(y_pred, 1).tolist()
        outputs += y.tolist()
    return model, sum(train_losses)/(idx + 1), f1_score(outputs, targets, average='macro'), train_corr/train_cnt


def evaluate(model, loader, criterion):
    model.cuda()
    model.eval()
    eval_losses = []

    targets = []
    outputs = []
    eval_cnt = 0
    eval_corr = 0
    for idx, data in enumerate(loader):
        x, y = data
        x, y = x.cuda(), y.cuda()
        y_pred = model(x.float())

        loss = criterion(y_pred, y)

        eval_losses.append(loss.detach())
        eval_cnt += len(x)

        eval_corr += (y == torch.argmax(y_pred, 1)).sum()
        targets += torch.argmax(y_pred, 1).tolist()
        outputs += y.tolist()
    return model, sum(eval_losses) / (idx + 1), f1_score(outputs, targets, average='macro'), eval_corr/ eval_cnt


def run(embedding_vector, verbose=True, early_stopping=True):
    args = json.load(open("config.json"))
    print('Preparing Train.............')
    ## DATASET
    if 'mfcc_data' not in os.listdir():
        Dataset.save_aug_tensor()
    if args['augmentation']:
        DataSet = Dataset.AugDset(args)
    else:
        DataSet = Dataset.DSet(args)
        
    ## DATALOADER
    ratio = [int(len(DataSet) * args['train_ratio']), len(DataSet) - int(len(DataSet) * args['train_ratio'])]
    train_set, val_set = torch.utils.data.random_split(DataSet, ratio)
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=2, drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=2)
    
    ## MODEL
    num_classes = len(os.listdir('data/genres'))
    model = BuildModel.Resnet101(num_classes, embedding_vector).get_model()

    ## CRITERION & OPTIMIZER & SECHEDULER
    criterion = nn.CrossEntropyLoss()
    optimizer = select_optimizer(model, args['Optim'])
    scheduler = select_scheduler(optimizer, args['Scheduler'])
    
    print('Training Start..............')
    history = {'train_losses':[],'train_f1s':[],'train_accs':[],'eval_losses':[],'eval_f1s':[],'eval_accs':[]}
    ## RUN
    for i in range(args['Scheduler']['T_max']):
        model, train_loss, train_f1, train_acc = train(model, loader=train_loader, criterion=criterion, optimizer=optimizer)
        model, eval_loss, eval_f1, eval_acc = evaluate(model, loader=val_loader, criterion=criterion)
        scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['train_f1s'].append(train_f1)
        history['train_accs'].append(train_acc)
        history['eval_losses'].append(eval_loss)
        history['eval_f1s'].append(eval_f1)
        history['eval_accs'].append(eval_acc)
        if verbose:
            print(f'epoch {i+1} train : train_loss = {train_loss:.4f}, train_f1 = {train_f1:.4f}, train_acc = {train_acc:.4f}')
            print(f'epoch {i+1} validation : val_loss = {eval_loss:.4f}, val_f1 = {eval_f1:.4f}, val_acc = {eval_acc:.4f}')
            print('-'*50)
        if early_stopping:
            current_eval_loss = eval_loss
            minimum_eval_loss = min(history['eval_losses'])
            if current_eval_loss > minimum_eval_loss:
                patience += 1
                if patience > 20:
                    print(f'early stop at best epoch {best_epoch}')
                    break
            else:
                best_model = model
                best_epoch = i
                patience = 0
    return best_model, history
        
if __name__ == '__main__':
    embedding_vector = int(input())
    model,history = run(embedding_vector)