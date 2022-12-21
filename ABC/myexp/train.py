import numpy as np
import torch
import pickle 
import os
from collections import OrderedDict

from argparse import Namespace
from sklearn.metrics import confusion_matrix

from myexp.amil import Attention_MIL
from myexp.co_attn import ABC
from myexp.utils import *


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    # loss_fn = nn.BCELoss()
    loss_fn = nn.CrossEntropyLoss()

    print('Done!')
    
    print('\nInit Model...', end=' ')
    # model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion
    
    print('omic_input_dim:', args.omic_input_dim)

    if args.model_type =='amil':
        model_dict = {'omic_input_dim': args.omic_input_dim, 
                     'fusion': args.fusion,
                     'n_classes': args.n_classes}
        model = Attention_MIL(**model_dict)
    elif args.model_type =='abc':
        model = ABC(fusion='concat', 
                    omic_sizes=args.omic_input_dim,
                    n_classes=args.n_classes,   # should be 6
                    )
    else:
        raise NotImplementedError
    
    model = model.to('cuda')

    print('Done!')
    # print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    # train_loader = get_split_loader(train_split, mode=args.mode, batch_size=args.batch_size)
    # val_loader = get_split_loader(val_split,  mode=args.mode, batch_size=args.batch_size)
    train_loader, val_loader = get_split_loader(train_split, val_split,
                                                mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    result = []
    if args.mode == 'coattn':
        for epoch in range(args.max_epochs):
            train_loop_coattn(epoch, model, train_loader, optimizer,
                                        args.n_classes, loss_fn)
            loss = validate_coattn(epoch, model, val_loader, loss_fn)
            # print(loss)
            if epoch % 5 == 0:
                result.append(loss)
                torch.save(model.state_dict(), os.path.join(args.results_dir, "{}_{}_checkpoint{}.pt".format(args.model_type, cur, epoch)))
    elif args.mode == 'path':
        for epoch in range(args.max_epochs):
            train_loop_path(epoch, model, train_loader, optimizer,
                                        args.n_classes, loss_fn)
            loss = validate_path(epoch, model, val_loader, loss_fn)
            # print(loss)
            if epoch % 5 == 0:
                result.append(loss)
                torch.save(model.state_dict(), os.path.join(args.results_dir, "{}_{}_checkpoint{}.pt".format(args.model_type, cur, epoch)))
    elif args.mode == 'pathomic':
        for epoch in range(args.max_epochs):
            train_loop_path(epoch, model, train_loader, optimizer,
                                        args.n_classes, loss_fn)
            loss = validate_path(epoch, model, val_loader, loss_fn)
            # print(loss)
            if epoch % 5 == 0:
                result.append(loss)
                torch.save(model.state_dict(), os.path.join(args.results_dir, "{}_{}_checkpoint{}.pt".format(args.model_type, cur, epoch)))
    
    print(result)
    with open(os.path.join(args.results_dir, "result.txt"), 'w') as fp:
        fp.write('\n'.join('%i %i' % x for x in result))




def train_loop_coattn(epoch, model, dataloader, optimizer,
                    n_classes, loss_fn):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_cls, train_loss = 0., 0.

    effective_slides = 0
    correct = 0
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3,
                    data_omic4, data_omic5, data_omic6, label) in enumerate(dataloader):
        if label == -1:
            continue
        effective_slides += 1

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        y = label.type(torch.LongTensor).to(device)
        label = label.to(device)
        if epoch == 29 and effective_slides == 421:
            pred_label, co_attn  = model(x_path=data_WSI, 
                                        x_omic1=data_omic1, 
                                        x_omic2=data_omic2, 
                                        x_omic3=data_omic3, 
                                        x_omic4=data_omic4, 
                                        x_omic5=data_omic5, 
                                        x_omic6=data_omic6)
            co_attn = co_attn.detach().cpu().numpy()
            np.save('co_attn.npy',co_attn)
            print("Co_attn saved!\n")
        else:
            pred_label, _  = model(x_path=data_WSI, 
                                        x_omic1=data_omic1, 
                                        x_omic2=data_omic2, 
                                        x_omic3=data_omic3, 
                                        x_omic4=data_omic4, 
                                        x_omic5=data_omic5, 
                                        x_omic6=data_omic6)


        # print("predict result:",pred_label, y)
        # label = torch.zeros_like(pred_label).to(device)
        # label[y] = 1
        # loss = loss_fn(pred_label, label)

        _, predicted = torch.max(pred_label, 1)
        correct += (predicted == label)

        loss = loss_fn(pred_label, y)

        loss_value = loss.item()

        train_loss_cls += loss_value

        if (batch_idx) % 100 == 0:
            # print(pred_label, label)
            print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value, label.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_cls /= effective_slides
    print('Epoch: {}, effective_slides: {}, train_loss_cls: {:.4f}'.format(epoch, effective_slides, train_loss_cls))
    print("Train Accuracy", correct / effective_slides, correct, effective_slides)



def validate_coattn(epoch, model, dataloader, loss_fn):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    effective_slides = 0
    correct = 0
    target = []
    pred = []
    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3,
                    data_omic4, data_omic5, data_omic6, label) in enumerate(dataloader):
        if label == -1:
            continue
        effective_slides += 1

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)

        target.append(int(label))

        y = label.type(torch.LongTensor).to(device)
        label = label.to(device)
        with torch.no_grad():
            pred_label, _  = model(x_path=data_WSI, 
                                        x_omic1=data_omic1, 
                                        x_omic2=data_omic2, 
                                        x_omic3=data_omic3, 
                                        x_omic4=data_omic4, 
                                        x_omic5=data_omic5, 
                                        x_omic6=data_omic6)

        # print("predict result:",pred_label, y)
        # label = torch.zeros_like(pred_label).to(device)
        # label[y] = 1
        # loss = loss_fn(pred_label, label)

        _, predicted = torch.max(pred_label, 1)
        # print(predicted == label, predicted, label)
        pred.append(int(predicted.detach().cpu()))

        correct += (predicted == label)

        if (batch_idx) % 10 == 0:
            loss = loss_fn(pred_label, y)
            print(loss)

    # calculate loss and error for epoch
    accuracy = correct / effective_slides
    # print(accuracy)
    print('Validation: {}, effective_slides: {}, accuracy: {} \n'.format(epoch, effective_slides, accuracy))
    if epoch == 29:
        cm = confusion_matrix(target, pred)
        np.save('confusion.npy', cm)

    return accuracy, correct



def train_loop_path(epoch, model, dataloader, optimizer,
                    n_classes, loss_fn):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_cls, train_loss = 0., 0.

    effective_slides = 0
    correct = 0
    for batch_idx, (data_WSI, data_omic, label) in enumerate(dataloader):
        if label == -1:
            continue
        effective_slides += 1
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y = label.type(torch.LongTensor).to(device)
        label = label.to(device)

        pred_label = model(x_path=data_WSI, x_omic=data_omic)

        _, predicted = torch.max(pred_label, 1)
        correct += (predicted == label)

        loss = loss_fn(pred_label, y)
        loss_value = loss.item()

        train_loss_cls += loss_value

        if (batch_idx) % 100 == 0:
            # print(pred_label, label)
            print('batch {}, loss: {:.4f}, label: {}'.format(batch_idx, loss_value, label.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_cls /= effective_slides
    print('Epoch: {}, effective_slides: {}, train_loss_cls: {:.4f}, train_loss: {:.4f}'.format(epoch, effective_slides,
                                                                                                train_loss_cls))
    print("Train Accuracy", correct / effective_slides, correct, effective_slides)


def validate_path(epoch, model, dataloader, loss_fn):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.eval()

    effective_slides = 0
    correct = 0
    for batch_idx, (data_WSI, data_omic, label) in enumerate(dataloader):
        if label == -1:
            continue
        effective_slides += 1
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y = label.type(torch.LongTensor).to(device)
        label = label.to(device)

        with torch.no_grad():
            pred_label = model(x_path=data_WSI, x_omic=data_omic)

        _, predicted = torch.max(pred_label, 1)
        # print(predicted == label, predicted, label)
        correct += (predicted == label)

        if (batch_idx) % 10 == 0:
            loss = loss_fn(pred_label, y)
            print(loss)

    # calculate loss and error for epoch
    accuracy = correct / effective_slides
    # print(accuracy)
    print('Validation: {}, effective_slides: {}, accuracy: {} \n'.format(epoch, effective_slides, accuracy))

    return accuracy, correct
