import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import h5py
from torch.utils.data import DataLoader, Dataset
from pointcloud_dataloader import ModelNet40
from h5_dataloader import ModelNet40 as ModelNet40h5
from models.pointnet_classifier import PointNetClassifier
import time
from utils import *
from models.DGCNN import PointNet
import argparse
import sklearn.metrics as metrics
from tqdm import tqdm

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    Train_dataset_path = os.path.join('Data',  'ModelNet40_', 'ModelNet40_train.h5')
    Test_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')

    train_dataset = ModelNet40(Train_dataset_path)
    #train_dataset =ModelNet40h5(2000,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    io = IOStream('checkpoints/' + '/train.log')
    # Parameters
    Epochs = 80
    reg_weight = 0.001
    snapshot_dir = 'checkpoints'
    snapshot = 5

    # Some timers and a counter
    forward_time = 0.
    backprop_time = 0.
    network_time = 0.
    batch_counter = 0
    printout = 20
    # Whether to save a snapshot
    save = False

    # Model Definetion
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('use %s' % device)
    model = PointNet(args).to(device)

    # model = PointNetClassifier(num_points=2000, K=3, class_num=40).to(device).double()
    #model = torch.nn.DataParallel(model)
    # optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # loss function
    loss = nn.CrossEntropyLoss()
    criterion = cal_loss
    regularization = nn.MSELoss()
    identity = torch.nn.Parameter(
        torch.eye(64, requires_grad=True, dtype=torch.double).to(device))

    for epoch in range(Epochs):
        if epoch % snapshot == 0 and epoch != 0:
            save = True
        print('Epoch {}'.format(epoch))

        model.train()
        train_loss = 0.0
        count = 0.0
        scheduler.step()
        train_pred = []
        train_true = []
        i = 0
        for (data, label) in tqdm(train_dataloader, ascii=True):
            i+=1
            # try:
            #     assert np.any(np.isnan(data.numpy())) == False
            # except AssertionError as e:
            #     print(e)
            #     print(label)
            #     draw_Point_Cloud(data[0])
            #     draw_Point_Cloud(data[1])
            #     raise
            # assert np.any(np.isnan(label.numpy())) == False
            batch_size = data.size()[0]
            optimizer.zero_grad()

            # Record starting time
            start_time = time.time()

            # conv input is : batch x dim x data_size
            logits = model(data.permute(0, 2, 1).float().to(device))
            # Compute forward pass time
            forward_finish = time.time()
            forward_time += forward_finish - start_time
            # print(y_pred.shape,y_pred.dtype)
            # print(label.shape, label.dtype)
            pred_loss = criterion(logits, label.long().to(device))
            pred_loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += pred_loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss * 1.0 / count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        if save:
            print('Saving model snapshot...')
            save_model(model, snapshot_dir, epoch)
            save = False

            # # print("pred_loss {}  regularization_loss {} total_error {}".format(
            # #     pred_loss, regularization_loss, total_error
            # # ))
            # # Compute backprop time
            # backprop_finish = time.time()
            # backprop_time += backprop_finish - forward_finish
            #
            # # Compute network time
            # network_finish = time.time()
            # network_time += network_finish - start_time
            #
            # # Increment batch counter
            # batch_counter += 1

            # # ------------------------------------------------------------------
            # # Print feedback
            # # ------------------------------------------------------------------
            #
            # if (i + 1) % printout == 0:
            #     # # vis
            #     # vis.plot('Total error', total_error.item())
            #     # vis.plot('Pred  error', pred_error.item())
            #     # vis.plot('Reg error', reg_error.item())
            #     # loss_data = "%.5f\t%.5f\t%.5f\n" % (total_error.item(), pred_error.item(), reg_error.item())
            #     # loss_data_file.flush()
            #     # loss_data_file.write(loss_data)
            #
            #     # Print progress
            #     io.cprint('Epoch {}/{}'.format(epoch + 1, Epochs))
            #
            #     # Print network speed
            #     io.cprint('{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop'))
            #     io.cprint('  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
            #           .format(network_time, forward_time, backprop_time))
            #
            #     # Print current error
            #     io.cprint('{:16}[ {:12}{:12} ]'.format('Total Error',
            #                                        'Pred Error', 'Reg Error'))
            #     io.cprint('  {:<14.4f}[   {:<10.4f}  {:<10.4f} ]'.format(
            #         total_error.item(), pred_loss.item(), regularization_loss.item()))
            #     io.cprint('\n')
            #
            #     # Reset timers
            #     forward_time = 0.
            #     backprop_time = 0.
            #     network_time = 0.

    print('Saving model snapshot final round...')
    save_model(model, snapshot_dir, 'final')
