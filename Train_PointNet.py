import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import h5py
from torch.utils.data import DataLoader, Dataset
from pointcloud_dataloader import ModelNet40
#from h5_dataloader import ModelNet40
from models.pointnet_classifier import PointNetClassifier
import time
from utils import *
from models.DGCNN import PointNet
import argparse
import sklearn.metrics as metrics

if __name__ == '__main__':
    # Training settings

    Train_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_train.h5')
    Test_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')

    train_dataset = ModelNet40(Train_dataset_path)
    #train_dataset =ModelNet40(2000,'train')
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
    #model = PointNet(args)


    model = PointNetClassifier(num_points=2000, K=3, class_num=40).to(device).double()
    model = torch.nn.DataParallel(model)
    # optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # loss function
    loss = nn.CrossEntropyLoss()
    ##criterion = cal_loss
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

        for i, (data, label )in enumerate(train_dataloader):

            batch_size = data.size()[0]
            optimizer.zero_grad()

            # Record starting time
            start_time = time.time()

            # conv input is : batch x dim x data_size
            logits, T2 = model(data.permute(0, 2, 1).to(device))
            # Compute forward pass time
            forward_finish = time.time()
            forward_time += forward_finish - start_time
            # print(y_pred.shape,y_pred.dtype)
            # print(label.shape, label.dtype)
            pred_loss  = loss(logits, label.long().to(device))
            # Also enforce orthogonality in the embedded transform
            reg_error = regularization(
                torch.bmm(T2, T2.permute(0, 2, 1)),
                identity.expand(T2.shape[0], -1, -1))

            # Total error is the weighted sum of the prediction error and the
            # regularization error
            total_error = pred_loss + reg_weight * reg_error

            # Backpropagate
            total_error.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += pred_loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            # Print FeedBack

            # print("pred_loss {}  regularization_loss {} total_error {}".format(
            #     pred_loss, regularization_loss, total_error
            # ))
            # Compute backprop time
            backprop_finish = time.time()
            backprop_time += backprop_finish - forward_finish

            # Compute network time
            network_finish = time.time()
            network_time += network_finish - start_time

            # Increment batch counter
            batch_counter += 1

            # ------------------------------------------------------------------
            # Print feedback
            # ------------------------------------------------------------------

            if (i + 1) % printout == 0:
                # # vis
                # vis.plot('Total error', total_error.item())
                # vis.plot('Pred  error', pred_error.item())
                # vis.plot('Reg error', reg_error.item())
                # loss_data = "%.5f\t%.5f\t%.5f\n" % (total_error.item(), pred_error.item(), reg_error.item())
                # loss_data_file.flush()
                # loss_data_file.write(loss_data)

                # Print progress
                io.cprint('Epoch {}/{}'.format(epoch + 1, Epochs))

                # Print network speed
                io.cprint('{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop'))
                io.cprint('  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
                          .format(network_time, forward_time, backprop_time))

                # Print current error
                io.cprint('{:16}[ {:12}{:12} ]'.format('Total Error',
                                                       'Pred Error', 'Reg Error'))
                io.cprint('  {:<14.4f}[   {:<10.4f}  {:<10.4f} ]'.format(
                    total_error.item(), pred_loss.item(), reg_error.item()))
                io.cprint('\n')

                # Reset timers
                forward_time = 0.
                backprop_time = 0.
                network_time = 0.

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






    print('Saving model snapshot final round...')
    save_model(model, snapshot_dir, 'final')

