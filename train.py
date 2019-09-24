import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import h5py
from torch.utils.data import DataLoader, Dataset
from pointcloud_dataloader import ModelNet40
from models.pointnet_classifier import PointNetClassifier
import time

def save_model(model, snapshot_dir, ep):
    save_path = os.path.join(snapshot_dir, 'snapshot{}.params' \
                             .format(ep))
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':

    Train_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_train.h5')
    Test_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')

    train_dataset = ModelNet40(Train_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)

    # Parameters
    Epochs = 60
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

    model = PointNetClassifier(num_points=2000, K=3, class_num=40).to(device).double()

    # optim
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # loss function
    loss = nn.CrossEntropyLoss()
    regularization = nn.MSELoss()
    identity = torch.nn.Parameter(
        torch.eye(64, requires_grad=True, dtype=torch.double).to(device))

    model.train()

    for epoch in range(Epochs):
        if epoch % snapshot == 0 and epoch != 0:
            save = True
        print('Epoch {}'.format(epoch))

        scheduler.step()

        for i, (data, label )in enumerate(train_dataloader):

            optimizer.zero_grad()

            # Record starting time
            start_time = time.time()

            # conv input is : batch x dim x data_size
            y_pred, T2 = model(data.permute(0, 2, 1).to(device))
            # Compute forward pass time
            forward_finish = time.time()
            forward_time += forward_finish - start_time


            pred_loss = loss(y_pred, label.to(device).long())
            regularization_loss = regularization(torch.bmm(T2, T2.permute(0, 2, 1)),
                                                 identity.expand(T2.shape[0], -1, -1))
            total_error = pred_loss + reg_weight * regularization_loss
            total_error.backward()
            optimizer.step()
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
                print('Epoch {}/{}'.format(epoch + 1, Epochs))

                # Print network speed
                print('{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop'))
                print('  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
                      .format(network_time, forward_time, backprop_time))

                # Print current error
                print('{:16}[ {:12}{:12} ]'.format('Total Error',
                                                   'Pred Error', 'Reg Error'))
                print('  {:<14.4f}[   {:<10.4f}  {:<10.4f} ]'.format(
                    total_error.item(), pred_loss.item(), regularization_loss.item()))
                print('\n')

                # Reset timers
                forward_time = 0.
                backprop_time = 0.
                network_time = 0.

            if save:
                print('Saving model snapshot...')
                save_model(model, snapshot_dir, epoch)
                save = False

        print('Saving model snapshot final round...')
        save_model(model, snapshot_dir, 'final')

