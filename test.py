import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pointcloud_dataloader import ModelNet40
from h5_dataloader import ModelNet40 as h5_ModelNet40
import sklearn.metrics as metrics
from models.pointnet_classifier import PointNetClassifier
from models.DGCNN import DGCNN
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import argparse
from tqdm import tqdm

def draw_Point_Cloud(Points, Lables=None, axis=True, **kags):
    from mpl_toolkits.mplot3d import Axes3D
    x_axis = Points[:, 0]
    y_axis = Points[:, 1]
    z_axis = Points[:, 2]
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x_axis, y_axis, z_axis, c=Lables)
    # 设置坐标轴显示以及旋转角度
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=10, azim=235)
    if not axis:
        # 关闭显示坐标轴
        plt.axis('off')

    plt.show()


if __name__ == '__main__':

    # Testing settings
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


    Test_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')

    class_names=  ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain',
     'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio',
     'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    num = list(range(0, len(class_names)))
    class_num_dict = dict(zip(class_names, num))

    test_dataset = ModelNet40(Test_dataset_path)
    #test_dataset = h5_ModelNet40(2000, 'train')
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = 'checkpoints/snapshotfinal.params'
    num_points = 2000
    dims = 3
    class_num = 40

    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []

    # Keep track of the number of samples seen
    total_num_samples = 0
    class_num_samples = np.zeros(class_num)

    # Create length-40 arrays to track per class accuracy
    class_correct = np.zeros(class_num)
    class_incorrect = np.zeros(class_num)

    # Also keep track of total accuracy
    total_correct = 0
    total_incorrect = 0


    io = IOStream('checkpoints/' + '/eval.log')

    # Instantiate the network
    #classifier = PointNetClassifier(num_points, dims).eval().cuda().double()

    classifier = DGCNN(args).eval().cuda()
    classifier.load_state_dict(torch.load(model_path))
    classifier = torch.nn.DataParallel(classifier)
    classifier.eval()
    io.cprint('Starting Eval...')
    test_true = []
    test_pred = []
    for data, label in tqdm(test_dataloader, ascii= True):

        data, label = data.float().to(device), label.squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        pred = classifier(data)
        preds = pred.max(dim=1)[1]

        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

        #preds = logits.max(dim=1)[1]
    #     _, preds = torch.max(F.softmax(logits, dim=1), 1)
    #     test_true.append(label.cpu().numpy())
    #     test_pred.append(preds.detach().cpu().numpy())
    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)




        idx = preds.cpu().numpy()
        target = label.cpu().numpy()
        total_num_samples += len(target)
        for j in range(len(target)):
            val = target[j] == idx[j]
            total_correct += val
            class_correct[target[j]] += val
            total_incorrect += np.logical_not(val)
            class_incorrect[target[j]] += np.logical_not(val)
            class_num_samples[target[j]] += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)
    io.cprint('Total Accuracy: {:2f}'.format(total_correct/float(total_num_samples)))
    io.cprint('Per Class Accuracy:')
    for i in range(len(class_correct)):
        io.cprint('{}: {:2f}'.format(class_names[i],
                                 class_correct[i] / float(class_num_samples[i])))
    io.cprint('Done!')
    classification_report = metrics.classification_report(test_true, test_pred, target_names=class_names)

    io.cprint(classification_report)
    io.close()
