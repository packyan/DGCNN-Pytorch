import torch
import os
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pointcloud_dataloader import ModelNet40
from h5_dataloader import ModelNet40 as h5_ModelNet40
import sklearn.metrics as metrics
from models.pointnet_classifier import PointNetClassifier
import numpy as np
import matplotlib.pyplot as plt
from utils import *



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
    #Test_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')

    #test_dataset = ModelNet40(Test_dataset_path)
    test_dataset = h5_ModelNet40(2000, 'train')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = 'checkpoints/snapshot30.params'
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


    io = IOStream('checkpoints/' + '/run.log')

    # Instantiate the network
    classifier = PointNetClassifier(num_points, dims).eval().cuda().double()
    classifier = torch.nn.DataParallel(classifier)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()

    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device).squeeze()

        # points = data[0]
        # print(label[0])
        # draw_Point_Cloud(points.cpu().numpy())
        # points = data[1]
        # print(label[1])
        # draw_Point_Cloud(points.cpu().numpy())
        # break
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        pred,_ = classifier(data.double())
        #preds = logits.max(dim=1)[1]
    #     _, preds = torch.max(F.softmax(logits, dim=1), 1)
    #     test_true.append(label.cpu().numpy())
    #     test_pred.append(preds.detach().cpu().numpy())
    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)
    # test_acc = metrics.accuracy_score(test_true, test_pred)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    # outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    # io.cprint(outstr)
        _, idx = torch.max(F.softmax(pred, dim=1), 1)

        idx = idx.cpu().numpy()
        target = target.cpu().numpy()
        total_num_samples += len(target)
        for j in range(len(target)):
            val = target[j] == idx[j]
            total_correct += val
            class_correct[target[j]] += val
            total_incorrect += np.logical_not(val)
            class_incorrect[target[j]] += np.logical_not(val)
            class_num_samples[target[j]] += 1

    io.cprint('Done!')
    io.cprint('Total Accuracy: {:2f}'.format(total_correct/float(total_num_samples)))
    io.cprint('Per Class Accuracy:')
    for i in range(len(class_correct)):
        io.cprint('{}: {:2f}'.format(i,
                                 class_correct[i] / float(class_num_samples[i])))
