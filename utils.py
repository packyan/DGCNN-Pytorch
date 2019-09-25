import torch
import os
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def save_model(model, snapshot_dir, ep):
    save_path = os.path.join(snapshot_dir, 'snapshot{}.params' \
                             .format(ep))
    torch.save(model.state_dict(), save_path)

def normalize_point_cloud(points):

    scaler = torch.max(points,dim=0).values - torch.min(points,dim=0).values
    #“扁平结构比嵌套结构更好” – 《Python之禅》
    scaler =torch.tensor(list(map(lambda  x : 1e-5 if x < 1e-5 else x, scaler)))
    points = (points - torch.min(points,dim=0).values) / scaler
    points_mean = torch.mean(points, dim = 0)
    return points - torch.mean(points, dim = 0)

def draw_Point_Cloud(Points, Lables=None, axis=True, **kags):

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