import torch
import numpy as np


class Data:
    def __init__(self, pos, face):
        self.pos = pos
        self.face = face
        # self.norm


class SamplePoints(object):
    r"""Uniformly samples :obj:`num` points on the mesh faces according to
    their face area.
    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
     From Pytorch-Geometric
    """

    def __init__(self, num, remove_faces=True, include_normals=False):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals

    def __call__(self, data):
        pos, face = data.pos, data.face
        assert pos.size(1) == 3 and face.size(0) == 3

        pos_max = pos.max()
        pos = pos / pos_max
        # print('pos / pos_max {}'.format(pos.shape))

        # area 求空间中三个点构成三角形的面积
        area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
        area = area.norm(p=2, dim=1).abs() / 2
        # print('area  norm {}'.format(area.shape))

        # 根据面积占比，获得哪个区域idx的概率比较大，然后根据这个概率随机抽样

        prob = area / area.sum()
        # print('prob = area / area.sum()  norm {}'.format(prob.shape))

        # 作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。
        # 输入是一个input张量，一个取样数量，和一个布尔值replacement。

        sample = torch.multinomial(prob, self.num, replacement=True)
        # print('sample : {}'.format(sample.shape))
        # print(sample[:100])
        # sample 为抽样得到的idx

        face = face[:, sample]

        # 取出抽样得到的face

        frac = torch.rand(self.num, 2, device=pos.device)

        # print('frac shape : {}'.format(frac.shape))

        mask = frac.sum(dim=-1) > 1
        # print('frac mask shape {}'.format(mask.shape))

        # 大小为2048的frac mask
        frac[mask] = 1 - frac[mask]

        vec1 = pos[face[1]] - pos[face[0]]
        vec2 = pos[face[2]] - pos[face[0]]

        if self.include_normals:
            data.norm = torch.nn.functional.normalize(vec1.cross(vec2), p=2)

        pos_sampled = pos[face[0]]
        # 把抽抽样得到的face 顶点第一个点作为pos_sampled
        # print('pos_sampled shape : {}'.format(pos_sampled.shape))
        # 随机的0,1 系数，
        pos_sampled += frac[:, :1] * vec1
        pos_sampled += frac[:, 1:] * vec2

        # 一点空间变换，让点位置更适合？ 没找到公式出处
        pos_sampled = pos_sampled * pos_max

        data.pos = pos_sampled

        if self.remove_faces:
            data.face = None

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.num)


def txt_PointsCloud_parser(path_to_off_file):
    # Read the OFF file
    with open(path_to_off_file, 'r') as f:
        contents = f.readlines()
    num_vertices = len(contents)
    # print(num_vertices)
    # Convert all the vertex lines to a list of lists
    vertex_list = [list(map(float, contents[i].strip().split(' '))) for i in list(range(0, num_vertices))]
    # Return the vertices as a 3 x N numpy array

    return np.array(vertex_list)


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None, device=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, device=None):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_txt_array(src, sep, start, end, dtype, device)


def read_off(path):
    with open(path, 'r') as f:
        src = f.read().split('\n')[:-1]
    return parse_off(src)


def parse_off(src):
    # Some files may contain a bug and do not have a carriage return after OFF.
    if src[0] == 'OFF':
        src = src[1:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = [int(item) for item in src[0].split()[:2]]

    pos = parse_txt_array(src[1:1 + num_nodes])

    face = src[1 + num_nodes:1 + num_nodes + num_faces]
    face = face_to_tri(face)

    data = Data(pos=pos, face=face)

    return data


def face_to_tri(face):
    face = [[int(x) for x in line.strip().split(' ')] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
    else:
        first, second = rect, rect

    return torch.cat([triangle, first, second], dim=0).t().contiguous()


# data = read_off(vertex_file)
# pos = data.pos
# face = data.face
# print(pos.shape)
# print(face.shape)
# vertex_file = 'chair_0001.off'
def SamplePoints_N(vertex_file, N):
    sampler = SamplePoints(N, remove_faces=True, include_normals=False)
    return sampler(read_off(vertex_file)).pos

# SamplePoints_N(vertex_file, 1024)
# import scipy.io as io
# result1 = np.array(points.pos)
# np.savetxt('npresult1.txt',result1)
# io.savemat('save.mat',{'result1':result1})

# points = txt_PointsCloud_parser('npresult1.txt')
# print(points.shape)

# f = open('pointClouds' + vertex_file, 'w+')
# for point in points.pos.item() :
# f.write(str(point))
# f.close()
