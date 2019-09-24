import torch
import torchvision
import h5py
import pandas as pd
import numpy as np
import random
import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset

'''
Two ways to split dataset to train set and validation dataset
    input : 
        train_iter is Dataset
        k is split 1/k dataset as validation dataset
    
    output : 
        train_dataset with size k-1/k Dataset
        validation_dataset with size 1/k Dataset
'''


def validation_dataset(train_iter, k):
    n = len(train_iter)
    idx = list(range(n))  # indices to all elements
    random.shuffle(idx)  # in-place shuffle the indices to facilitate random splitting
    # 形成一个对训练集idx 的 随机排序
    vail_index = idx[0: int(n / k)]
    train_index = idx[int(n / k):]
    val_set = torch.utils.data.Subset(train_iter, vail_index)
    train_set = torch.utils.data.Subset(train_iter, train_index)
    return train_set, val_set


def random_split_validation(train_data, k):
    n = len(train_data)
    n_val = int(n / k)
    n_train = n - n_val
    return torch.utils.data.random_split(train_data, (n_train, n_val))


'''
Get all train data file path and its class 

'''


def get_path_piar(data_set_path, is_train=True):
    class_names = os.listdir(data_set_path)
    num = list(range(0, len(class_names)))
    class_num_dict = dict(zip(class_names, num))
    all_files = [['Path', ' Class']]
    for index, class_name in enumerate(class_names):
        if (is_train):
            file_path = os.path.join(data_set_path, class_name, 'train')
        else:
            file_path = os.path.join(data_set_path, class_name, 'test')
        files = os.listdir(file_path)

        # files_path = [[os.path.join(os.getcwd(),file_path,file), class_name] for file in files ]
        files_path = [[os.path.join(os.getcwd(), file_path, file), class_name] for file in files]
        all_files += (files_path)
        # list append 与 + 操作不一样
    return all_files


'''
save dataset file path to a csv file

data_set_path = the main dir of you data set.
path2save =  where to save csv file.
'''


def save_data_path(data_set_path, path2save):
    train_files_path = get_path_piar(data_set_path, is_train=True)
    test_files_path = get_path_piar(data_set_path, is_train=False)

    with open(os.path.join(path2save, 'train_files.csv'), 'w') as f:
        for path in train_files_path:
            f.write(path[0] + ',' + path[1] + '\n')

    with open(os.path.join(path2save, 'test_files.csv'), 'w') as f:
        for path in test_files_path:
            f.write(path[0] + ',' + path[1] + '\n')


class ReadDataFromFloder(Dataset):
    def __init__(self, data_set_path, is_train=True):
        self.data_path = pd.read_csv(data_set_path)

        # 写一些transforms操作,对于不同阶段，可能不同，例如train 时候会加入一些噪声，或者旋转等
        self.transformations = {'train': transforms.Compose([transforms.ToTensor()]),

                                'test': transforms.Compose([transforms.ToTensor()])}
        self.is_train = is_train

    # 这个函数根据数据的类型是变化的，因为不同类型的数据，读取为tensor的操作也不同。
    # 例如可以是cv.imread()
    def txt_PointsCloud_parser(self, path_to_off_file):
        # Read the OFF file
        with open(path_to_off_file, 'r') as f:
            contents = f.readlines()
        num_vertices = len(contents)
        # print(num_vertices)
        # Convert all the vertex lines to a list of lists
        vertex_list = [list(map(float, contents[i].strip().split(' '))) for i in list(range(0, num_vertices))]
        # Return the vertices as a 3 x N numpy array
        return np.array(vertex_list)
        # return torch.tensor(vertex_list)

    def augment_data(self, vertices):
        # Random rotation about the Y-axis
        theta = 2 * np.pi * np.random.rand(1)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float)
        # print(Ry)
        vertices = np.dot(vertices, Ry)
        # Add Gaussian noise with standard deviation of 0.2

        vertices += np.random.normal(scale=0.02, size=vertices.shape)
        return vertices

    def __getitem__(self, index):
        # stuff

        # 根据index 拿到 对应的文件路径
        path = self.data_path.iloc[index, 0]

        # 从路径 读取数据 这个函数可以优化，例如用h5文件格式
        data = self.txt_PointsCloud_parser(path)

        # 返回值应该是一个tensor 才能被网络consume,
        # 所以手动转tensor 或者 transform

        if self.is_train:

            # data = self.augment_data(data)
            data = self.transformations['train'](data)


        else:
            data = self.transformations['test'](data)

        label = self.data_path.iloc[index, 1]

        return torch.squeeze(data), label

    def __len__(self):
        return len(self.data_path)


class ModelNet40(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_set_path : h5 file path
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, dataset_path):

        with h5py.File(dataset_path, 'r') as h5file:
            data_tensor, target_tensor = h5file['points'][()], h5file['label'][()]

            # print(data_tensor.shape, data_tensor.shape)

        assert data_tensor.shape[0] == target_tensor.shape[0]
        if isinstance(data_tensor, np.ndarray):

            self.data_tensor = torch.from_numpy(data_tensor)
            self.target_tensor = torch.from_numpy(target_tensor)

        else:
            self.data_tensor = data_tensor
            self.target_tensor = target_tensor

    def __getitem__(self, index):
        # print(index)
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


if __name__ == '__main__':
    data_set_path = os.path.join('Data', 'ModelNet40_')

    train_files_path = os.path.join(data_set_path, 'train_files.csv')
    test_files_path = os.path.join(data_set_path, 'test_files.csv')

    train_point_h5_path = os.path.join(data_set_path, 'ModelNet40_train.h5')
    test_point_h5_path = os.path.join(data_set_path, 'ModelNet40_test.h5')

    # get namt : number dict
    name = os.listdir(data_set_path)
    num = list(range(0, len(name)))
    class_num_dict = dict(zip(name, num))
    print(class_num_dict)

    ##{'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table': 8, 'toilet': 9}

    # read data from data_set_path and save csv file to data_set_path
    save_data_path(data_set_path, data_set_path)

    '''
     from train point data loder read pointcloud data
     and save them to a h5 file
     '''
    print('save train h5 file')
    train_point_data_set = ReadDataFromFloder(train_files_path)
    train_point_data_loader = torch.utils.data.DataLoader(train_point_data_set, \
                                                          batch_size=1, shuffle=False)

    data_h5py, label_h5py = [], []
    for data, label in train_point_data_loader:
        data_h5py.append((torch.squeeze(data)).numpy())
        label_h5py.append(class_num_dict[label[0]])

    with h5py.File(train_point_h5_path, 'w') as f:
        f.create_dataset('points', data=data_h5py)
        f.create_dataset('label', data=label_h5py)

    with h5py.File(train_point_h5_path, 'r') as f:
        x, y = f['points'][()], f['label'][()]
        print(x.shape, y.shape)

    '''
    from test point data loder read pointcloud data
    and save them to a h5 file
    '''
    print('saving test h5 file')
    test_point_data_set = ReadDataFromFloder(test_files_path)
    test_point_data_loader = torch.utils.data.DataLoader(test_point_data_set, \
                                                         batch_size=1, shuffle=False)
    data_h5py, label_h5py = [], []
    for data, label in test_point_data_loader:
        data_h5py.append((torch.squeeze(data)).numpy())
        label_h5py.append(class_num_dict[label[0]])

    with h5py.File(test_point_h5_path, 'w') as f:
        f.create_dataset('points', data=data_h5py)
        f.create_dataset('label', data=label_h5py)

    with h5py.File(test_point_h5_path, 'r') as f:
        x, y = f['points'][()], f['label'][()]
        print(x.shape, y.shape)
