from h5_dataloader import ModelNet40 as MN40
from pointcloud_dataloader import ModelNet40 as myMN40
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
import os
import torch




Train_dataset_path = os.path.join('Data', 'ModelNet40_', 'ModelNet40_test.h5')
h5_off_dataset = myMN40(Train_dataset_path)
h5_dataloader = DataLoader(h5_off_dataset, batch_size=1, shuffle=False)

h5_ply_dataset = MN40(2000,'test')
draw_Point_Cloud(h5_ply_dataset[1][0])
print(h5_ply_dataset[1][1])

draw_Point_Cloud(h5_off_dataset[23][0])

d  = normalize_point_cloud(h5_off_dataset[23][0])
draw_Point_Cloud(d)
print(h5_off_dataset[23][1])

#for data, label in tqdm(h5_dataloader):
