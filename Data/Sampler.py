import os
import os.path as osp

from Sample_points import SamplePoints_N
import scipy.io as io
import numpy as np

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
    
if __name__== '__main__': 

    input_pairs = []
    dataset_root_path = 'ModelNet40'
    processed_path = 'ModelNet40_'
    test = 1
    print('Processing')
    #  List of tuples grouping a label with a class
    gt_key = os.listdir(dataset_root_path)
    #gt_key_list = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    # print('gt_key {}'.format(gt_key))
    for idx, obj in enumerate(gt_key):
        
        if test:
            path_to_files = osp.join(dataset_root_path, obj, 'test')
        else:
            path_to_files = osp.join(dataset_root_path, obj, 'train')
        files = os.listdir(path_to_files)
        filepaths = [(osp.join(path_to_files, file), idx)
                     for file in files]
        input_pairs = input_pairs + filepaths
        #print(input_pairs)
        
        
    for i, (path, idx ) in enumerate(input_pairs):
        floader_name = gt_key[idx]
        #floader_name = 'bathtub' ....
        if test: 
            save_path = os.path.join(processed_path,floader_name, 'test')
            
        else:
            save_path = os.path.join(processed_path,floader_name, 'train')
            
        isExists = os.path.exists(save_path)
        if not isExists:						#判断如果文件不存在,则创建
            os.makedirs(save_path)	
            print("%s 目录创建成功"%floader_name)
       
       # file_path_to_save =save_path + '/'+path.split('/')[-1].split('.')[0]+ '.points' 
        
        file_path_to_save = os.path.join(save_path,floader_name+str(i)+ '.points')
        
        vertices = SamplePoints_N(path, 2000)
        # # import scipy.io as io
        np.savetxt(file_path_to_save,np.array(vertices))
        print(file_path_to_save)
        
        # else:
            # print("%s 目录已经存在"%i)	
            # continue			#如果文件不存在,则继续上述操作,直到循环结束
            
        # print(path,idx, i)