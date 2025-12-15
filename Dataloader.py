import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
import random
from numpy.random import permutation


class MultiViewDataset(Dataset):
    def __init__(self, data,label,args):
        self.data = data
        self.label = label
        self.args = args

    def __len__(self):
        return self.label[0].shape[0]

    def __getitem__(self, idx):
        data0 = torch.from_numpy(self.data[0][idx])
        data1 = torch.from_numpy(self.data[1][idx])
        true_label = np.int64(self.label[abs(1-self.args.main_view)][idx])
        return data0,data1,true_label



def MultiViewDatasetLoader(args):
    data = []
    mat = sio.loadmat('/home/shanghui/dsh/datasets/MVC/'+args.dataset+'.mat')
    scaler = preprocessing.MinMaxScaler()
    data.append(scaler.fit_transform(mat['X'][0][0]).astype(np.float32))
    data.append(scaler.fit_transform(mat['X'][0][1]).astype(np.float32))
    label = np.squeeze(mat['Y']).astype(np.uint8)
    dim_list = [da.shape[1] for da in data]
    num_sample = label.shape[0]
    y = np.empty(shape=(2, len(label)))
    y[0] = y[1] = label
    aligned_index =[]
    unaligned_index =[]
    if args.unalign_ratio!=0:
        aligned_index, unaligned_index = aligned_data_split(num_sample, args.unalign_ratio, args.seed)
        shuffle_index = permutation(unaligned_index)
        data[args.main_view][unaligned_index] = data[args.main_view][shuffle_index]
        y[args.main_view][unaligned_index] = y[args.main_view][shuffle_index]
    dataloader = MultiViewDataset(data,y,args)
    return dataloader, num_sample, dim_list,aligned_index,unaligned_index,y


def aligned_data_split(num_sample, unalign_ratio, seed):
    random.seed(seed)
    random_idx = random.sample(range(num_sample), num_sample)
    align_num = np.ceil((1 - unalign_ratio) * num_sample).astype(int)
    align_idx = np.array(sorted(random_idx[0:align_num]))
    unalign_num = np.floor(unalign_ratio * num_sample).astype(int)
    unalign_idx = np.array(sorted(random_idx[-unalign_num:]))
    return align_idx, unalign_idx