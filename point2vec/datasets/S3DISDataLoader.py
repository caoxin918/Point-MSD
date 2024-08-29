#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import os, sys, h5py, numpy as np
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch


def download_S3DIS():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')):
        www = 'https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
    if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version')):
        if not os.path.exists(os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')):
            print('Please download Stanford3dDataset_v1.2_Aligned_Version.zip \
                from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under data/')
            sys.exit(0)
        else:
            zippath = os.path.join(DATA_DIR, 'Stanford3dDataset_v1.2_Aligned_Version.zip')
            os.system('unzip %s' % (zippath))
            os.system('rm %s' % (zippath))


def prepare_test_data_semseg():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(os.path.join(DATA_DIR, 'stanford_indoor3d')):
        os.system('python sem_part/collect_indoor3d_data.py')
    if not os.path.exists(os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')):
        os.system('python sem_part/gen_indoor3d_h5.py')


def load_data_semseg(partition, test_area, train_area):
    ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            for area in train_area:
                if "Area_" + area in room_name:
                    train_idxs.append(i)
                    break
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


class S3DIS(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1', train_area=['1','2','3','4','5','6']):
        self.data, self.seg = load_data_semseg(partition, test_area, train_area)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]


class S3DISDataModule(pl.LightningDataModule):
    """ """

    def __init__(
        self,
        data_dir: str = "./data/indoor3d_sem_seg_hdf5_data",
        train_area = ['1','2','3','4','5'],
        test_area: str = '6',
        batch_size: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_area = train_area
        self.test_area = test_area

        # self.classes = ['ceiling', 'floor', 'wall', 'beam', 'column',
        #            'window', 'door', 'table', 'chair', 'sofa',
        #            'bookcase', 'board', 'clutter']
        #
        # self.class2label = {cls: i for i, cls in enumerate(self.classes)}
        # self.seg_classes = self.class2label
        # self.seg_label_to_cat = {}
        # for i, cat in enumerate(self.seg_classes.keys()):
        #     self.seg_label_to_cat[i] = cat


    def setup(self, stage: Optional[str] = None):
        self.train_dataset = S3DIS(4096, self.train_area)  # type: ignore
        self.test_dataset = S3DIS(4096, 'test', self.test_area)  # type: ignore

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            shuffle=True,
            drop_last=True,  # type: ignore
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=4,
            persistent_workers=True,
        )



