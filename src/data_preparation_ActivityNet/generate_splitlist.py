import numpy as np
import os, json, h5py, math, pdb, glob


splitdataset_path_small = './ActivityNet_dataset_split_small.npz'

splitdataset_path_full = './ActivityNet_dataset_split_full.npz'

train_path = '../../data/ActivityNet/captions/train.json'
val_path = '../../data/ActivityNet/captions/val_merge.json'


def generate_split_list():

    train_list = json.load(open(train_path)).keys()
    val_list = json.load(open(val_path)).keys()

    print len(train_list)
    print len(val_list)

    all_train_num = len(train_list)
    half_train_num = int(0.5*all_train_num)

    train_list_small = train_list[:half_train_num]
    train_list_full = train_list

    print train_list_small
    print val_list

    print len(train_list_full)
    print len(train_list_small)
    print len(val_list)

    np.savez(splitdataset_path_full,train = train_list_full, val = val_list)
    np.savez(splitdataset_path_small,train = train_list_small, val = val_list)


if __name__ == '__main__':
    generate_split_list()



