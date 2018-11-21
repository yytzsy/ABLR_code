import numpy as np
import h5py
import json
from generate_nolabel import *


def generate_split():

    train_txt = '../../data/Tacos/pre/tacos_train_0_74.txt'
    val_txt = '../../data/Tacos/pre/tacos_test_100_126.txt'
    test_txt = '../../data/Tacos/pre/tacos_val_75_99.txt'
    save_split_path = '../../data/Tacos/pre/tacos_split.npz'

    file = open(train_txt,'r')
    lines = file.readlines()
    train_list = []
    for line in lines:
        video_name = line.split('_')
        video_name = video_name[0]
        train_list.append(video_name)

    file = open(test_txt,'r')
    lines = file.readlines()
    test_list = []
    for line in lines:
        video_name = line.split('_')
        video_name = video_name[0]
        test_list.append(video_name)

    file = open(val_txt,'r')
    lines = file.readlines()
    val_list = []
    for line in lines:
        video_name = line.split('_')
        video_name = video_name[0]
        val_list.append(video_name)
    

    np.savez(save_split_path, train = list(set(train_list)), test = list(set(test_list)), val = list(set(val_list)) ) 
    print len(list(set(train_list)))
    print len(list(set(test_list)))
    print len(list(set(val_list)))


def generate_caption_json(datasplit):


    split_path = '../../data/Tacos/pre/tacos_split.npz'
    output_json = '../../data/Tacos/pre/'+datasplit+'.json'
    video_dir_path = '/Tacos/videos/'
    video_sentence_index_path = '/Tacos/index/'

    split_dict = np.load(split_path)
    video_list = split_dict[datasplit]

    output_dict = {}
    for video in video_list:

        video_dict = {}
        video_path = video_dir_path + video
        frame_num,fps = get_total_frame_number(video_path)
        print fps
        video_dict['duration'] = frame_num
        file = open(video_sentence_index_path+video.split('.')[0]+'.sentences.tsv','r')
        lines = file.readlines()

        sentences_list = []
        timestamps_list = []
        for line in lines:
            content = line.split('\t')
            sentence = content[1]
            time = [int(content[3]),int(content[4])]
            sentences_list.append(sentence)
            timestamps_list.append(time)

        video_dict['timestamps'] = timestamps_list
        video_dict['sentences'] = sentences_list
        output_dict[video] = video_dict


    f = open(output_json,"w")
    json.dump(output_dict,f)
    print 'dump!'


if __name__ == '__main__':
    generate_split()
    generate_caption_json('train')
    generate_caption_json('test')
    generate_caption_json('val')