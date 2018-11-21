import os
import cv2
import json
import pdb
import numpy as np

video_path = '' # path for activityNet videos
output_path = '../../data/ActivityNet/middle-labels/'

seg_num = 128

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Split data to train data, valid data and test data
def splitdata(path, train_num, val_num):
    lst = os.listdir(path)
    name = []
    for ele in lst:
        name.append(os.path.splitext(ele)[0])

    print len(name)
    print name[0:100]
    name = np.random.permutation(name)
    print name[0:100]

    train = name[0:train_num]
    val = name[train_num:train_num+val_num]
    test = name[train_num+val_num:]
    np.savez('msvd_dataset',train=train, val=val, test=test)


def get_total_frame_number(fn):
    cap = cv2.VideoCapture(fn)
    if not cap.isOpened():
        print "could not open :",fn
        sys.exit() 
    length = float(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return length

def getlist(Dir):
    pylist = []
    for root, dirs, files in os.walk(Dir):
        for ele in files:
            pylist.append(root+'/'+ele)

    return pylist

def get_frame_list(frame_num,seg_num):

    num_clip8 = np.floor(frame_num/8)-1
    clip_per_seg = float(num_clip8/seg_num)

    seg_left = np.zeros(seg_num)
    seg_right = np.zeros(seg_num)

    for i in range(seg_num):
        seg_left[i] = int(i * clip_per_seg)
        seg_right[i] = int((i+1) * clip_per_seg)
        if seg_right[i] > num_clip8:
            seg_right[i] = num_clip8
            
    clip_num_list = []
    for i in range(seg_num):
        clip_num_list.append([ seg_left[i],seg_right[i] ])

    return clip_num_list


def get_label_list(fname):
    
    frame_len = get_total_frame_number(fname)
    frame_list = get_frame_list(frame_len,seg_num)
    if frame_list == -1:
        return
    label_list = [-1]*len(frame_list)
    label_list[-1] = 0
    fname = fname.split('/')[-1].split('.')[0]
    outfile = output_path+str(fname)+'.json'
    if not os.path.isfile(outfile):
    	json.dump([frame_list, label_list], open(outfile,"w"))

if __name__=='__main__':
    b = getlist(video_path)
    count = 0
    for ele in b:
        fname = ele
        if not os.path.isfile(output_path+str(fname)+'.json'):
            print fname
            get_label_list(fname)
        count += 1
    print len(b)
    print count

