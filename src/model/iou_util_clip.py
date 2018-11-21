import numpy as np
import h5py
import json
import random
import logging




def calculate_IOU(groundtruth, predict):

    groundtruth_init = max(0,groundtruth[0])
    groundtruth_end = groundtruth[1]

    predict_init = max(0,predict[0])
    predict_end = predict[1]

    init_min = min(groundtruth_init,predict_init)
    end_max = max(groundtruth_end,predict_end)

    init_max = max(groundtruth_init,predict_init)
    end_min = min(groundtruth_end,predict_end)

    if end_min < init_max:
        return 0

    IOU = ( end_min - init_max ) * 1.0 / ( end_max - init_min)
    return IOU

  

def calculate_mean_IOU(log,file_path,clip_num):

    result = np.load(file_path)
    predict_timestamps = result['predict_timestamps']
    groundtruth_timestamps = result['groundtruth_timestamps']
    duration = result['duration']
    caption = result['caption']
    video = result['video']
    predict_attention_weights_v =  result['predict_attention_weights_v']

    iou_list = []
    ix = 0
    for i1,i2 in zip(predict_timestamps,groundtruth_timestamps):
        predict_left = int(i1[0] * clip_num) * 1.0 / clip_num * duration[ix]
        predict_right = int(i1[1] * clip_num) * 1.0 / clip_num * duration[ix]
        ground = i2
        iou = calculate_IOU(ground,(predict_left,predict_right))
        iou_list.append(iou)
        ix += 1

    mean_iou = np.mean(iou_list)
    logging.info("************************************")
    logging.info("mean_iou = {:f}".format(mean_iou))
    print 'mean_iou = {:}'.format(mean_iou)
    logging.info("************************************")


def calculate_map_IOU(logging,file_path,clip_num):

    result = np.load(file_path)
    predict_timestamps = result['predict_timestamps']
    groundtruth_timestamps = result['groundtruth_timestamps']
    duration = result['duration']
    predict_attention_weights_v =  result['predict_attention_weights_v']

    count_01 = 0
    count_03 = 0
    count_05 = 0
    count_07 = 0
    count_all = 0
    ix = 0
    for i1,i2 in zip(predict_timestamps,groundtruth_timestamps):
        predict_left = int(i1[0] * clip_num) * 1.0 / clip_num * duration[ix]
        predict_right = int(i1[1] * clip_num) * 1.0 / clip_num * duration[ix]
        ground = i2
        iou = calculate_IOU(ground,(predict_left,predict_right))
        if iou >= 0.1:
            count_01 += 1
        if iou >= 0.3:
            count_03 += 1
        if iou >= 0.5:
            count_05 += 1
        if iou >= 0.7:
            count_07 += 1
        count_all+=1
        ix+=1

    logging.info("Recall_iou_0.1 = {:}".format(count_01*1.0/count_all))
    logging.info("Recall_iou_0.3 = {:}".format(count_03*1.0/count_all))
    logging.info("Recall_iou_0.5 = {:}".format(count_05*1.0/count_all))
    logging.info("Recall_iou_0.7 = {:}".format(count_07*1.0/count_all))