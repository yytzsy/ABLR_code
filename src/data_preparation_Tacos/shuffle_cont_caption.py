import numpy as np
import h5py


VIDEO_SEG_NUM = 256

def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)

video_data_path_train = '../../data/Tacos/h5py/cont_captions/train.txt'

new_path = '../../data/Tacos/h5py/cont_caption_random/'

video_list_train = get_video_data_HL(video_data_path_train)


h5py_part_list = [[] for i in range(20)]
for i in range(len(video_list_train)):
    index = i % 20
    h5py_part_list[index].append(video_list_train[i])


count = 0
for part_list in h5py_part_list:

    fname = []
    title = []
    data = np.zeros([VIDEO_SEG_NUM,100*len(part_list),4096])+0.0
    label = np.zeros([VIDEO_SEG_NUM,100*len(part_list)])+0.0
    timestamps = []
    duration = []
    norm_timestamps = []


    for idx, item in enumerate(part_list):
        print item
        current_batch = h5py.File(item)
        current_fname = current_batch['fname']
        current_title = current_batch['title']
        current_data = current_batch['data']
        current_label = current_batch['label']
        current_timestamps = current_batch['timestamps']
        current_duration = current_batch['duration']
        current_norm_timestamps = current_batch['norm_timestamps']

        fname = fname + list(current_fname)
        title = title + list(current_title)
        data[:,idx*100:(idx+1)*100,:] = current_data
        label[:,idx*100:(idx+1)*100] = current_label
        timestamps = timestamps + list(current_timestamps)
        duration = duration + list(current_duration)
        norm_timestamps = norm_timestamps + list(current_norm_timestamps)

    index = np.arange(100*len(part_list))
    np.random.shuffle(index)
    fname = [fname[i] for i in index]
    title =  [title[i] for i in index]
    data = data[:,index,:]
    label = label[:,index]
    timestamps = [timestamps[i] for i in index]
    duration = [duration[i] for i in index]
    norm_timestamps = [norm_timestamps[i] for i in index]

    for idx,item in enumerate(part_list):
        batch = h5py.File(new_path+'/'+'train'+str(count)+'.h5','w')
        batch['fname'] = fname[idx*100:(idx+1)*100]
        batch['title'] = title[idx*100:(idx+1)*100]
        batch['data'] = data[:,idx*100:(idx+1)*100,:]
        batch['label'] = label[:,idx*100:(idx+1)*100] 
        batch['timestamps'] = timestamps[idx*100:(idx+1)*100]
        batch['duration'] = duration[idx*100:(idx+1)*100]
        batch['norm_timestamps'] = norm_timestamps[idx*100:(idx+1)*100]
        count = count + 1
        print count



















