import h5py
import numpy as np
import json
import pdb
import unicodedata
import glob
import os
import json

path = '../../data/Tacos/h5py/'
feature_folder = 'cont_captions'

if not os.path.exists(path+feature_folder):
    os.makedirs(path+feature_folder)

train_captions_path = '../../data/Tacos/pre/train.json'
test_captions_path = '../../data/Tacos/pre/test.json'
val_captions_path = '../../data/Tacos/pre/val.json'

batch_size = 100 # each mini batch will have batch_size sentences
n_length = 256
video_fts_dim = 4096


def trans_video_youtube(datasplit):

    if datasplit == 'train':
        merge_j = json.load(open(train_captions_path))
    elif datasplit == 'test':
        merge_j = json.load(open(test_captions_path))
    elif datasplit == 'val':
        merge_j = json.load(open(val_captions_path))

    List = open(path+'cont/'+datasplit+'.txt').read().split('\n')[:-1] #get a list of h5 file, each file is a minibatch
    initial = 0
    cnt = 0
    fname = []
    title = []
    data = []
    label = []
    timestamps = []
    duration = []
    norm_timestamps = []
    
    for ele in List:
        print ele
        print initial
        train_batch = h5py.File(ele)
        for idx, video in enumerate(train_batch['title']):
            if video in merge_j.keys():
                for capidx, caption in enumerate(merge_j[video]['sentences']): 
                    if len(caption.split(' ')) < 35:
                        fname.append(video)
                        duration.append(merge_j[video]['duration']) 
                        timestamps.append( merge_j[video]['timestamps'][capidx] )
                        norm_stamps = [merge_j[video]['timestamps'][capidx][0]/merge_j[video]['duration'], merge_j[video]['timestamps'][capidx][1]/merge_j[video]['duration']]
                        norm_timestamps.append(norm_stamps)
                        title.append(unicodedata.normalize('NFKD', caption).encode('ascii','ignore'))
                        data.append(train_batch['data'][:,idx,:]) #insert item shape is (n_length,dim), so the data's shape will be (n_x,n_length,dim), so it need transpose
                        label.append(train_batch['label'][:,idx])
                        cnt += 1 #sentence is enough for batch_size
                        if cnt == batch_size:
                            print(path+feature_folder+'/'+datasplit+str(initial)+'.h5')
                            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
                            data = np.transpose(data,(1,0,2))
                            batch['data'] = np.array(data)#np.zeros((n_length,batch_size,4096*2))
                            fname = np.array(fname)
                            title = np.array(title)
                            batch['duration'] = duration
                            batch['fname'] = fname
                            batch['title'] = title
                            batch['timestamps'] = timestamps
                            batch['norm_timestamps'] = norm_timestamps
                            batch['label'] = np.transpose(np.array(label)) #np.zeros((n_length,batch_size))
                            fname = []
                            duration = []
                            timestamps = []
                            norm_timestamps = []
                            title = []
                            label = []
                            data = []
                            cnt = 0
                            initial += 1
        if ele == List[-1] and len(fname) > 0:
            while len(fname) < batch_size:
                fname.append('')
                title.append('')
                timestamps.append([-1,-1])
                norm_timestamps.append([-1,-1])
                duration.append(-1)
            batch = h5py.File(path+feature_folder+'/'+datasplit+str(initial)+'.h5','w')
            batch['data'] = np.zeros((n_length,batch_size,video_fts_dim))
            batch['data'][:,:len(data),:] = np.transpose(np.array(data),(1,0,2))#np.zeros((n_length,batch_size,4096+1024))
            fname = np.array(fname)
            title = np.array(title)
            batch['fname'] = fname
            batch['title'] = title
            batch['duration'] = duration
            batch['timestamps'] = timestamps
            batch['norm_timestamps'] = norm_timestamps
            batch['label'] = np.ones((n_length,batch_size))*(-1)
            batch['label'][:,:len(data)] = np.array(label).T



def getlist(feature_folder_name, split):
    list_path = os.path.join(path, feature_folder_name+'/')
    List = glob.glob(list_path+split+'*.h5')
    f = open(list_path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')


if __name__ == '__main__':
    trans_video_youtube('train')
    trans_video_youtube('test')
    trans_video_youtube('val')
    getlist(feature_folder,'train')
    getlist(feature_folder,'val')
    getlist(feature_folder,'test')
