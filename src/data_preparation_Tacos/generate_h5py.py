import numpy as np
import os, json, h5py, math, pdb, glob


MAX_LEN = 256
BATCH_SIZE = 10

feature_path = '/Tacos/FTS/c3d_fts_bin_fc6_16/' 
h5py_path = '../../data/Tacos/h5py'
label_path = '../../data/Tacos/middle-labels'
splitdataset_path = '../../data/Tacos/pre/tacos_split.npz'


def get_max_len(path):
    lst = []
    for root, dirs, files in os.walk(path):
        for ele in files:
            if ele.endswith('json'):
                lst.append(root+'/'+ele)
    print lst
    cnt = []
    for ele in lst:
        a = json.load(open(ele))
        cnt.append(len(a[0]))    
    return max(cnt)     


def read_c3d(fn):
    f = open(fn, "rb")
    s = np.fromfile(f, dtype=np.int32)
    f = open(fn, "rb")
    f.seek(20)
    v = np.fromfile(f, dtype=np.float32)
    return s, v



def get_c3d(f_path,ftype):
    v = []
    if not os.path.exists(f_path):
        print 'not exists!!!!'
        return v
    v_f_tmp = os.listdir(f_path)
    v_f = np.sort(v_f_tmp)
    for v_ele in v_f:
        if v_ele.endswith(ftype):
            [_,v_tmp] = read_c3d(os.path.join(f_path,v_ele))
            v.append(v_tmp)
    v = np.array(v)
    print np.shape(v)
    return v



def get_VGG(f_path,ftype):
    if not os.path.exists(f_path + '.npz'):
        return []
    v = np.load(f_path + '.npz')[ftype]
    v=np.array(v)
    return v


def check_HL_nonHL_exist(label):
    idx = len(np.where(label == 1)[0])
    idy = len(np.where(label == 0)[0])
    return idx > 0 and idy > 0


def generate_h5py(X, y, q, fname, dataset, feature_folder_name, batch_start = 0):
    print 'generate_h5py'
    dirname = os.path.join(h5py_path, feature_folder_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    num = len(np.unique(q)) # how much videos here
    if num % BATCH_SIZE == 0:
        batch_num = int(num / BATCH_SIZE)
    else:
        batch_num = int(num / BATCH_SIZE) + 1
    q_idx = 0
    f_txt = open(os.path.join(dirname, dataset + '.txt'), 'w') 
    for i in xrange(batch_start, batch_start + batch_num): # every h5 file contains BATCH_SIZE videos
        train_filename = os.path.join(dirname, dataset + str(i) + '.h5') #each file is a mini-batch
        if os.path.isfile(train_filename):
            q_idx += BATCH_SIZE
            continue
        with h5py.File(train_filename, 'w') as f:
            f['data'] = np.zeros([MAX_LEN,BATCH_SIZE,X.shape[1]])  # a batch of features
            #f['label'] = np.zeros([MAX_LEN,BATCH_SIZE,2])
            f['label'] = np.zeros([MAX_LEN,BATCH_SIZE]) - 1  # all is -1
            f['cont'] = np.zeros([MAX_LEN,BATCH_SIZE])
            #f['reindex'] = np.zeros([MAX_LEN,BATCH_SIZE])
            f['reindex'] = np.zeros(MAX_LEN)
            fname_tmp = []
            title_tmp = []
            for j in xrange(BATCH_SIZE):
                X_id = np.where(q == q_idx)[0]  # find the video segment features of video q_idx
                #while(len(X_id) == 0 or not check_HL_nonHL_exist(y[X_id])):
                while(len(X_id) == 0):
                    q_idx += 1
                    X_id = np.where(q == q_idx)[0]
                    if q_idx > max(q):
                        while len(fname_tmp) < BATCH_SIZE: #if the video is not enough for the last mini batch, insert ' '
                            fname_tmp.append('')
                            title_tmp.append('')
                        fname_tmp = [a.encode('utf8') for a in fname_tmp]
                        title_tmp = [a.encode('utf8') for a in title_tmp]
                        fname_tmp = np.array(fname_tmp)
                        title_tmp = np.array(title_tmp)
                        f['fname'] = fname_tmp
                        f['title'] = title_tmp
                        print train_filename
                        f_txt.write(train_filename + '\n')
                        return
                f['data'][:len(X_id),j,:] = X[X_id,:]
                f['label'][:len(X_id),j] = y[X_id]
                f['cont'][1:len(X_id)+1,j] = 1
                f['reindex'][:len(X_id)] = np.arange(len(X_id))
                f['reindex'][len(X_id):] = len(X_id)
                fname_tmp.append(fname[q_idx])
                title_tmp.append(fname[q_idx])
                if q_idx == q[-1]:
                    while len(fname_tmp) < BATCH_SIZE:
                        fname_tmp.append('')
                        title_tmp.append('')
                    fname_tmp = [a.encode('utf8') for a in fname_tmp]
                    title_tmp = [a.encode('utf8') for a in title_tmp]
                    fname_tmp = np.array(fname_tmp)
                    title_tmp = np.array(title_tmp)
                    f['fname'] = fname_tmp
                    f['title'] = title_tmp
                    print train_filename
                    f_txt.write(train_filename + '\n')
                    return
                q_idx += 1
            fname_tmp = [a.encode('utf8') for a in fname_tmp]
            title_tmp = [a.encode('utf8') for a in title_tmp]
            fname_tmp = np.array(fname_tmp)
            title_tmp = np.array(title_tmp)
            f['fname'] = fname_tmp
            f['title'] = title_tmp
        print train_filename
        f_txt.write(train_filename + '\n')



def get_feats_depend_on_label(label, per_f, v, idx):
    #get features in one video, the feature is represented as clip features
    #label means the video clip division
    #perf: 1(VGG) 16(C3D)
    #C3D 16 frames a feature, and VGG 1 frames a feature
    X = []  # feature
    y = []  # indicate if video is finished
    q = []  # idx is the index of video in train/test/val dataset, all the segment in video will be tagged as idx in list q
    for l_index in xrange(len(label[0])):
        low = int(label[0][l_index][0])
        up = int(label[0][l_index][1])+1
        up_ = up
        #pdb.set_trace()
        if  low >= len(v) or low == up:
            X.append(X[-1])
        else:
            X.append(np.mean(v[low:up,:],axis=0))
        y.append(label[1][l_index])
        q.append(idx)
    return X, y, q


def load_feats(files, dataset, feature = 'c3d'):
    #dataset: 'train' 'val' or 'test'
    #feature: 'c3d' 'VGG' 'cont'(conconation of c3d and VGG)
    X = [] # feature
    y = [] # indicate if video is finished
    q = [] # the index of video in train/test/val dataset
    fname = []
    idx = 0 # video index in the dataset 
    for ele in files: 
        print ele, idx
        index = ele.find('.')
        ele = ele[:index]
        l_path = os.path.join(label_path, ele + '.json')
        label = json.load(open(l_path))
        if len(label[0]) > MAX_LEN:
            print 'exceed max_len!!!!!'
            sys.exit(-6)

        f_path = os.path.join(feature_path, ele)
        if feature == 'c3d':
            v = get_c3d(f_path,'fc6-1')
            per_f = 8
            if len(v) == 0:
                print 'feature read null !!!!'
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, idx)
        elif feature == 'VGG':
            v = get_VGG(f_path,'fc6')
            per_f = 1
            if len(v) == 0:
                continue
            [x_tmp, y_tmp, q_tmp] = get_feats_depend_on_label(label, per_f, v, idx)
        elif feature == 'cont':
            v1 = get_c3d(os.path.join(feature_path,'c3d',ele),'fc6-1')
            per_f1 = 8
            v2 = get_VGG(os.path.join(feature_path,'VGG',ele),'fc6')
            per_f2 = 1
            if len(v1) == 0 or len(v2) == 0:
                print "fuck!!"
                continue
            [x1_tmp, y1_tmp, q1_tmp] = get_feats_depend_on_label(label, per_f1, v1, idx)
            [x2_tmp, y2_tmp, q2_tmp] = get_feats_depend_on_label(label, per_f2, v2, idx)
            x_tmp = map(list, zip(*(zip(*x1_tmp) + zip(*x2_tmp))))
            y_tmp = y1_tmp
            q_tmp = q1_tmp
        X += x_tmp
        y += y_tmp
        q += q_tmp
        #pdb.set_trace()
        fname.append(ele+'.avi')
        idx += 1
    return np.array(X), np.array(y), np.array(q), np.array(fname)


def Normalize(X, normal = 0):
    if normal == 0:
        mean = np.mean(X,axis = 0)
        std = np.std(X,axis = 0)
        idx = np.where(std == 0)[0]
        std[idx] = 1
    else:
        mean = normal[0]
        std = normal[1]
    X = (X - mean) / std
    return X, mean, std


def driver(inp_type, Rep_type, outp_folder_name):
    dataset = 'train'
    num = 10
    List = np.load(splitdataset_path)[dataset] # get the train,val or test training video name
    for iii in range(int(math.ceil(len(List) / (num*1.0)))):
        [X, y, Q, fname] = load_feats(List[iii*num:min(len(List),(iii+1)*num)], dataset, Rep_type)
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*num)
    
    dataset = 'val'
    List = np.load(splitdataset_path)[dataset]
    for iii in range(int(math.ceil(len(List) / (num*1.0)))):
        [X, y, Q, fname] = load_feats(List[iii*num:min(len(List),(iii+1)*num)], dataset, Rep_type)
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*num)
        
    dataset = 'test'
    List = np.load(splitdataset_path)[dataset]
    for iii in range(int(math.ceil(len(List) / (num*1.0)))):
        [X, y, Q, fname] = load_feats(List[iii*num:min(len(List),(iii+1)*num)], dataset, Rep_type)
        if inp_type == 'h5py':
            generate_h5py(X, y, Q, fname, dataset, outp_folder_name, batch_start = iii*num)
    

def getlist(path, split):
    List = glob.glob(path+split+'*.h5')
    print path+split+'.txt'
    f = open(path+split+'.txt','w')
    for ele in List:
        f.write(ele+'\n')

    
if __name__ == '__main__':

    driver('h5py','c3d','cont')
    path = os.path.join(h5py_path, 'cont' + '/')
    getlist(path,'train')
    getlist(path,'val')
    getlist(path,'test')


