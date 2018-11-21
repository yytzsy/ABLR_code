#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
from collections import defaultdict
from tensorflow.python.ops import rnn
from tensorflow.contrib.rnn.python.ops import rnn_cell
from keras.preprocessing import sequence
import unicodedata
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
import logging
import string

from iou_util_clip import calculate_mean_IOU
from iou_util_clip import calculate_map_IOU


############## Parameters #################

model_mode = 'AW'
dataset = 'TACOS'
if dataset == 'ActivityNet':
    dim_image_encode = -1
    dim_image = 500
    dim_word = 300
    dim_hidden= 256
    dim_hidden_regress = 64
    n_frame_step = 128
    n_caption_step = 35
    n_epochs = 200
    batch_size = 100
    learning_rate = 0.001 #0.0001
    alpha_regress = 1.0
    alpha_attention = 5.0
    regress_layer_num = 2
    iter_test = 10
    iter_localize = 5
    gpu_id = 0
    word_embedding_path = '../../data/glove.840B.300d_dict.npy'
    video_data_path_train = '../../data/ActivityNet/h5py/cont_captions/train.txt'
    video_data_path_val = '../../data/ActivityNet/h5py/cont_captions/val.txt'
    video_feat_path = '../../data/ActivityNet/h5py/cont_captions'
    model_save_dir = '../../save_models/ActivityNet/'
    result_save_dir = '../../save_results/ActivityNet/'
    words_dir = '../../words/ActivityNet/'
    wordtoix_path = words_dir+'wordtoix.npy'
    ixtoword_path = words_dir+'ixtoword.npy'
    word_fts_path = words_dir+'word_glove_fts_init.npy'
elif dataset == 'TACOS':
    dim_image_encode = 500
    dim_image = 4096
    dim_word = 300
    dim_hidden= 256
    dim_hidden_regress = 64
    n_frame_step = 256
    n_caption_step = 35
    n_epochs = 200
    batch_size = 100
    learning_rate = 0.0001 #0.0001
    alpha_regress = 1.0
    alpha_attention = 5.0
    regress_layer_num = 2
    iter_test = 10
    iter_localize = 1
    gpu_id = 0  
    word_embedding_path = '../../data/glove.840B.300d_dict.npy'
    video_data_path_train = '../../data/Tacos/h5py/cont_caption_random/train.txt'
    video_data_path_val = '../../data/Tacos/h5py/cont_captions/test.txt'
    video_feat_path = '../../data/Tacos/h5py/cont_captions'
    model_save_dir = '../../save_models/TACOS/'
    result_save_dir = '../../save_results/TACOS/'
    words_dir = '../../words/TACOS/'
    wordtoix_path = words_dir+'wordtoix.npy'
    ixtoword_path = words_dir+'ixtoword.npy'
    word_fts_path = words_dir+'word_glove_fts_init.npy'

retrain = False
if retrain:
    MODEL = ''

#########################  Path ###############


def make_prepare_path(regress_layer_num,dataset):

    sub_dir = 'alpha_attention'+str(alpha_attention)+'_'+'alpha_regress'+str(alpha_regress)+'_'+'dimHidden'+str(dim_hidden)+'_'+'lr'+str(learning_rate)
    sub_dir = 'ABLR_'+dataset+'_'+model_mode+'_regressLayer'+ str(regress_layer_num) + '_' +sub_dir

    #########################  for logging  #########################################################

    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    log_file_name = 'screen_output_'+str(time_stamp)+'_'+sub_dir+'.log'
    fh = logging.FileHandler(filename=log_file_name, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logging.root.addHandler(fh)

    model_save_path = model_save_dir+sub_dir
    result_save_path = result_save_dir+sub_dir

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    if not os.path.exists(words_dir):
        os.makedirs(words_dir)

    return sub_dir, logging, regress_layer_num, model_save_path, result_save_path



def get_word_embedding(word_embedding_path,wordtoix_path,ixtoword_path,extracted_word_fts_init_path):
    print('loading word features ...')
    word_fts_dict = np.load(open(word_embedding_path)).tolist()
    wordtoix = np.load(open(wordtoix_path)).tolist()
    ixtoword = np.load(open(ixtoword_path)).tolist()
    word_num = len(wordtoix)
    extract_word_fts = np.random.uniform(-3,3,[word_num,300]) 
    count = 0
    for index in range(word_num):
        if ixtoword[index] in word_fts_dict:
            extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
            count = count + 1
    np.save(extracted_word_fts_init_path,extract_word_fts)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract a CNN features')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--net', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--layer',dest='regress_layer_num',
                        help = 'choose number of regression layers',
                        default = 2, type = int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def smooth_l1_loss(predict_location,video_location):
    diff = predict_location - video_location
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    smooth_l1norm = tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5),1)
    return tf.reduce_sum(smooth_l1norm)


class ABLR():
    def __init__(self, model_mode, word_emb_init, dim_image, dim_image_encode, dim_word, n_words, dim_hidden, batch_size, n_frame_step, drop_out_rate, regress_layer_num, dim_hidden_regress, bias_init_vector=None):
        self.dim_image = dim_image
        self.dim_image_encode = dim_image_encode
        self.dim_word = dim_word
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_frame_step = n_frame_step
        self.drop_out_rate = drop_out_rate
        self.regress_layer_num = regress_layer_num
        self.dim_hidden_regress = dim_hidden_regress
        self.model_mode = model_mode

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(initial_value = word_emb_init, name='Wemb')

        if self.dim_image == 4096:
            self.embed_video_w = tf.Variable(tf.random_uniform([self.dim_image, self.dim_image_encode], -0.1,0.1), name='embed_video_w')
            self.embed_video_b = tf.Variable(tf.zeros([self.dim_image_encode]), name='embed_video_b')

        # LSTM for encode video first
        self.lstm_video = LSTMCell(num_units = dim_hidden, use_peepholes = True)
        self.lstm_video_dropout = DropoutWrapper(self.lstm_video,output_keep_prob=1 - self.drop_out_rate)

        # Reverse LSTM for encode video
        self.reverse_lstm_video = LSTMCell(num_units = dim_hidden, use_peepholes = True)
        self.reverse_lstm_video_dropout = DropoutWrapper(self.reverse_lstm_video, output_keep_prob=1 - self.drop_out_rate)

        # LSTM for encode sentence first
        self.lstm_sentence = LSTMCell(num_units = dim_hidden, use_peepholes = True)
        self.lstm_sentence_dropout = DropoutWrapper(self.lstm_sentence,output_keep_prob=1 - self.drop_out_rate)

        # Reverse LSTM for encode sentence
        self.reverse_lstm_sentence = LSTMCell(num_units = dim_hidden, use_peepholes = True)
        self.reverse_lstm_sentence_dropout = DropoutWrapper(self.reverse_lstm_sentence,output_keep_prob=1 - self.drop_out_rate)

        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_att_w_q = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w_q')
        self.embed_att_W_qa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_W_qa')
        self.embed_att_U_qa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_U_qa')
        self.embed_att_b_qa = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_b_qa')

        self.w_fuse_video = tf.Variable(tf.random_uniform([2*dim_hidden, dim_hidden], -0.1,0.1), name='w_fuse_video')
        self.b_fuse_video = tf.Variable(tf.zeros([dim_hidden]), name='b_fuse_video')

        self.w_fuse_sentence = tf.Variable(tf.random_uniform([2*dim_hidden, dim_hidden], -0.1,0.1), name='w_fuse_sentence')
        self.b_fuse_sentence = tf.Variable(tf.zeros([dim_hidden]), name='b_fuse_sentence')

        if self.model_mode == 'AF':
            self.w_fuse = tf.Variable(tf.random_uniform([2*dim_hidden, dim_hidden], -0.1,0.1), name='w_fuse')
            self.b_fuse = tf.Variable(tf.zeros([dim_hidden]), name='b_fuse')
            if regress_layer_num == 1:
                self.regress_w = tf.Variable(tf.random_uniform([dim_hidden, 2], -0.1,0.1), name='regress_w')
                self.regress_b = tf.Variable( tf.zeros([2]), name='regress_b')
            else:
                self.regress_w = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden_regress], -0.1,0.1), name='regress_w')
                self.regress_b = tf.Variable( tf.zeros([dim_hidden_regress]), name='regress_b')
                self.regress_w_2 = tf.Variable(tf.random_uniform([dim_hidden_regress, 2], -0.1,0.1), name='regress_w_2')
                self.regress_b_2 = tf.Variable( tf.zeros([2]), name='regress_b_2')
        elif self.model_mode == 'AW':
            if regress_layer_num == 1:
                self.regress_w = tf.Variable(tf.random_uniform([n_frame_step, 2], -0.1,0.1), name='regress_w')
                self.regress_b = tf.Variable( tf.zeros([2]), name='regress_b')
            else:
                self.regress_w = tf.Variable(tf.random_uniform([n_frame_step, dim_hidden_regress], -0.1,0.1), name='regress_w')
                self.regress_b = tf.Variable( tf.zeros([dim_hidden_regress]), name='regress_b')
                self.regress_w_2 = tf.Variable(tf.random_uniform([dim_hidden_regress, 2], -0.1,0.1), name='regress_w_2')
                self.regress_b_2 = tf.Variable( tf.zeros([2]), name='regress_b_2')


    def build_model(self):  # caption step: means the sentence length, # n_frame_step: means the video segment nums

        with tf.variable_scope(tf.get_variable_scope()) as model_scope:

            loss_mask = tf.placeholder(tf.float32,[self.batch_size])

            video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step, self.dim_image]) # b x n x d
            video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step]) # b x n
            video_location = tf.placeholder(tf.float32, [self.batch_size,2])
            video_attention_weights = tf.placeholder(tf.float32,[self.batch_size,n_frame_step])

            caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step]) # b x n_caption_step
            caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step]) # b x n_caption_step
            caption_emb_mask = tf.placeholder(tf.float32, [n_caption_step,self.batch_size, dim_hidden]) # indicate where do caption end
            reverse_caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step])
            caption_length = tf.placeholder(tf.int32,[self.batch_size])


            if self.dim_image == 4096:
                video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
                video_emb = tf.nn.xw_plus_b(video_flat, self.embed_video_w, self.embed_video_b) # (b x n) x h
                video_emb = tf.reshape(video_emb, [self.batch_size, self.n_frame_step, self.dim_image_encode]) # b x n x h
            else:
                video_emb = video

            ################################################ for encode video with LSTM ###################################################

            # ordinal sequence
            state1_video = ( tf.zeros([self.batch_size, self.lstm_video.state_size[0]]), tf.zeros([self.batch_size, self.lstm_video.state_size[1]] ) )
            output_video_list = []
            for i in range(n_frame_step):
                current_embed_video = video_emb[:,i,:] # bxnxd
                with tf.variable_scope("LSTM_video"):
                    output_video, state1_video = self.lstm_video_dropout(current_embed_video, state1_video)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                output_video_list.append(output_video) # nxbxh
            ordinal_sequence_video_emb = tf.stack(output_video_list) # n x b x h

            # reverse sequence
            reverse_state1_video = ( tf.zeros([self.batch_size, self.reverse_lstm_video.state_size[0]]), tf.zeros([self.batch_size, self.reverse_lstm_video.state_size[1]] ) )
            reverse_output_video_list = []
            for i in range(n_frame_step):
                current_embed_video = video_emb[:,n_frame_step-i-1,:] # bxnxd
                with tf.variable_scope("Reverse_LSTM_video"):
                    reverse_output_video, reverse_state1_video = self.reverse_lstm_video_dropout(current_embed_video, reverse_state1_video)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                reverse_output_video_list.append(reverse_output_video) # nxbxh
            new_reverse_output_video_list = []
            for i in range(n_frame_step):
                new_reverse_output_video_list.append(reverse_output_video_list[n_frame_step-i-1])
            reverse_sequence_video_emb = tf.stack(new_reverse_output_video_list) # n x b x h

            sequence_video_emb = tf.concat([ordinal_sequence_video_emb,reverse_sequence_video_emb],2) # nxbx2h
            sequence_video_emb_flat = tf.reshape(sequence_video_emb,[-1,self.dim_hidden*2]) #(nxb,2h)
            sequence_video_combine = tf.nn.relu(tf.nn.xw_plus_b(sequence_video_emb_flat,self.w_fuse_video,self.b_fuse_video)) 
            sequence_video_combine = tf.reshape(sequence_video_combine,[n_frame_step,self.batch_size,self.dim_hidden])

            #################################################  for encode sentence #########################################################

            # ordinal sequence
            state1_sentence = ( tf.zeros([self.batch_size, self.lstm_sentence.state_size[0]]), tf.zeros([self.batch_size, self.lstm_sentence.state_size[1]] ) )
            output1_sentence_list = []
            for i in range(n_caption_step):
                with tf.device("/cpu:0"):
                    current_embed_sentence = tf.nn.embedding_lookup(self.Wemb, caption[:,i]) # caption b x n_caption_step
                with tf.variable_scope("LSTM_sentence"):
                    output1_sentence, state1_sentence = self.lstm_sentence_dropout( current_embed_sentence , state1_sentence ) # b x h
                    if i > 0: tf.get_variable_scope().reuse_variables()
                output1_sentence_list.append(output1_sentence)
            ordinal_sequence_sentence_emb = tf.stack(output1_sentence_list) # n x b x h

            # reverse sequence
            reverse_state1_sentence = ( tf.zeros([self.batch_size, self.reverse_lstm_sentence.state_size[0]]), tf.zeros([self.batch_size, self.reverse_lstm_sentence.state_size[1]] ) )
            reverse_output1_sentence_list = []
            for i in range(n_caption_step):
                with tf.device("/cpu:0"):
                    current_embed_sentence = tf.nn.embedding_lookup(self.Wemb,reverse_caption[:,i])
                with tf.variable_scope("Reverse_LSTM_sentence"):
                    reverse_output1_sentence, reverse_state1_sentence = self.reverse_lstm_sentence_dropout( current_embed_sentence, reverse_state1_sentence)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                reverse_output1_sentence_list.append(reverse_output1_sentence)
            reverse_output1_sentence_list = tf.stack(reverse_output1_sentence_list) # n x b x h
            reverse_sequence_sentence_emb = tf.reverse_sequence(reverse_output1_sentence_list, seq_lengths = caption_length, seq_dim = 0, batch_dim = 1)

            sequence_sentence_emb = tf.concat([ordinal_sequence_sentence_emb,reverse_sequence_sentence_emb],2)
            sequence_sentence_emb_flat = tf.reshape(sequence_sentence_emb,[-1,self.dim_hidden*2]) #(nxb,2h)
            sequence_sentence_combine = tf.nn.relu(tf.nn.xw_plus_b(sequence_sentence_emb_flat,self.w_fuse_sentence,self.b_fuse_sentence)) 
            sequence_sentence_combine = tf.reshape(sequence_sentence_combine,[n_caption_step,self.batch_size,self.dim_hidden])
            sentence_embedding = tf.multiply(sequence_sentence_combine,caption_emb_mask)
            sentence_embedding = tf.reduce_sum(sentence_embedding,0) # b x h

            ################################################   alternative attention module ##################################################

            # attend video with sentence fts
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step,1,1]) # n x h x 1
            image_part = tf.matmul(sequence_video_combine, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_frame_step,1,1])) + self.embed_att_ba # n x b x h
            e = tf.tanh(tf.matmul(sentence_embedding, self.embed_att_Wa) + image_part)
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            attention_list = tf.multiply(alphas, sequence_video_combine) # n x b x h ; multiply is element x element
            attended_video_fts = tf.reduce_sum(attention_list,0) # b x h       #  visual feature by soft-attention weighted sum
            predict_attention_weights_v = tf.transpose(tf.div(e_hat_exp,denomin))  # b x n

            #attend sentence with video fts
            brcst_v_q = tf.tile(tf.expand_dims(self.embed_att_w_q,0), [n_caption_step,1,1])
            sentence_part = tf.matmul(sequence_sentence_combine,tf.tile(tf.expand_dims(self.embed_att_U_qa,0),[n_caption_step,1,1])) + self.embed_att_b_qa
            e_q = tf.tanh(tf.matmul(attended_video_fts,self.embed_att_W_qa)+sentence_part)
            e_q = tf.matmul(e_q,brcst_v_q)
            e_q = tf.reduce_sum(e_q,2)
            e_hat_exp_q = tf.multiply(tf.transpose(caption_mask),tf.exp(e_q))
            denomin_q = tf.reduce_sum(e_hat_exp_q,0)
            denomin_q = denomin_q + tf.to_float(tf.equal(denomin_q, 0))
            alphas_q = tf.tile(tf.expand_dims(tf.div(e_hat_exp_q,denomin_q),2),[1,1,self.dim_hidden])
            attention_list_q = tf.multiply(alphas_q, sequence_sentence_combine)
            attended_sentence_fts = tf.reduce_sum(attention_list_q,0) # b x h
            predict_attention_weights_q = tf.transpose(tf.div(e_hat_exp_q,denomin_q))

            #attend video with attended sentence fts
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step,1,1]) # n x h x 1
            image_part = tf.matmul(sequence_video_combine, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_frame_step,1,1])) + self.embed_att_ba # n x b x h
            e = tf.tanh(tf.matmul(attended_sentence_fts, self.embed_att_Wa) + image_part)
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            attention_list = tf.multiply(alphas, sequence_video_combine) # n x b x h ; multiply is element x element
            attended_video_fts = tf.reduce_sum(attention_list,0) # b x h       #  visual feature by soft-attention weighted sum
            predict_attention_weights_v = tf.transpose(tf.div(e_hat_exp,denomin))  # b x n

            ############################################## regress module ################################################################

            if self.model_mode == 'AF':
                multimodel_fts_concat = tf.concat([attended_video_fts, attended_sentence_fts],1)
                multimodel_fts_hidden = tf.nn.relu(tf.nn.xw_plus_b(multimodel_fts_concat, self.w_fuse, self.b_fuse))
                if self.regress_layer_num == 1:
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(multimodel_fts_hidden, self.regress_w, self.regress_b)) # b x 2
                else:
                    predict_hidden =  tf.nn.xw_plus_b(multimodel_fts_hidden, self.regress_w, self.regress_b) # b x h
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_hidden, self.regress_w_2, self.regress_b_2)) # b x 2
            else: # AW
                if self.regress_layer_num == 1:
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_attention_weights_v, self.regress_w, self.regress_b)) # b x 2
                else:
                    predict_hidden =  tf.nn.xw_plus_b(predict_attention_weights_v, self.regress_w, self.regress_b) # b x h
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_hidden, self.regress_w_2, self.regress_b_2)) # b x 2

            loss_regression_tmp = smooth_l1_loss(predict_location,video_location)
            loss_regression =  tf.multiply(tf.constant(alpha_regress),tf.reduce_mean(loss_regression_tmp))

            batch_ground_weight = tf.multiply(video_attention_weights, tf.log(predict_attention_weights_v+tf.constant(1e-12)))
            batch_video_attention_weight = tf.reduce_sum(batch_ground_weight,1)
            loss_attention = -tf.reduce_mean( tf.multiply(loss_mask,batch_video_attention_weight) )
            loss_attention = tf.multiply(tf.constant(alpha_attention),loss_attention)

            loss = loss_regression + loss_attention

            return loss, loss_regression, loss_attention, video, video_mask, video_location, video_attention_weights, \
                    caption, caption_mask, caption_emb_mask, loss_mask, predict_attention_weights_v, predict_attention_weights_q, \
                    reverse_caption, caption_length


    def build_localizer(self):

       with tf.variable_scope(tf.get_variable_scope()) as model_scope:

            video = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step, self.dim_image]) # b x n x d
            video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_frame_step]) # b x n

            caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step]) # b x n_caption_step
            caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step]) # b x n_caption_step
            caption_emb_mask = tf.placeholder(tf.float32, [n_caption_step,self.batch_size, dim_hidden]) # indicate where do caption end
            reverse_caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step])
            caption_length = tf.placeholder(tf.int32,[self.batch_size])

            if self.dim_image == 4096:
                video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
                video_emb = tf.nn.xw_plus_b(video_flat, self.embed_video_w, self.embed_video_b) # (b x n) x h
                video_emb = tf.reshape(video_emb, [self.batch_size, self.n_frame_step, self.dim_image_encode]) # b x n x h
            else:
                video_emb = video

            ################################################ for encode video with LSTM ###################################################

            # ordinal sequence
            state1_video = ( tf.zeros([self.batch_size, self.lstm_video.state_size[0]]), tf.zeros([self.batch_size, self.lstm_video.state_size[1]] ) )
            output_video_list = []
            for i in range(n_frame_step):
                current_embed_video = video_emb[:,i,:] # bxnxd
                with tf.variable_scope("LSTM_video"):
                    output_video, state1_video = self.lstm_video_dropout(current_embed_video, state1_video)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                output_video_list.append(output_video) # nxbxh
            ordinal_sequence_video_emb = tf.stack(output_video_list) # n x b x h

            # reverse sequence
            reverse_state1_video = ( tf.zeros([self.batch_size, self.reverse_lstm_video.state_size[0]]), tf.zeros([self.batch_size, self.reverse_lstm_video.state_size[1]] ) )
            reverse_output_video_list = []
            for i in range(n_frame_step):
                current_embed_video = video_emb[:,n_frame_step-i-1,:] # bxnxd
                with tf.variable_scope("Reverse_LSTM_video"):
                    reverse_output_video, reverse_state1_video = self.reverse_lstm_video_dropout(current_embed_video, reverse_state1_video)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                reverse_output_video_list.append(reverse_output_video) # nxbxh
            new_reverse_output_video_list = []
            for i in range(n_frame_step):
                new_reverse_output_video_list.append(reverse_output_video_list[n_frame_step-i-1])
            reverse_sequence_video_emb = tf.stack(new_reverse_output_video_list) # n x b x h

            sequence_video_emb = tf.concat([ordinal_sequence_video_emb,reverse_sequence_video_emb],2) # nxbx2h
            sequence_video_emb_flat = tf.reshape(sequence_video_emb,[-1,self.dim_hidden*2]) #(nxb,2h)
            sequence_video_combine = tf.nn.relu(tf.nn.xw_plus_b(sequence_video_emb_flat,self.w_fuse_video,self.b_fuse_video)) 
            sequence_video_combine = tf.reshape(sequence_video_combine,[n_frame_step,self.batch_size,self.dim_hidden])

            #################################################  for encode sentence #########################################################

            # ordinal sequence
            state1_sentence = ( tf.zeros([self.batch_size, self.lstm_sentence.state_size[0]]), tf.zeros([self.batch_size, self.lstm_sentence.state_size[1]] ) )
            output1_sentence_list = []
            for i in range(n_caption_step):
                with tf.device("/cpu:0"):
                    current_embed_sentence = tf.nn.embedding_lookup(self.Wemb, caption[:,i]) # caption b x n_caption_step
                with tf.variable_scope("LSTM_sentence"):
                    output1_sentence, state1_sentence = self.lstm_sentence_dropout( current_embed_sentence , state1_sentence ) # b x h
                    if i > 0: tf.get_variable_scope().reuse_variables()
                output1_sentence_list.append(output1_sentence)
            ordinal_sequence_sentence_emb = tf.stack(output1_sentence_list) # n x b x h

            # reverse sequence
            reverse_state1_sentence = ( tf.zeros([self.batch_size, self.reverse_lstm_sentence.state_size[0]]), tf.zeros([self.batch_size, self.reverse_lstm_sentence.state_size[1]] ) )
            reverse_output1_sentence_list = []
            for i in range(n_caption_step):
                with tf.device("/cpu:0"):
                    current_embed_sentence = tf.nn.embedding_lookup(self.Wemb,reverse_caption[:,i])
                with tf.variable_scope("Reverse_LSTM_sentence"):
                    reverse_output1_sentence, reverse_state1_sentence = self.reverse_lstm_sentence_dropout( current_embed_sentence, reverse_state1_sentence)
                    if i > 0: tf.get_variable_scope().reuse_variables()
                reverse_output1_sentence_list.append(reverse_output1_sentence)
            reverse_output1_sentence_list = tf.stack(reverse_output1_sentence_list) # n x b x h
            reverse_sequence_sentence_emb = tf.reverse_sequence(reverse_output1_sentence_list, seq_lengths = caption_length, seq_dim = 0, batch_dim = 1)

            sequence_sentence_emb = tf.concat([ordinal_sequence_sentence_emb,reverse_sequence_sentence_emb],2)
            sequence_sentence_emb_flat = tf.reshape(sequence_sentence_emb,[-1,self.dim_hidden*2]) #(nxb,2h)
            sequence_sentence_combine = tf.nn.relu(tf.nn.xw_plus_b(sequence_sentence_emb_flat,self.w_fuse_sentence,self.b_fuse_sentence)) 
            sequence_sentence_combine = tf.reshape(sequence_sentence_combine,[n_caption_step,self.batch_size,self.dim_hidden])
            sentence_embedding = tf.multiply(sequence_sentence_combine,caption_emb_mask)
            sentence_embedding = tf.reduce_sum(sentence_embedding,0) # b x h

            ################################################   alternative attention module ##################################################

            # attend video with sentence fts
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step,1,1]) # n x h x 1
            image_part = tf.matmul(sequence_video_combine, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_frame_step,1,1])) + self.embed_att_ba # n x b x h
            e = tf.tanh(tf.matmul(sentence_embedding, self.embed_att_Wa) + image_part)
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            attention_list = tf.multiply(alphas, sequence_video_combine) # n x b x h ; multiply is element x element
            attended_video_fts = tf.reduce_sum(attention_list,0) # b x h       #  visual feature by soft-attention weighted sum
            predict_attention_weights_v = tf.transpose(tf.div(e_hat_exp,denomin))  # b x n

            #attend sentence with video fts
            brcst_v_q = tf.tile(tf.expand_dims(self.embed_att_w_q,0), [n_caption_step,1,1])
            sentence_part = tf.matmul(sequence_sentence_combine,tf.tile(tf.expand_dims(self.embed_att_U_qa,0),[n_caption_step,1,1])) + self.embed_att_b_qa
            e_q = tf.tanh(tf.matmul(attended_video_fts,self.embed_att_W_qa)+sentence_part)
            e_q = tf.matmul(e_q,brcst_v_q)
            e_q = tf.reduce_sum(e_q,2)
            e_hat_exp_q = tf.multiply(tf.transpose(caption_mask),tf.exp(e_q))
            denomin_q = tf.reduce_sum(e_hat_exp_q,0)
            denomin_q = denomin_q + tf.to_float(tf.equal(denomin_q, 0))
            alphas_q = tf.tile(tf.expand_dims(tf.div(e_hat_exp_q,denomin_q),2),[1,1,self.dim_hidden])
            attention_list_q = tf.multiply(alphas_q, sequence_sentence_combine)
            attended_sentence_fts = tf.reduce_sum(attention_list_q,0) # b x h
            predict_attention_weights_q = tf.transpose(tf.div(e_hat_exp_q,denomin_q))

            #attend video with attended sentence fts
            brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_frame_step,1,1]) # n x h x 1
            image_part = tf.matmul(sequence_video_combine, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_frame_step,1,1])) + self.embed_att_ba # n x b x h
            e = tf.tanh(tf.matmul(attended_sentence_fts, self.embed_att_Wa) + image_part)
            e = tf.matmul(e, brcst_w)
            e = tf.reduce_sum(e,2) # n x b
            e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
            denomin = tf.reduce_sum(e_hat_exp,0) # b
            denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
            alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            attention_list = tf.multiply(alphas, sequence_video_combine) # n x b x h ; multiply is element x element
            attended_video_fts = tf.reduce_sum(attention_list,0) # b x h       #  visual feature by soft-attention weighted sum
            predict_attention_weights_v = tf.transpose(tf.div(e_hat_exp,denomin))  # b x n

            ############################################## regress module ################################################################

            if self.model_mode == 'AF':
                multimodel_fts_concat = tf.concat([attended_video_fts, attended_sentence_fts],1)
                multimodel_fts_hidden = tf.nn.relu(tf.nn.xw_plus_b(multimodel_fts_concat, self.w_fuse, self.b_fuse))
                if self.regress_layer_num == 1:
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(multimodel_fts_hidden, self.regress_w, self.regress_b)) # b x 2
                else:
                    predict_hidden =  tf.nn.xw_plus_b(multimodel_fts_hidden, self.regress_w, self.regress_b) # b x h
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_hidden, self.regress_w_2, self.regress_b_2)) # b x 2
            else: # AW
                if self.regress_layer_num == 1:
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_attention_weights_v, self.regress_w, self.regress_b)) # b x 2
                else:
                    predict_hidden =  tf.nn.xw_plus_b(predict_attention_weights_v, self.regress_w, self.regress_b) # b x h
                    predict_location = tf.nn.relu(tf.nn.xw_plus_b(predict_hidden, self.regress_w_2, self.regress_b_2)) # b x 2

            return video, video_mask, caption, caption_mask, caption_emb_mask, predict_location,\
                   predict_attention_weights_v, predict_attention_weights_q, reverse_caption, caption_length



def get_video_data_HL(video_data_path):
    files = open(video_data_path)
    List = []
    for ele in files:
        List.append(ele[:-1])
    return np.array(List)


def get_video_data_jukin(video_data_path_train, video_data_path_val):
    title = []
    video_list_train = get_video_data_HL(video_data_path_train) # get h5 file list
    video_list_val = get_video_data_HL(video_data_path_val)
    for ele in video_list_train:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
            title.append(batch_title[i])
    for ele in video_list_val:
        batch_data = h5py.File(ele,'r')
        batch_fname = batch_data['fname']
        batch_title = batch_data['title']
        for i in xrange(len(batch_fname)):
            title.append(batch_title[i])
    title = np.array(title)
    video_caption_data = pd.DataFrame({'Description':title})
    return video_caption_data, video_list_train, video_list_val



def preProBuildWordVocab(logging,sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    logging.info('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    logging.info('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector



def localizing_one(sess, video_feat_path, video_tf, video_mask_tf, caption_tf, reverse_caption_tf, caption_length_tf, caption_mask_tf, caption_emb_mask_tf, \
                   predict_location_tf, predict_attention_weights_v_tf, predict_attention_weights_q_tf):

    save_video_name = []
    save_duration = []
    save_caption = []
    save_grountruth_timestamps = []
    save_predict_timestamps = []
    save_predict_attention_weight_v = []
    save_predict_attention_weight_q = []

    test_data_batch = h5py.File(video_feat_path,'r')
    current_captions_tmp = test_data_batch['title']
    current_captions = []
    for ind in range(batch_size):
        current_captions.append(current_captions_tmp[ind])
    current_captions = np.array(current_captions)
    for ind in range(batch_size):
        for c in string.punctuation: 
            current_captions[ind] = current_captions[ind].replace(c,'')
    current_video_feat = np.zeros((batch_size, n_frame_step, dim_image))
    current_video_mask = np.zeros((batch_size, n_frame_step))

    wordtoix = (np.load(wordtoix_path)).tolist()

    for ind in xrange(batch_size):
        current_video_feat[ind,:,:] = test_data_batch['data'][:n_frame_step,ind,:]
        idx = np.where(test_data_batch['label'][:,ind] != -1)[0]
        if(len(idx) == 0):
            continue
        current_video_mask[ind,:idx[-1]+1] = 1.

    current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
    current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_step-1)
    current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
    current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
    nonzeros = np.array( map(lambda x: (x != 0).sum(), current_caption_matrix ))
    for ind, row in enumerate(current_caption_masks):
        row[:nonzeros[ind]] = 1
    current_caption_emb_mask = np.zeros((n_caption_step,batch_size,dim_hidden)) + 0.0
    for ii in range(batch_size):
        current_caption_emb_mask[:nonzeros[ii],ii,:] = 1.0 / (nonzeros[ii]*1.0)

    reverse_current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[::-1] if word in wordtoix], current_captions)
    reverse_current_caption_matrix = sequence.pad_sequences(reverse_current_caption_ind, padding='post', maxlen=n_caption_step-1)
    reverse_current_caption_matrix = np.hstack( [reverse_current_caption_matrix, np.zeros( [len(reverse_current_caption_matrix),1]) ] ).astype(int)

    current_caption_length = np.zeros((batch_size))
    for ind in range(batch_size):
        current_caption_length[ind] = np.sum(current_caption_masks[ind])

    predict_video_location, predict_attention_weight_v, predict_attention_weight_q = sess.run([predict_location_tf, predict_attention_weights_v_tf, predict_attention_weights_q_tf], \
                                                                                                feed_dict={video_tf:current_video_feat, \
                                                                                                           video_mask_tf:current_video_mask, \
                                                                                                           caption_tf: current_caption_matrix, \
                                                                                                           caption_mask_tf: current_caption_masks,\
                                                                                                           caption_emb_mask_tf:current_caption_emb_mask,\
                                                                                                           reverse_caption_tf: reverse_current_caption_matrix,\
                                                                                                           caption_length_tf: current_caption_length})

    for ind in xrange(batch_size):
        save_video_name.append(test_data_batch['fname'][ind])
        save_duration.append(test_data_batch['duration'][ind])
        save_caption.append(test_data_batch['title'][ind])
        save_grountruth_timestamps.append(test_data_batch['timestamps'][ind])
        save_predict_timestamps.append(predict_video_location[ind])
        save_predict_attention_weight_v.append(predict_attention_weight_v[ind])
        save_predict_attention_weight_q.append(predict_attention_weight_q[ind])

    return save_video_name, save_duration, save_caption, save_grountruth_timestamps, \
           save_predict_timestamps, save_predict_attention_weight_v, save_predict_attention_weight_q



def localizing_all(logging, sess, val_data, video_tf, video_mask_tf, caption_tf, reverse_caption_tf, caption_length_tf, caption_mask_tf, caption_emb_mask_tf, \
                        predict_location_tf, predict_attention_weights_v_tf, predict_attention_weights_q_tf, test_model_path, result_save_dir):

    video_name_list = []
    duration_list = []
    caption_list = []
    grountruth_timestamps_list = []
    predict_timestamps_list = []
    predict_attention_weight_v_list = []
    predict_attention_weight_q_list = []

    for _, video_feat_path in enumerate(val_data):
        logging.info(video_feat_path)

        [video_name, duration, caption, groundtruth_timestamps, predict_timestamps, predict_attention_weights_v, predict_attention_weights_q] = \
        localizing_one(sess, video_feat_path, video_tf, video_mask_tf, caption_tf, reverse_caption_tf, caption_length_tf, caption_mask_tf, caption_emb_mask_tf,\
                       predict_location_tf, predict_attention_weights_v_tf, predict_attention_weights_q_tf)

        video_name_list = video_name_list + video_name
        duration_list = duration_list + duration
        caption_list = caption_list + caption
        grountruth_timestamps_list = grountruth_timestamps_list + groundtruth_timestamps
        predict_timestamps_list = predict_timestamps_list + predict_timestamps
        predict_attention_weight_v_list = predict_attention_weight_v_list + predict_attention_weights_v
        predict_attention_weight_q_list = predict_attention_weight_q_list + predict_attention_weights_q

    time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    index = test_model_path.find('model-')
    index_str = test_model_path[index:]
    final_save_path = result_save_dir+'/'+time_stamp+'_predict_result_'+index_str+'.npz'
    np.savez(final_save_path, video = video_name_list, duration = duration_list, caption = caption_list, \
        groundtruth_timestamps = grountruth_timestamps_list, predict_timestamps = predict_timestamps_list, \
        predict_attention_weights_v = predict_attention_weight_v_list, predict_attention_weights_q = predict_attention_weight_q_list)
    calculate_mean_IOU(logging,final_save_path,n_frame_step)
    calculate_map_IOU(logging,final_save_path,n_frame_step)


def train(sub_dir, logging, regress_layer_num, model_save_dir, result_save_dir):

    meta_data, train_data, val_data = get_video_data_jukin(video_data_path_train, video_data_path_val)
    captions = meta_data['Description'].values
    for c in string.punctuation:
        captions = map(lambda x: x.replace(c, ''), captions)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(logging, captions, word_count_threshold=1)

    np.save(ixtoword_path, ixtoword)
    np.save(wordtoix_path, wordtoix)
    get_word_embedding(word_embedding_path,wordtoix_path,ixtoword_path,word_fts_path)
    word_emb_init = np.array(np.load(open(word_fts_path)).tolist(),np.float32)

    logging.info('regress_layer_num = {:f}'.format(regress_layer_num*1.0))

    model = ABLR(
            model_mode = model_mode,
            word_emb_init=word_emb_init,
            dim_image=dim_image,
            dim_image_encode = dim_image_encode,
            dim_word = dim_word,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_frame_step=n_frame_step,
            drop_out_rate = 0.5,
            regress_layer_num = regress_layer_num,
            dim_hidden_regress = dim_hidden_regress,
            bias_init_vector=None)

    tf_loss, tf_loss_regression, tf_loss_attention, tf_video, tf_video_mask, tf_video_location, tf_video_attention_weights,\
    tf_caption, tf_caption_mask, tf_caption_emb_mask, tf_loss_mask, tf_predict_attention_weights_v, tf_predict_attention_weights_q,\
    tf_reverse_caption, tf_caption_length = model.build_model()


    test_video_tf, test_video_mask_tf, test_caption_tf, test_caption_mask_tf, test_caption_emb_mask_tf,\
    test_predict_location_tf, test_predict_attention_weights_v_tf, test_predict_attention_weights_q_tf,\
    test_reverse_caption_tf, test_caption_length_tf = model.build_localizer()


    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gvs = optimizer.compute_gradients(tf_loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)


    ############################################# start training ####################################################
    tf.initialize_all_variables().run()
    with tf.device("/cpu:0"):
        if retrain:
            saver.restore(sess, MODEL)

    all_batches_count = 0
    tStart_total = time.time()
    for epoch in range(n_epochs):
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data)) # each item in loss_epoch record the loss of this h5 file

        for current_batch_file_idx in xrange(len(train_data)):

            all_batches_count = all_batches_count +  1
            logging.info("current_batch_file_idx = {:d}".format(current_batch_file_idx))
            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx],'r')
            current_captions_tmp = current_batch['title']
            current_captions = []
            for ind in range(batch_size):
                current_captions.append(current_captions_tmp[ind])
            current_captions = np.array(current_captions)
            for ind in range(batch_size):
                for c in string.punctuation: 
                    current_captions[ind] = current_captions[ind].replace(c,'')
            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_weights = np.zeros((batch_size,n_frame_step))
            current_video_masks = np.zeros((batch_size, n_frame_step))
            current_video_location = np.zeros((batch_size,2))
            current_loss_mask = np.ones(batch_size)+0.0
            for i in range(batch_size):
                current_captions[i] = current_captions[i].strip()
                if current_captions[i] == '':
                    current_captions[i] = '.'
                    current_loss_mask[i] = 0.0
            for ind in xrange(batch_size):
                current_feats[ind,:,:] = current_batch['data'][:n_frame_step,ind,:]
                current_video_location[ind,:] = np.array(current_batch['norm_timestamps'][ind])
                if dataset == 'ActivityNet':
                    current_weights[ind,:] = np.array(current_batch['weights'][:n_frame_step,ind])
                elif dataset == 'TACOS':
                    left = int(current_video_location[ind][0]*n_frame_step)
                    right = min(n_frame_step,int(current_video_location[ind][1]*n_frame_step)+1)
                    current_weights[ind,left:right] = 1.0 / (right - left + 1)
                idx = np.where(current_batch['label'][:,ind] != -1)[0] #find this video's segment finish point
                if len(idx) == 0:
                    continue
                current_video_masks[ind,:idx[-1]+1] = 1

            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum(), current_caption_matrix )) # save the sentence length of this batch
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            current_caption_emb_mask = np.zeros((n_caption_step,batch_size,dim_hidden)) + 0.0
            for ii in range(batch_size):
                current_caption_emb_mask[:nonzeros[ii],ii,:] = 1.0 / (nonzeros[ii]*1.0)

            reverse_current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[::-1] if word in wordtoix], current_captions)
            reverse_current_caption_matrix = sequence.pad_sequences(reverse_current_caption_ind, padding='post', maxlen=n_caption_step-1)
            reverse_current_caption_matrix = np.hstack( [reverse_current_caption_matrix, np.zeros( [len(reverse_current_caption_matrix),1]) ] ).astype(int)

            current_caption_length = np.zeros((batch_size))
            for ind in range(batch_size):
                current_caption_length[ind] = np.sum(current_caption_masks[ind])

            _, loss_val, loss_regression, loss_attention, train_attention_v, train_attention_q = sess.run(
                    [train_op, tf_loss, tf_loss_regression, tf_loss_attention, tf_predict_attention_weights_v, tf_predict_attention_weights_q],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask: current_video_masks,
                        tf_video_location: current_video_location,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks,
                        tf_caption_emb_mask: current_caption_emb_mask,
                        tf_loss_mask: current_loss_mask,
                        tf_video_attention_weights: current_weights,
                        tf_reverse_caption: reverse_current_caption_matrix,
                        tf_caption_length: current_caption_length
                        })
            loss_epoch[current_batch_file_idx] = loss_val

            logging.info("loss = {:f}  loss_regression = {:f}  loss_attention = {:f}".format(loss_val,loss_regression,loss_attention))
            tStop = time.time()


        logging.info("Epoch: {:d} done.".format(epoch))
        tStop_epoch = time.time()
        logging.info('Epoch Time Cost: {:f} s'.format(round(tStop_epoch - tStart_epoch,2)))


        #################################################### localize ################################################################################################
        if np.mod(epoch, iter_localize) == 0 or epoch == n_epochs -1:

            logging.info('Epoch {:d} is done. Saving the model ...'.format(epoch))
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)

            logging.info('Localizing videos ...'.format(epoch))
            test_model_path = model_save_dir + '/model-' + str(epoch)
            logging.info('Localizing testing set ...')
            localizing_all(logging, sess, val_data, test_video_tf, test_video_mask_tf,\
                            test_caption_tf, test_reverse_caption_tf, test_caption_length_tf, test_caption_mask_tf, test_caption_emb_mask_tf, \
                            test_predict_location_tf, test_predict_attention_weights_v_tf, \
                            test_predict_attention_weights_q_tf, test_model_path, result_save_dir)


    logging.info("Finally, saving the model ...")
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=epoch)
    tStop_total = time.time()
    logging.info("Total Time Cost: {:f} s".format(round(tStop_total - tStart_total,2)))



def localize(sub_dir, logging, regress_layer_num, result_save_dir, model_path, video_feat_path=video_feat_path):
    meta_data, train_data, val_data = get_video_data_jukin(video_data_path_train, video_data_path_val)
    ixtoword = pd.Series(np.load(ixtoword_path).tolist())
    word_emb_init = np.array(np.load(open(word_fts_path)).tolist(),np.float32)
    
    model = ABLR(
            model_mode = model_mode,
            word_emb_init=word_emb_init,
            dim_image=dim_image,
            dim_image_encode = dim_image_encode,
            dim_word = dim_word,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_frame_step=n_frame_step,
            drop_out_rate = 0,
            regress_layer_num = regress_layer_num,
            dim_hidden_regress = dim_hidden_regress,
            bias_init_vector=None)


    test_video_tf, test_video_mask_tf, test_caption_tf, test_caption_mask_tf, test_caption_emb_mask_tf,\
    test_predict_location_tf, test_predict_attention_weights_v_tf, test_predict_attention_weights_q_tf,\
    test_reverse_caption_tf, test_caption_length_tf = model.build_localizer()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    localizing_all(logging, sess, val_data, test_video_tf, test_video_mask_tf,\
                    test_caption_tf, test_reverse_caption_tf, test_caption_length_tf, test_caption_mask_tf, test_caption_emb_mask_tf, \
                    test_predict_location_tf, test_predict_attention_weights_v_tf, \
                    test_predict_attention_weights_q_tf, model_path, result_save_dir)


if __name__ == '__main__':

    args = parse_args()
    sub_dir, logging, regress_layer_num, model_save_dir, result_save_dir = make_prepare_path(args.regress_layer_num,dataset)

    if args.task == 'train':
        with tf.device('/gpu:0'):
            train(sub_dir, logging, regress_layer_num, model_save_dir, result_save_dir)
    elif args.task == 'localize':
        with tf.device('/gpu:0'):
            localize(sub_dir, logging, regress_layer_num, result_save_dir = result_save_dir, model_path = args.model)