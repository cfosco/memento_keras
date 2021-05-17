# Imports
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, GlobalAveragePooling2D, Input, TimeDistributed
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import os
import cv2
from cv2 import imread, resize
from scipy.stats import spearmanr
import keras.backend as K
from keras.optimizers import Adam, SGD
# import seaborn as sns
import pandas as pd
import sys
from PIL import Image
import json
import pickle
sys.path.append('../')
# from imgaug import augmenters as iaa
import math
from keras_training import MultiGPUCheckpoint
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from IPython.display import clear_output, display

import i3d_config as cfg


def load_2_frames_opencv(video_file, space_between_frames, verbose=True, is_train = False, img_size = 224):
    cap = cv2.VideoCapture(video_file)
    if cap:
        print('File %s successfully opened in load_2_frames_opencv' % video_file)
    else:
        print('File %s load unsuccessful' % video_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not is_train:
        init_frame_idx = 5
    else:
        init_frame_idx = np.random.randint(0, n_frames-space_between_frames-1)

    end_frame_idx = init_frame_idx + space_between_frames

    if verbose:
        print('video_file:', video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(video_file.split('/')[-1], 'loaded. \nwidth, height, n_frames, fps:',width, height, n_frames, fps)

    success=True
    i=0
    found_frames=False

    while success and not found_frames:
        success,frame = cap.read()
        if success:
            if i == init_frame_idx:
                init_frame = resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), img_size)/255.0
            elif i == end_frame_idx:
                end_frame = resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), img_size)/255.0
                found_frames = True
            i+=1

    return init_frame, end_frame


def load_2_frames(frames_folder, space_between_frames, verbose=True, is_train = False, img_size = 224):

    frame_names = os.listdir(frames_folder)

    n_frames = len(frame_names)

    if not is_train:
        init_frame_idx = 5
    else:
        init_frame_idx = np.random.randint(0, n_frames-space_between_frames-1)

    end_frame_idx = init_frame_idx + space_between_frames


    init_frame = cv2.imread(os.path.join(frames_folder,frame_names[init_frame_idx]))
    init_frame = resize(cv2.cvtColor(init_frame, cv2.COLOR_BGR2RGB), img_size)/255.0

    end_frame = cv2.imread(os.path.join(frames_folder,frame_names[end_frame_idx]))
    end_frame = resize(cv2.cvtColor(end_frame, cv2.COLOR_BGR2RGB), img_size)/255.0

    return init_frame, end_frame

class ShowFuturePred(keras.callbacks.Callback):
    def __init__(self, gen, freq=100, res_to_show=1):
        self.freq=freq
        self.gen = gen
        self.res_to_show=res_to_show

    def on_train_begin(self, logs={}):
        self.counter=0

    def on_batch_end(self, batch, logs={}):
        self.counter+=1

        # plot input image of batch
        if not self.counter%self.freq :
            if isinstance(self.gen, keras.utils.Sequence):
                batch = self.gen.__getitem__(self.counter%len(self.gen))
            else:
                batch = next(self.gen)

            for i in range(min(self.res_to_show, len(batch[0]))):
                plt.figure(figsize=[9,9])
                plt.subplot(2,2,1)
                plt.imshow(batch[0][i])
                plt.title('Initial frame (input to net)')

                # plot frame to predict of batch
                plt.subplot(2,2,2)
                plt.imshow(batch[2][i])
                plt.title('Frame to pred')

                # get predicted representation
                pred = self.model.predict(batch[0])[i]

                target = batch[1][i]

                # Prepare to visualize repr by transforming into 2d heatmap
                root = int(np.floor(np.sqrt(REPR_SIZE)))

                # show target representation
                repres_viz = target
                repres_viz = np.reshape(repres_viz[:root**2], (root,root))
                repres_viz = (repres_viz-np.min(repres_viz))/(np.max(repres_viz)-np.min(repres_viz))
                plt.subplot(2,2,3)
                plt.imshow(repres_viz)
                plt.title('Target repr')

                # show predicted representation
                repres_viz = pred
                repres_viz = np.reshape(repres_viz[:root**2], (root,root))
                repres_viz = (repres_viz-np.min(repres_viz))/(np.max(repres_viz)-np.min(repres_viz))
                plt.subplot(2,2,4)
                plt.imshow(repres_viz)
                plt.title('PREDICTED repr')

                # show MSE between representations
                plt.suptitle('MSE BETWEEN THESE REPRESENTATIONS: '+str(np.mean((target-pred)**2)))
                plt.show()


class MemGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, img_size=(224,224), shuffle=True, augment=False, mixup=False, two_losses=False):
        self.image_filenames, self.labels = np.array(image_filenames), np.array(labels)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.mixup=mixup
        self.two_losses=two_losses

        if augment:
            sometimes = lambda aug: iaa.Sometimes(0.4, aug)

            self.seq = iaa.Sequential([
                    sometimes(iaa.CropAndPad(px=(0, 20))), # crop images from each side by 0 to 16px (randomly chosen)
                    iaa.Fliplr(0.5), # horizontally flip 50% of the images
                    sometimes(iaa.CoarseDropout(p=0.1, size_percent=0.05)),
                    sometimes(iaa.Affine(rotate=(-15, 15)))
                ])


        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        images=[]
        for file_name in batch_x:
#             print(file_name)
            im = imread(file_name)[...,::-1]
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


            images.append(im)

        if self.augment:
            images = self.seq.augment_images(images)


        img_batch = np.array([resize(im/255.0, self.img_size) for im in images])

        if self.augment and self.mixup:
            if np.random.rand()>0.85:
                alpha = np.random.rand()
                blended = img_batch[0]*(1.0-alpha) + img_batch[1]*alpha
                blended_y = batch_y[0]*(1.0-alpha) + batch_y[1]*alpha

#                 print('blending images with alpha %.4f!!!' % alpha)
#                 print('min max of blended', blended.min(), blended.max(), type(blended[0,0,0]))

                img_batch[0] = blended
                batch_y[0] = blended_y

        if self.two_losses:
            return img_batch, [np.array(batch_y),np.array(batch_y)]
        else:
            return img_batch, np.array(batch_y)


    def on_epoch_end(self):
        if self.shuffle:
            idxs = list(range(len(self.image_filenames)))
            np.random.shuffle(idxs)
            self.image_filenames = self.image_filenames[idxs]
            self.labels = self.labels[idxs]





class FramesGenerator(Sequence):

    def __init__(self, vid_filenames, labels=None, batch_size=1, img_size=(224,224)):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        vid_name = self.image_filenames[idx]
        if self.labels:
            batch_y = self.labels[idx]

        frames = load_video_opencv(vid_name)/255.0

        print(frames[0])

        return np.array([resize(im, self.img_size) for im in frames]), np.zeros(len(frames))



# Frame and representation generator

class ReprGenerator(Sequence):

    def __init__(self, filenames,  model_for_repr, data_path, batch_size = 32, img_size=(224,224),
                 shuffle=True, show=False, space_between_frames=5, is_train=False, verbose=False,
                 return_end_frame=False, return_filenames=False, data_type='video'):
        self.filenames = np.array(filenames)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.reprnet = model_for_repr
        self.space_between_frames = space_between_frames
        self.is_train=is_train
        self.show=show
        self.data_path = data_path
        self.verbose = verbose
        self.return_end_frame = return_end_frame
        self.return_filenames = return_filenames
        self.data_type=data_type

        if shuffle:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_names = self.filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        end_frames = []

        for f in batch_names:
            if self.data_type=='video':
                init_frame,end_frame = load_2_frames_opencv(os.path.join(self.data_path,f), space_between_frames=self.space_between_frames,
                                                         is_train=self.is_train, img_size = self.img_size, verbose=self.verbose)
            elif self.data_type=='frames':
                empty_folder = True
                folder = os.path.join(self.data_path,f)
            else:
                raise ValueError('Datatype not supported')



                if self.is_train:

                    while empty_folder:

                        if os.path.isdir(folder):
                            if len(os.listdir(folder)) > 40:
                                init_frame,end_frame = load_2_frames(folder, space_between_frames=self.space_between_frames,
                                                        is_train=self.is_train, img_size = self.img_size, verbose=self.verbose)
                                empty_folder = False
                            else:
                                print('Folder %s has less than 40 frames. Sampling a new random one.' % folder)
                                folder = os.path.join(self.data_path, self.filenames[np.random.randint(0,len(self.filenames))])

                        else:
                            print('Folder %s does not exist. Sampling a new random one.' % folder)
                            folder = os.path.join(self.data_path, self.filenames[np.random.randint(0,len(self.filenames))])

                else:
                    if os.path.isdir(folder) and len(os.listdir(folder)) > 40:
                        init_frame,end_frame = load_2_frames(folder, space_between_frames=self.space_between_frames,
                                                        is_train=self.is_train, img_size = self.img_size, verbose=self.verbose)
                    else:
                        init_frame = end_frame = np.zeros((self.img_size[0], self.img_size[1], 3))


            end_frames.append(end_frame)
            batch_x.append(init_frame)

            if self.show:
                plt.figure()
                plt.imshow(init_frame)
                plt.title('Initial frame (input to net)')
                plt.figure()
                plt.imshow(end_frame)
                plt.title('end frame (frame to get representation on)')
                plt.show()


        start = timer()
        repre_batch = self.reprnet.predict(np.array(end_frames))
        end = timer()
#         print('predict_time:', end-start)

        if self.return_filenames:
            return np.array(batch_x), np.array(repre_batch), batch_names
        elif self.return_end_frame:
            return np.array(batch_x), np.array(repre_batch), np.array(end_frames)
        else:
            return np.array(batch_x), np.array(repre_batch)

    def on_epoch_end(self):
        if self.shuffle:
            idxs = list(range(len(self.filenames)))
            np.random.shuffle(idxs)
            self.filenames = self.filenames[idxs]

class RankCorr(keras.callbacks.Callback):
    def __init__(self, gen_val, true_mem, print_=True, idx_to_look_at=[2], preds_savefile=None, get_gt_from_gen=False):

        if isinstance(true_mem, list):
            true_mem = np.array(true_mem)

        if len(true_mem.shape)<=1 or true_mem.shape[1]==1:
            true_mem = np.array([true_mem])

        self.gen_val = gen_val
        self.true_mem = true_mem
        self.print = print_
        self.idx_to_look_at = idx_to_look_at
        self.preds_savefile = preds_savefile
        self.get_gt_from_gen = get_gt_from_gen


    def on_epoch_end(self, batch, logs={}):


        if self.get_gt_from_gen:
            true_mem = []
            gen_out = []
            for inp, out in self.gen_val:
                gen_out.extend(out[0])
            for i in self.idx_to_look_at:
                true_mem[i] = [g[i] for g in gen_out]


        if self.print:
            print('PREDICTING FOR RANK CORRELATION....')

        preds = self.model.predict_generator(self.gen_val, verbose=self.print, workers=10)


        print("preds[0][:5]",preds[0][:5])
        print("len(preds)",len(preds))
        print("self.true_mem[:5]",self.true_mem[:5])

        if isinstance(preds, list) and len(preds)<5:
            # Case multiple outputs
            print("Preds has multiple outputs!")
            p1 = preds[0]
        else:
            # Case one output
            print("Preds is not a list or has len less than 5, p1=preds")
            p1 = preds

        print("len(p1) (should be bs*num_steps)",len(p1))

        if p1.shape[1]==2: # Case (mem, alpha)
            print("We have p1.shape[1]==2: we are in the case mem,alpha. Extracting mem.")
            mem_preds = [[p[0] for p in p1]]
        elif p1.shape[1]>=5: # Case 8 preds
            print("Found %d predictions in p1. Creating mem_preds based on idx to look at" % p1.shape[1])
            mem_preds=[]
            for i in self.idx_to_look_at:
                mem_preds.append([p[i] for p in p1])
        else:
            mem_preds = [p1]

        print("len(mem_preds)", len(mem_preds))
        print("np.array(mem_preds).shape",np.array(mem_preds).shape)

        rc_val = {}
        for i,m in enumerate(mem_preds):
            rc_val[i]=spearmanr(self.true_mem[i][:len(m)], m)[0]

        logs['rc_val'] = rc_val[0] if len(rc_val)==1 else rc_val[2]

        if self.print:
            print('\n --- VAL RANK CORR: %s --- \n\n' % rc_val )


        if self.preds_savefile is not None:
            print("SAVING PREDICTIONS")
            files = self.gen_val.files
            dict_preds = {files[i]:p1[i] for i in range(len(preds))}
            dict_save = {'preds':dict_preds, 'rc_val': rc_val}
            with open(self.preds_savefile ,"wb+") as f:
                pickle.dump(dict_save, f)

# class InteractivePlot(keras.callbacks.Callback):
#     def __init__(self):
#         pass
#
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.val_rcs = []
#         self.val_mses = []
#         self.logs = []
#         self.batchnr = 0
#         self.icount = 0
#
#     def on_train_end(self, logs={}):
#         pass
#
#     def get_rc():
#         return np.random.randint()
#
#     def on_epoch_end(self, epoch, logs={}):
#         self.batchnr = 0
#         loss_train = logs.get('loss')
#         val_mse = logs.get('val_mse')
#         va_rc = self.get_rc()
#         self.losses.append(loss_train)
#         self.val_rcs.append(val_rc)
#         self.val_mses.append(val_mse)
#
#         self.icount+=1
#
#         self.plot_graph(arrs_to_plot, desc)
#         #plt.savefig(self.logfile.replace('.txt', '.png'), bbox_inches='tight', format='png')
#         plt.show()
#
#     def on_batch_end(self, batch, logs=None):
#         self.batchnr+=1
#
#         if self.batchnr % 10 == 0:
#             self.losses.append(logs.get('loss'))
#             self.plot_graph()
#
#     def plot_graph():
#         arrs_to_plot = [self.losses, self.val_mses, self.val_rcs]
#         desc = ['loss', 'val mse', 'val rc']
#         clear_output(wait=True)
#         plt.figure(figsize=(14,10))
#         for i in range(len(arrs_to_plot)):
#             plt.subplot(1, len(arrs_to_plot), i+1)
#             plt.plot(range(len(arrs_t_plot[i])), arrs_to_plot[i], label=desc[i])
#             plt.legend()


def step_decay(init_lr = 0.0001, drop = 0.1, epochs_drop = 3.0, min_lr=1e-7):
    '''Function fed to keras' LearningRateScheduler to control lr decay'''
    def inner(epoch):
        lrate = init_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        if lrate < min_lr: lrate = min_lr
        if not (epoch+1)%epochs_drop:
            print('REDUCING LR. NEW LR IS:', lrate)
        return lrate
    return inner


def loss_alpha(bs=8, T=180., pts=100):

    def loss(y_true, y_pred):
        base_true = y_true[:, 0]
        alpha_true = y_true[:, 1]
        base_pred = y_pred[:, 0]
        alpha_pred = y_pred[:, 1]

        x = K.repeat_elements(K.expand_dims(K.constant(np.linspace(0, 180, pts)), axis=0), rep=bs, axis=0)
        base_true = K.repeat_elements(K.expand_dims(base_true, axis=1), rep=pts, axis=1)
        base_pred = K.repeat_elements(K.expand_dims(base_pred, axis=1), rep=pts, axis=1)

        alpha_true = K.repeat_elements(K.expand_dims(alpha_true, axis=1), rep=pts, axis=1)
        alpha_pred = K.repeat_elements(K.expand_dims(alpha_pred, axis=1), rep=pts, axis=1)

        y_true = alpha_true * (x - T) + base_true
        y_pred = alpha_pred * (x - T) + base_pred
        return K.mean(K.sum(K.abs(y_true - y_pred), axis=1))

    return loss


def loss_alpha_mse(a=1, b=100, c=10, bs=8, T=180., pts=100, print_=False):

    def inner(y_true, y_pred):
        base_true = y_true[:, 0]
        alpha_true = y_true[:, 1]
        base_pred = y_pred[:, 0]
        alpha_pred = y_pred[:, 1]

        mse_mem = (base_true-base_pred)**2
        mse_alpha = (alpha_true*10-alpha_pred*10)**2

        x = K.repeat_elements(K.expand_dims(K.constant(np.linspace(0, 180, pts)), axis=0), rep=bs, axis=0)
        base_true = K.repeat_elements(K.expand_dims(base_true, axis=1), rep=pts, axis=1)
        base_pred = K.repeat_elements(K.expand_dims(base_pred, axis=1), rep=pts, axis=1)

        alpha_true = K.repeat_elements(K.expand_dims(alpha_true, axis=1), rep=pts, axis=1)
        alpha_pred = K.repeat_elements(K.expand_dims(alpha_pred, axis=1), rep=pts, axis=1)

        y_true = alpha_true * (x - T) + base_true
        y_pred = alpha_pred * (x - T) + base_pred

        if print_:
            print('line_loss, K.mean(mse_mem), K.mean(mse_alpha): ',
                  a*K.get_value(K.mean(K.sum(K.abs(y_true - y_pred), axis=1))),
                  b*K.get_value(K.mean(mse_mem)),
                  c*K.get_value(K.mean(mse_alpha)))

        return a*K.mean(K.sum(K.abs(y_true - y_pred), axis=1)) + b*K.mean(mse_mem) + c*K.mean(mse_alpha)

    return inner


def custom_mse(y_true, y_pred):
    pass


def get_sample_weights(scores, n_bins=20, factor=0.5, use_inv_freq=True):

    plt.figure(figsize=[10,4])
    plt.subplot(1,2,1)
    plt.hist(scores, bins=n_bins, alpha=0.5)
    heights, edges = np.histogram(scores, bins=n_bins)

#     print(heights, edges)
    max_h = max(heights)

    idxs = np.digitize(scores, edges)
    idxs[idxs>n_bins]=n_bins

    h = heights[idxs-1]
    if use_inv_freq:
        weights = 1/h
    else:
        weights = 0.5+factor*(max_h-h)/max_h

    plt.subplot(1,2,2)
    plt.scatter(scores, weights, color='orange')
    plt.title('weights')

    return weights


def get_ckpt_name(bp, dataset, data_type, frozen, resc, gpus, bs, loss_type, weighted, do, a,b,c):
    name = '%s_%s_fzn%d_resc%d_LT%s_a%db%dc%d_w%s_gpus%d_bs%d_do%.2f_ep{epoch:02d}_valloss{val_loss:.4f}_valrc{rc_val:.4f}.hdf5' % (dataset, data_type, frozen, resc, loss_type,  a,b,c, weighted, gpus, bs, do)
    return os.path.join(bp, name)

def get_ckpt_name_cap(bp, dataset, data_type, frozen, resc, gpus, bs, loss_type, weighted, do, a,b,c, use_rc=True, predict_sentences=True):

    val_rc_string= '_valrc{rc_val:.4f}' if use_rc else ''
    val_acc_string ='val_td_output_captions_accuracy' if predict_sentences else 'val_output_captions_accuracy'
    name = '%s_%s_fzn%d_resc%d_LT%s_a%db%dc%d_w%s_gpus%d_bs%d_do%.2f_ep{epoch:02d}_valloss{val_loss:.4f}%s_capacc{%s:.3f}.hdf5' % (dataset, data_type, frozen, resc, loss_type,  a,b,c, weighted, gpus, bs, do, val_rc_string, val_acc_string)
    return os.path.join(bp, name)


def freeze_unfreeze_net(i3d, gpus=1, freeze=True):

    if gpus>1:
#         print('GPUPS>1. Freezing all layers in inner i3d')
        i3d.layers[-2].layers[2].trainable = not freeze
        for l in i3d.layers[-2].layers[2].layers:
            l.trainable = not freeze

    else:
#         print('Freezing all but last 4 layers')
        i3d.layers[2].trainable = not freeze
        for l in i3d.layers[2].layers:
            l.trainable = not freeze


    return i3d


def define_callbacks(ckpt_filepath, gen_val, name_to_mem_alpha=None, val_names=None, true_mem=None, lr=0.0001, drop=0.1,
                    epochs_drop=10, ckpt_period=1, use_cb_rankcorr=True, use_cb_plot=False,
                    preds_savefile=None, idx_to_look_at=[2], get_gt_from_gen=False):

    if true_mem is None:
        true_mem = [name_to_mem_alpha[n][0] for n in val_names]
    cbs = []
    if use_cb_rankcorr:
        cb_rankcorr = RankCorr(gen_val, true_mem, preds_savefile=preds_savefile, idx_to_look_at=idx_to_look_at, get_gt_from_gen=get_gt_from_gen)
        cbs.append(cb_rankcorr)

    cb_chk = MultiGPUCheckpoint(ckpt_filepath, monitor='val_loss',
                             verbose=1, save_best_only=False,
                             save_weights_only=True, period=ckpt_period)
    cbs.append(cb_chk)

    cb_sched = LearningRateScheduler(step_decay(init_lr = lr, drop=drop, epochs_drop=epochs_drop))
    cbs.append(cb_sched)

    if use_cb_plot:
        cb_plot = InteractivePlot()
        cbs.append(cb_plot)

    return cbs




class InteractivePlot(keras.callbacks.Callback):
    def __init__(self, state=None, statename=None):
        self.statename=statename
        if statename:
            self.state=np.load(statename)
        else:
            self.state=state

        self.batchnr=0

    def on_train_begin(self, logs={}):
        self.losses = [] if self.state is None else self.state['loss']
        self.val_rcs = [] if self.state is None else self.state['val_rc']
        self.val_mses = [] if self.state is None else self.state['val_mse']
        self.logs = [] if self.state is None else self.state['logs']

        if self.state is None:
            self.state={'loss':[], 'val_rc':[], 'val_mse':[], 'logs':[]}

        # self.batchnr = 0
        # self.icount = 0

    def on_train_end(self, logs={}):
        pass


    def on_epoch_end(self, epoch, logs={}):
        self.batchnr = 0
        val_mse = logs.get('val_mse')
        val_rc = logs.get('rc_val')
        print('VAL MSE:', val_mse)
        print('VAL_RC:', val_rc)

        self.losses.append(np.mean([logs.get('loss')]+self.losses[-20:]))
        self.val_rcs.append(val_rc)
        self.val_mses.append(val_mse)

#         self.icount+=1

        self.plot_graph()

        #plt.savefig(self.logfile.replace('.txt', '.png'), bbox_inches='tight', format='png')
        plt.show()

    def on_batch_end(self, batch, logs=None):
        self.batchnr+=1

        self.losses.append(np.mean([logs.get('loss')]+self.losses[-20:]))

        if self.batchnr % 10 == 0:
            self.update_state(logs)
            self.plot_graph()

    def update_state(self, logs):
        self.state['losses'] = self.losses
        self.state['val_mses'] = self.val_mses
        self.state['val_rcs'] = self.val_rcs
        self.state['logs'] = logs

        if self.statename:
            dire = '../states'
            if not os.path.exists(dire):
                os.makedirs(dire)
            np.save(os.path.join(dire,self.statename),self.state)

    def plot_graph(self):
        arrs_to_plot = [self.losses, self.val_mses, self.val_rcs]
        desc = ['loss', 'val mse', 'val rc']
        colors = ['cornflowerblue', 'limegreen', 'orange']
        markers = [None, '.', '.']
        clear_output(wait=True)
        plt.figure(figsize=(14,10))
        for i in range(len(arrs_to_plot)):
            plt.subplot(1, len(arrs_to_plot), i+1)
            plt.plot(range(len(arrs_to_plot[i])), arrs_to_plot[i], label=desc[i], color=colors[i], marker=markers[i])
            plt.legend()
        print('Last loss: %.5f' % self.losses[-1])
        print('Last RC: %.5f' % self.val_rcs[-1] if self.val_rcs else 0)
        print('Last MSE: %.5f' % self.val_mses[-1] if self.val_mse else 0)

        plt.show()


def get_train_test_sets_from_pkl(split_pkl, names_with_slash=False, return_test=False, json_for_test_names=None):
    with open(split_pkl, 'rb') as f:
        splits = pickle.load(f)
        
    train = splits[0]
    val = splits[1]
    
    if return_test:
        if json_for_test_names is not None: # Use this json to get test names
            test = json_for_test_names['test']
        else:  # Search for test videos in data_path
            test = []
            for v in os.listdir(video_path):
                if v not in train and v not in val:
                    test.append(v)
        return train, val, test
    
    return train, val
            
    

def get_train_test_sets_from_json(train_val_test_split_json, train_json=None, val_json=None, test_json=None, names_with_slash=False, return_test=False):

    with open(train_val_test_split_json, 'r') as f:
        tt_s = json.load(f)

    train_names=[]
    val_names=[]
    test_names=[]

    if names_with_slash:
        for n in tt_s['train']:
            cl = n.split('_')[0]
            cl = cl.replace('-','+')
            newn = cl+'/'+'_'.join(n.split('_')[1:])
            train_names.append(newn)

        for n in tt_s['val']:
            cl = n.split('_')[0]
            cl = cl.replace('-','+')
            newn = cl+'/'+'_'.join(n.split('_')[1:])
            val_names.append(newn)

        for n in tt_s['test']:
            cl = n.split('_')[0]
            cl = cl.replace('-','+')
            newn = cl+'/'+'_'.join(n.split('_')[1:])
            test_names.append(newn)
    else:
        train_names = sorted([ n[:-4]+'.npy' for n in tt_s['train'] ])
        val_names = sorted([ n[:-4]+'.npy' for n in tt_s['val'] ])
        test_names = sorted([ n[:-4]+'.npy' for n in tt_s['test'] ])

    if return_test:
        return train_names, val_names, test_names
    else:
        return train_names, val_names


def get_name_to_mem_dict(mem_scores_file, alpha_file, labels_path, names_with_slash=False):

    with open(os.path.join(labels_path, mem_scores_file)) as f:
        name_to_mem = json.load(f)

    with open(os.path.join(labels_path, alpha_file)) as f:
        name_to_alpha = json.load(f)

    # Getting dict in same format as vid names
    name_to_mem_alpha={}

    if names_with_slash:
        for k,v in name_to_mem.items():
            name_to_mem_alpha[k] = np.array([v, name_to_alpha[k]])

    else:
        for k,v in name_to_mem.items():
            newk = k.replace('/','_').replace('+','-')[:-4]+'.npy'
            name_to_mem_alpha[newk] = np.array([v, name_to_alpha[k]])

    return name_to_mem_alpha, name_to_mem, name_to_alpha

def ntn(s):
    return s.replace('/','_').replace('+','-')[:-4]+'.npy'

def get_memento_data(train_data, val_data, test_data, return_captions=True, tokenize=True):
    
    json_train = json.load(open(train_data))
    json_val = json.load(open(val_data))
    json_test = json.load(open(test_data))
    
    train_captions = {}
    test_captions = {}
    val_captions = {}
    
    train_names = []
    test_names = []
    val_names = []

    name_to_mem_alpha={}
    if tokenize:
        vocab = json.load(open("../../memento_data/vocab_3.json")) 
        update_d = lambda a,b: update_cap_dict_tokenize(a,b,vocab)
    else:
        update_d = update_cap_dict
        
    for e in json_train:
        train_captions = update_d(train_captions, e)
        train_names.append(ntn(e['filename']))
        name_to_mem_alpha[ntn(e['filename'])] = np.array([e['mem_score'], e['alpha']])
    for e in json_test:
        test_captions = update_d(test_captions, e)
        test_names.append(ntn(e['filename']))
        name_to_mem_alpha[ntn(e['filename'])] = np.array([e['mem_score'], e['alpha']])
    for e in json_val:
        val_captions = update_d(val_captions, e)
        val_names.append(ntn(e['filename']))
        name_to_mem_alpha[ntn(e['filename'])] = np.array([e['mem_score'], e['alpha']])
    
    return train_names, val_names, test_names, name_to_mem_alpha, train_captions, val_captions, test_captions

def update_cap_dict(dc, e):
    dc[e['filename']] = {'indexed_captions':e['captions']}
    return dc

def update_cap_dict_tokenize(dc, e, vocab):
    
    START_TOK = 2986
    END_TOK = 2987
    
    tok_caps = []
    for cap in e['captions']:
#         print(cap.split())
        tok_cap = []
        for word in cap.split():
            w = word.lower().strip().replace('.','')
            if w in vocab:
#                 print(w, vocab.index(w)+1)
                tok_cap.append(vocab.index(w)+1)
        tok_cap = [START_TOK] + tok_cap + [END_TOK] + [0]*(cfg.MAX_CAP_LEN - len(tok_cap))
        
#         print("TOKENIZED CAPTION:",tok_cap)
        tok_caps.append(tok_cap)
    dc[e['filename']] = {'indexed_captions': tok_caps}
    return dc
        
    