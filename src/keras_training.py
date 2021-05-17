#!/usr/bin/env python

'''File with necessary functions for training the keras i3d model.'''

__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from generator import VideoSeqGenerator, build_label2str, read_videofile_txt
from keras.models import Model
import keras
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD, Nadam
import argparse
import i3d_config as cfg
from vid_utils import plot_frames
from keras_models import build_model_multigpu, resnet_lstm_model
import tensorflow as tf
import warnings
from keras.metrics import top_k_categorical_accuracy

#######################################
### TRAINING FUNCTIONS ###
#######################################

def train_keras(model, txt_loc, val_txt_loc=None,
            epochs = cfg._EPOCHS,
            steps=cfg._STEPS,
            val_steps=cfg._VAL_STEPS,
            save_ckpt=cfg._SAVE_CKPT+'/weights',
            batch_size=cfg._BATCH_SIZE_TRAIN, verbose=cfg._VERBOSE,
            max_queue_size=cfg._MAX_QUEUE_TRAIN, workers=cfg._WORKERS_TRAIN,
            use_mp=cfg._USE_MP_TRAIN, print_results=False,
            classes_to_process=None, func_gen=False, has_label_col=False, **gen_params):
    ''' Training function for keras models. Assumes a particular folder structure:
    Data must be in args.data_path, and there must be a parallelfolder args.data_path_aux
    where the csv file with the labels is loacted, as well as the train and test split txt files
    indicating the path of the videos to be used for each split.'''

    if classes_to_process and classes_to_process != 'all':
        if isinstance(classes_to_process, str):
            classes_to_process = read_txt_oneliner(classes_to_process)
        # Restrict files to classes we're interested in
        train_files = read_videofile_txt(txt_loc, ret_label_array=has_label_col)
        if has_label_col:
            train_files = [f for f in zip(train_files[0],train_files[1]) if f[1] in classes_to_process]
        else:
            train_files = [f for f in train_files if f.split('/')[0] in classes_to_process]

        test_files = read_videofile_txt(val_txt_loc, ret_label_array=has_label_col)
        if has_label_col:
            test_files = [f for f in zip(test_files[0],test_files[1]) if f[1] in classes_to_process]
        else:
            test_files = [f for f in test_files if f.split('/')[0] in classes_to_process]

        # Define Generators
        if func_gen:
            vid_gen = video_generator(files=train_files, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
            val_gen = video_generator(files=test_files, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
        else:
            vid_gen = VideoSeqGenerator(files=train_files, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
            val_gen = VideoSeqGenerator(files=test_files, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
    else:
        # Define Generators
        if func_gen:
            vid_gen = video_generator(txt_loc, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
            val_gen = video_generator(val_txt_loc, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
        else:
            vid_gen = VideoSeqGenerator(txt_loc, batch_size=batch_size, has_label_col=has_label_col, **gen_params)
            val_gen = VideoSeqGenerator(val_txt_loc, batch_size=batch_size, has_label_col=has_label_col, **gen_params)

    # Define callbacks
    cb_chk = MultiGPUCheckpoint(save_ckpt, monitor='val_loss',
                             verbose=1, save_best_only=False,
                             save_weights_only=True, mode='auto', period=2)
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto',  cooldown=0, min_lr=1e-8)

    cbs = [cb_chk, cb_lr]
    if print_results:
        l2s_dict = build_label2str(gen_params['label_csv'])
        cb_show = ShowFrames(gen=val_gen, freq=10, vids_to_show=2, label2str_dict=l2s_dict)
        cbs.append(cb_show)

    print('Ready to train')
    # Training
    model.fit_generator(vid_gen, steps_per_epoch=steps, epochs=epochs,
                            verbose=verbose, callbacks=cbs,
                            validation_data=val_gen, validation_steps=val_steps,
                            max_queue_size=max_queue_size,workers=workers, use_multiprocessing=use_mp)


def instantiate_model(model_type,
                        optimizer=cfg._OPTIMIZER,
                        lr=cfg._LR,
                        ckpt = cfg._INIT_CKPT,
                        type=cfg._TYPE, num_classes =cfg._NUM_CLASSES,
                        verbose=cfg._VERBOSE, gpus=cfg._GPUS,
                        downsample_factor=None,
                        run_locally=False):

    model = build_model_multigpu(ckpt=ckpt, type=type, num_classes=num_classes,
                            verbose=verbose, gpus=gpus, model_type=model_type,
                            downsample_factor=downsample_factor, run_locally=run_locally)


    # Define optimizer
    if optimizer == 'Adam':
        opt = Adam(lr=lr) #momentum=0.9)\
    elif optimizer == 'SGD':
        opt = SGD(lr=lr, momentum=0.9)
    else:
        raise ValueError('Unknown optimizer')

    print('Compiling model')

    # Compile model
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', top_2_accuracy, top_5_accuracy])
    if verbose:
        model.summary()

    return model

##########################
## KERAS CUSTOM METRICS ##
##########################
def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

############################
##### CUSTOM CALLBACKS #####
############################

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_batch_end(self, batch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();


class MultiGPUCheckpoint(ModelCheckpoint):

    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model

class ShowFrames(keras.callbacks.Callback):
    '''Custom callback to show results while the network is training. Shows `vids_to_show` random
    results from the set of files indicated in `txt_file`.'''

    def __init__(self,txt_file='',freq=100, vids_to_show=1, label2str_dict=None, csv_path='', gen=None):
        self.freq=freq
        self.vids_to_show=vids_to_show
        self.l2s_dict=label2str_dict
        if gen is not None:
            self.gen=gen
        else:
            self.gen = video_generator(txt_file, batch_size=vids_to_show, is_train=False, label_csv=csv_path)

    def on_train_begin(self, logs={}):
        self.counter=0

    def on_batch_end(self, batch, logs={}):
        self.counter+=1
        if not self.counter % self.freq:
            if isinstance(self.gen, keras.utils.Sequence):
                batch = self.gen.__getitem__(self.counter%len(self.gen))
            else:
                batch = next(self.gen)
            num_vids_in_batch = len(batch[1])
            if num_vids_in_batch < self.vids_to_show:
                print('Asked to show %d results, but only %d available in batch. Showing all available.' % (self.vids_to_show, num_vids_in_batch))
                self.vids_to_show = num_vids_in_batch
            y_true = np.argmax(batch[1],axis=1)
            y_pred = np.argmax(self.model.predict(batch[0]), axis=1)
            for i in range(self.vids_to_show):
                if self.l2s_dict:
                    str_pred = self.l2s_dict[y_pred[i]]
                    str_true = self.l2s_dict[y_true[i]]
                else:
                    str_pred=str_true='-'
                caption = 'Predicted: '+str_pred+' / '+str(y_pred[i])+' - True: '+str_true+' / '+str(y_true[i])
                plot_frames(batch[0][i], title=caption, idxs =[0,16,32,48,-1])

class SaveHistory(keras.callbacks.Callback):
    def __init__(self,**kargs):
        super(CustomModelCheckPoint,self).__init__(**kargs)
        self.it_acc = {}
        self.it_acc_top2 = {}
        self.it_acc_top5 = {}
        self.it_loss = {}
        self.epoch_acc = {}
        self.epoch_acc_top2 = {}
        self.epoch_acc_top5 = {}
        self.epoch_valacc = {}
        self.epoch_valacc_top2 = {}
        self.epoch_valacc_top5 = {}
        self.epoch_loss = {}
        self.epoch_valloss = {}
        def on_epoch_begin(self, epoch, logs={}):
            # Things done on beginning of epoch.
            return

        def on_epoch_end(self, epoch, logs={}):
            # things done on end of the epoch
            self.epoch_accuracy[epoch] = logs.get("acc")
            self.epoch_loss[epoch] = logs.get("loss")
            self.model.save_weights("name-of-model-%d.h5" %epoch) # save the model


def process_dataset_arg(dataset, args, is_train=True):
    '''Uses the value of the `dataset` argument to define the following peripheric arguments:
        - data_path
        - num_classes
        - load_func
        - prep_func
        - txt_path
        - val_txt_path
        - labels_path

        This function is also used in the eval py file. If adding a new dataset,
        just add another `if` clause following the pattern of the existing ones.
        Any arguments explicitly given by the user on the cmd line will not be modified here.
        If using this with the eval script, val_txt_path will not be defined.
    '''

    if is_train:
        mode = 'train'
    else:
        mode = 'test'

    # If dataset is one of the known datasets, set the appropriate arguments
    if args.dataset == 'HMDB51':
        if args.data_path is None:
            args.data_path = cfg._PATH_HMDB
        if args.num_classes is None:
            args.num_classes = 51
        if args.load_func is None:
            args.load_func = 'npy_'+args.type

    elif args.dataset == 'HMDB51f':
        if args.data_path is None:
            args.data_path = '../../../datasets/HMDB51/frames' #cfg._PATH_HMDB
        if args.num_classes is None:
            args.num_classes = 51
        if args.prep_func is None:
            args.prep_func = 'fast_rgb'
        if args.load_func is None:
            args.load_func = 'frames_raw' #'npy_'+args.type

    elif args.dataset == 'HMDB51v':
        if args.data_path is None:
            args.data_path = '../../../datasets/HMDB51/videos' #cfg._PATH_HMDB
        if args.num_classes is None:
            args.num_classes = 51
        if args.prep_func is None:
            args.prep_func = 'i3d_rgb'
        if args.load_func is None:
            args.load_func = 'vids_opencv' #'npy_'+args.type

    elif args.dataset == 'UCF101':
        if args.data_path is None:
            args.data_path = cfg._PATH_UCF
        if args.num_classes is None:
            args.num_classes = 101
        if args.prep_func is None:
            args.prep_func = 'i3d_'+args.type
        if args.load_func is None:
            args.load_func = 'vids_opencv'

    elif args.dataset == 'kinetics':
        if args.data_path is None:
            args.data_path = '../../../datasets/Kinetics'
        if args.num_classes is None:
            args.num_classes = 400
        if args.prep_func is None:
            args.prep_func = 'i3d_'+args.type
        if args.load_func is None:
            args.load_func = 'vids_opencv'

    elif args.dataset == 'mixed_crops5':
        if args.data_path is None:
            args.data_path = '../../../datasets/IVA_Videos/crops_mixed'
        if args.num_classes is None:
            args.num_classes = 5
        if args.prep_func is None:
            args.prep_func = 'fast_'+args.type
        if args.load_func is None:
            args.load_func = 'frames_raw'


    elif args.dataset == 'moments_mini':
        if args.data_path is None:
            args.data_path = '../../moments/Moments_in_Time_Mini'
        if args.num_classes is None:
            args.num_classes = 200
        if args.prep_func is None:
            args.prep_func = 'fast_'+args.type
        if args.load_func is None:
            args.load_func = 'vids_opencv'
        if args.txt_path is None:
            args.txt_path = '../../moments/split/trainingSetMini_wp.csv'
        if args.labels_path is None:
            args.labels_path = '../../moments/split/moments_mini_categories.txt'
        if mode=='train' and args.val_txt_path is None:
            args.val_txt_path = '../../moments/split/validationSetMini_wp.csv'

    elif args.dataset == 'moments':
        if args.data_path is None:
            args.data_path = '../../moments/moments_nov17_frames'
        if args.num_classes is None:
            args.num_classes = 339
        if args.prep_func is None:
            args.prep_func = 'fast_'+args.type
        if args.load_func is None:
            args.load_func = 'frames_raw'
        if args.txt_path is None:
            args.txt_path = '../../moments/split/rgb_trainingSet_nov17.csv'
        if args.labels_path is None:
            args.labels_path = '../../moments/split/categoryList_nov17.csv'
        if mode=='train' and args.val_txt_path is None:
            args.val_txt_path = '../../moments/split/rgb_testingSet_nov17.csv'


    elif args.dataset is not None:
        warnings.warn('Unrecognized dataset: '+str(args.dataset))

    if args.txt_path is None:
        args.txt_path = os.path.join(args.data_path+'_aux',mode+'_split0'+str(args.split)+'.txt')
    if args.labels_path is None:
        args.labels_path = os.path.join(args.data_path+'_aux','labels.csv')
    if mode=='train' and args.val_txt_path is None:
        args.val_txt_path = os.path.join(args.data_path+'_aux','test_split0'+str(args.split)+'.txt')

    return args



################
##### MAIN #####
################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ## MAIN ARGUMENTS
    parser.add_argument('--init_ckpt',
        help='Path to checkpoint. If this is not provided, \
        the net starts with the precomputed kinetics weights.',
        type=str,
        default=cfg._INIT_CKPT)

    parser.add_argument('--dataset',
        help='Name of dataset. Alternatively, the user can pass the num_classes, \
        path to data, prep and load funcs separately instead of setting this parameter.',
        type=str,
        default=cfg._DATASET)

    parser.add_argument('--gpus',
        help='Number of GPUs to use. If 1 or None, no multi_gpu model is instanciated.',
        type=int,
        default=cfg._GPUS)

    parser.add_argument('--verbose',
        help='Verbosity value.',type=int,
        default=cfg._VERBOSE)

    ## TRAIN SPECIFIC ARGUMENTS
    parser.add_argument('--epochs',
        help='Epochs to train.',
        type=int,
        default=cfg._EPOCHS)
    parser.add_argument('--steps',
        help='Number of steps per epoch',
        type=int,
        default=cfg._STEPS)

    parser.add_argument('--steps_manual',
        help='Number of steps to yield per epoch, but controlled via manual code (when you dont trust fit_generator...)',
        type=int,
        default=cfg._STEPS)

    parser.add_argument('--val_steps',
        help='Number of validation steps per epoch.',
        type=int,
        default=cfg._VAL_STEPS)
    parser.add_argument('--batch_size',
        help='Batch size, fed to the evaluate_generator function',
        type=int,
        default=cfg._BATCH_SIZE_TRAIN)
    parser.add_argument('--lr',
        help='Learning Rate.',
        type=float,
        default=cfg._LR)
    parser.add_argument('--save_ckpt',
        help='Where to save the checkpoint as the model trains.',
        type=str,
        default=cfg._SAVE_CKPT)
    parser.add_argument('--max_queue',
        help='Max queue size. Fed to evaluate_generator function.',
        type=int,
        default=cfg._MAX_QUEUE_TRAIN)
    parser.add_argument('--workers',
        help='Number of workers for evaluate_generator',
        type=int,
        default=cfg._WORKERS_TRAIN)
    parser.add_argument('--use_mp',
        help='Use multiprocessing or not. Boolean.',
        type=int,
        default=cfg._USE_MP_TRAIN)
    parser.add_argument('--augment',
        help='Augmentation types. Send augmentation srtings separated by comas, e.g. "flip,cutout,edge" (no spaces)',
        type=str,
        default=cfg._AUGMENT)

    ## NEW DATASET ARGS
    ## These arguments override the default arguments set by --dataset
    parser.add_argument('--data_path',
        help='Path to the training folder. In this folder, videos should be in subfolders per classname: classname/videoname',
        type=str, default=cfg._DATA_PATH)

    parser.add_argument('--num_classes',
        help='Number of classes for the given dataset',
        type=int, default=cfg._NUM_CLASSES)

    parser.add_argument('--prep_func',
        help='Preprocessing function',type=str,
        default=cfg._PREP_FUNC)

    parser.add_argument('--load_func',
        help='Load Function',type=str,
        default=cfg._LOAD_FUNC)


    ## OTHER OPTIONAL ARGS
    parser.add_argument('--txt_path',
        help='Path to txt containing paths of videos to train on, in format classname/videoname',
        type=str, default=cfg._TXT_PATH)

    parser.add_argument('--val_txt_path',
        help='Path to txt containing paths of videos to validate on, in format classname/videoname',
        type=str, default=cfg._VAL_TXT_PATH)

    parser.add_argument('--split',
        help='Split number. Optional. If txt_path is defined, this arg is overrided.',
        type=int,
        default=cfg._SPLIT)

    parser.add_argument('--labels_path',
    help='Path to csv containing label names and their corresponding values',
    type=str, default=cfg._LABELS_PATH)

    parser.add_argument('--type',
    help='Type of net to load. one of [flow, rgb]',
    type=str, default=cfg._TYPE)

    parser.add_argument('--classes_to_process',
    help='Txt with list of classes to process, separated by newline. If nothing is provided, \
    all classes found in dataset folder will be evaluated.',
    type=str, default=cfg._CLASSES)

    parser.add_argument('--plot',
    help='Wether to plot intermediate results or not.',
    type=int, default=0)

    parser.add_argument('--model_type',
    help='Model type to train. Must match the given init_ckpt, if any',
    type=str, default='i3d')

    parser.add_argument('--optimizer',
    help='Optimizer to use. Adam and SGD supported.',
    type=str, default='Adam')

    parser.add_argument('--downsample',
    help='Int factor by which to downsample temporally the input videos before running them through the model.',
    type=int, default=None)

    parser.add_argument('--remove_excess',
    help='Automatically removes some of given files to make sure that data is multiple of batch size.',
    type=int, default=1)

    parser.add_argument('--run_locally',
    help='Indicates that the script will be run on local machine or notebook instead of SaturnV. \
    This makes the program search for certain internal chekpoints in the correct locations.',
    type=int, default=0)

    parser.add_argument('--has_label_col',
    help='Indicates that the txt files have a column with the labels of each file. If this is true, \
    we dont have to get the labels from the folder structure in the name.',
    type=int, default=0)


    args = parser.parse_args()

    args = process_dataset_arg(args.dataset, args, is_train=True)

    augment = args.augment.split(',')
    if args.classes_to_process:
        str_classes='sub'
    else:
        str_classes='all'
    if not os.path.isdir(args.save_ckpt):
        os.makedirs(args.save_ckpt)
    if not os.path.isdir(os.path.join(args.save_ckpt,args.model_type)):
        os.makedirs(os.path.join(args.save_ckpt,args.model_type))
    if args.dataset == None:
        args.dataset = args.data_path.split('/')[-1]


    save_ckpt = os.path.join(args.save_ckpt,args.model_type,
    'weights_'+args.dataset+
    '_'+str_classes+
    '_classes'+str(args.num_classes)+
    '_ds'+str(args.downsample)+
    '_split'+str(args.split)+
    '_'+args.type+
    '_gpus'+str(args.gpus)+
    '_ep{epoch:02d}_valloss{val_loss:.2f}_valacc{val_acc:.2f}.hdf5')

    print('Args:')
    for arg in vars(args):
        print (arg, getattr(args, arg))

    # Define generator parameters
    gen_params = dict(dataset_path=args.data_path,
                      label_csv=args.labels_path,
                      augment=augment,
                      load_func=args.load_func,
                      preprocess_func=args.prep_func,
                      remove_excess_files=args.remove_excess,
                      shuffle=True,
                      is_train=True,
                      it_per_epoch=args.steps_manual)

    # Instantiate model
    model = instantiate_model(args.model_type, optimizer=args.optimizer, lr=args.lr,
                                        ckpt = args.init_ckpt, type=args.type,
                                        num_classes=args.num_classes,
                                        downsample_factor=args.downsample,
                                        verbose=args.verbose,
                                        run_locally=args.run_locally,
                                        gpus=args.gpus)

    # Train model
    train_keras(model, args.txt_path, args.val_txt_path,save_ckpt = save_ckpt,
            epochs=args.epochs, steps=args.steps, val_steps=args.val_steps,
            batch_size=args.batch_size, max_queue_size=args.max_queue,
            workers=args.workers, use_mp=args.use_mp, verbose=args.verbose,
            classes_to_process=args.classes_to_process,
            print_results=args.plot, has_label_col=args.has_label_col, **gen_params )
