
'''
keras generator for videos. contains functions for loading and processing videos for
multiple architectures, currently mainly focused on i3d from DeepMind.
'''

__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"

import numpy as np
import random
import os
# from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
# from skimage import filters
# from skimage.filters import roberts, sobel, scharr, prewitt
import pandas as pd
from timeit import default_timer as timer
from vid_utils import load_video_opencv, load_video_opencv_fixed_frames, load_video_imageio, load_video_skvideo, plot_frames
from vid_utils import time_crop, video_spatial_crop, video_resize, video_rescale, video_truncate_and_rescale, video_resample
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import keras
import sys
sys.path.append('../')

# from imgaug import augmenters as iaa
# from imgaug import parameters as iap
import i3d_config as cfg
from keras.preprocessing import image
from IPython.display import clear_output


def build_str2label(csv_path):
    lines = pd.read_csv(csv_path, header=None, sep=None).values
    label_dictionary = {}
    for line in lines:
        label_dictionary[line[0]] = line[1]

    return label_dictionary


def build_label2str(csv_path):
    lines = pd.read_csv(csv_path, header=None, sep=None).values
    label_dictionary = {}
    for line in lines:
        label_dictionary[line[1]] = line[0]

    return label_dictionary

def preprocess_spoof(vid_list, is_train=True):
    pass

def preprocess_frames_and_bboxes(batch, is_train=True):
    '''Preprocess function for ROI pooling. Preprocesses both input frames and bboxes accordingly.'''

    N= len(batch)
    channels = batch[0].shape[3]

    vid_array = []

    for i in range(N):
        # start = timer()
        ## Resize, smallest dim 256 pixels, bilinear interpolation
        resi_frames, scale_x, scale_y = video_resize_fixed(vid_list[i])

        bbox[i, :, 0] = int(bbox[i, :, 0]*scale_x)
        bbox[i, :, 1] = int(bbox[i, :, 1]*scale_y)
        bbox[i, :, 2] = int(bbox[i, :, 2]*scale_x)
        bbox[i, :, 3] = int(bbox[i, :, 3]*scale_y)

        # t1=timer()
        ## During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        # crop_frames = video_spatial_crop(resi_frames, size=cfg._IMAGE_SIZE, randomize=is_train)
        # t2=timer()
        ## Rescale between -1 and 1
        resc_frames = video_rescale(vid_list[i], maxi=1, mini=0)
        # t3=timer()
        ## Append frames to array
        vid_array.append( resc_frames )
        # t4=timer()
        # print('resizing: %.3f, sp_crop: %.3f, rescaling: %.3f, assigning to array: %.3f' % (t1-start, t2-t1,t3-t2,t4-t3))
    return vid_array

def preprocess_i3d_rgb(vid_list, is_train=True):
    ''' Preprocess a list of raw videos. The videos must be in numpy array format.'''

    N= len(vid_list)
    # print('len(vid_list)',len(vid_list))
    # print('vid_listshape',vid_list[0].shape)
    channels = vid_list[0].shape[3]
    vid_array = np.zeros((N, cfg._NUM_FRAMES, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, channels), dtype=np.float32)

    for i,vid in enumerate(vid_list):
        # Sample at 25 fps.
        # resa_frames = video_resample(vid, fps=25)

        # During training, randomly select a sequence of N frames. During test, take the middle N frames.
        resa_frames = time_crop(vid, N=cfg._NUM_FRAMES, is_train= is_train)

        # Resize, smallest dim 256 pixels, bilinear interpolation
        resi_frames = video_resize(resa_frames)

        # During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        crop_frames = video_spatial_crop(resi_frames, size=cfg._IMAGE_SIZE, randomize=is_train)

        # Rescale between -1 and 1
        resc_frames = video_rescale(crop_frames)

        # making sure that we're using float32
        vid_array[i] = resc_frames.astype(np.float32)

    return vid_array

def preprocess_rgb_fast_with_caption(X, is_train=True):
    return [preprocess_rgb_fast(X[0], is_train=is_train), X[1]]

def preprocess_flow_fast_with_caption(X, is_train=True):
    return [preprocess_flow_fast(X[0], is_train=is_train), X[1]]

def preprocess_rgb_fast(vid_list, is_train=True):
    '''Preprocess input frames faster than preprocess_i3d. for this to work,
    The vid_list has to contain videos that are either numpy arrays or lists of frames,
    frames being numpy arrays of the same width, height and channel. The frames must contain values from 0 to 255.
    Also, most importantly, the videos must already be time cropped,
    which means they must have exactly _NUM_FRAMES frames.

    If planning on feeding videos of more or less than cfg._NUM_FRAMES (for testing on untrimmed or unlooped videos, for
    example), use the preprocess_rgb_fast_var_input function '''

    N= len(vid_list)
    channels = vid_list[0].shape[3]

    vid_array = np.zeros((N, cfg._NUM_FRAMES, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, channels), dtype=np.float32)


    full_time=0
    for i in range(N):
#         start = timer()
        ## Resize, smallest dim 256 pixels, bilinear interpolation
#         resi_frames = video_resize(vid_list[i])

#         t1=timer()
        ## During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        crop_frames = video_spatial_crop(vid_list[i], size=cfg._IMAGE_SIZE, randomize=is_train)

        ##DEBUG
        # crop_frames/=255.

#         t2=timer()
        ## Rescale between -1 and 1 - THIS HAPPENS INSIDE THE NETWORK NOW
        # resc_frames = video_rescale(crop_frames)
#         t3=timer()

        ## Append frames to array
        vid_array[i] = crop_frames.astype(np.float32)
#         t4=timer()
#         print('resizing: %.4f, sp_crop: %.4f, rescaling: %.4f, assigning to array: %.4f' % (t1-start, t2-t1,t3-t2,t4-t3))
#         full_time+=t4-start

#     print('Full time on this batch of %d:' % N, full_time)
    return vid_array

def preprocess_rgb_fast_var_input(vid_list, is_train=True):
    '''Preprocess input frames faster than preprocess_i3d. for this to work,
    The vid_list has to contain videos that are either numpy arrays or lists of frames,
    frames being numpy arrays of the same width, height and channel. The frames must contain values from 0 to 255.

    This function doesn't assume a fixed number of input frames. Used for feeding
    batches where the videos have different number of frames (e.g. testing situation)
    '''

    N= len(vid_list)
    channels = vid_list[0].shape[3]

    vid_array = []

    for i in range(N):
        # Resize, smallest dim 256 pixels, bilinear interpolation
        resi_frames = video_resize(vid_list[i])

        # During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        crop_frames = video_spatial_crop(resi_frames, size=cfg._IMAGE_SIZE, randomize=is_train)

        # Rescale between -1 and 1
        resc_frames = video_rescale(crop_frames)

        vid_array.append(resc_frames.astype(np.float32))

    return vid_array


def preprocess_i3d_flow(vid_list, is_train=False):

    # print('len(vid_list)',len(vid_list))
    # print('vid_listshape',vid_list[0].shape)
    channels = vid_list[0].shape[3]
    vid_array = np.empty((len(vid_list), cfg._NUM_FRAMES, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, channels), dtype=np.float32)

    for i,vid in enumerate(vid_list):
        # Sample at 25 fps.
        # resa_frames = video_resample(vid, fps=25)

        # During training, randomly select a sequence of N frames. During test, take the middle N frames.
        resa_frames = time_crop(vid, N=cfg._NUM_FRAMES, is_train= is_train)

        # Resize, smallest dim 256 pixels, bilinear interpolation
        resi_frames = video_resize(resa_frames)

        # During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        crop_frames = video_spatial_crop(resi_frames, size=cfg._IMAGE_SIZE, randomize=is_train)

        # Rescale between -1 and 1
        resc_frames = video_truncate_and_rescale(crop_frames)

        # making sure that we're using float32
        vid_array[i] = resc_frames.astype(np.float32)

    return vid_array

def preprocess_flow_fast(vid_list, is_train=False):

    # print('len(vid_list)',len(vid_list))
    # print('vid_listshape',vid_list[0].shape)
    channels = vid_list[0].shape[3]
    vid_array = np.empty((len(vid_list), cfg._NUM_FRAMES, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, channels), dtype=np.float32)

    for i,vid in enumerate(vid_list):
        # Sample at 25 fps.
        # resa_frames = video_resample(vid, fps=25)

        # During training, randomly select a sequence of N frames. During test, take the middle N frames.
#         resa_frames = time_crop(vid, N=cfg._NUM_FRAMES, is_train= is_train)

        # Resize, smallest dim 256 pixels, bilinear interpolation - done offline
#         resi_frames = video_resize(vid)

        # During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
        crop_frames = video_spatial_crop(vid, size=cfg._IMAGE_SIZE, randomize=is_train)

        # Rescale between -1 and 1 - done inside the net
#         resc_frames = video_truncate_and_rescale(crop_frames)

        # making sure that we're using float32
        vid_array[i] = crop_frames.astype(np.float32)

    return vid_array

def load_hmdb_npy(filenames, type='rgb', hmdb_npy_path = '../data/HMDB51', is_train=False, add_suffix=False):
    '''Npy file loader for the HMDB dataset. Returns rgb or flow npy files.'''

    # Change this if loading anything other than the preprocessed npy files found in ../data/HMDB51

    if type == 'rgb':
        channels = 3
        suffix = '_rgb'
    elif type =='flow':
        channels = 2
        suffix = '_flow'
    else:
        raise ValueError('Unrecognized type of image to load')

    if not add_suffix: suffix=''

    N = len(filenames)
    loaded_npy = np.zeros((N, cfg._NUM_FRAMES, 256, 256, channels))

    i=0
    for file in filenames:
        npy_name = file.split('.')[0]+suffix+'.npy'

#         print(os.path.join(hmdb_npy_path, npy_name))
        try:
            npy_arr = np.load(os.path.join(hmdb_npy_path, npy_name))
            loaded_npy[i] = time_crop(npy_arr,cfg._NUM_FRAMES)
        except Exception as err:
            print(err)
            print('Exception caught with video:', npy_name)
        i+=1


    return loaded_npy

def load_vids_from_path(filenames, path, load_func='opencv', is_train=False):
    '''Raw video loader for files in the string array filenames. each string in the list
    Must be a valid path of a .avi or .mpeg file'''

    if isinstance(load_func, str):
        if load_func == 'opencv':
            load_func = load_video_opencv_fixed_frames
        elif load_func == 'opencv_full':
            load_func = load_video_opencv
        elif load_func == 'imageio':
            load_func = load_video_imageio
        elif load_func == 'skvideo':
            load_func = load_video_skvideo
        else:
            raise ValueError('Unknown load function')

    videos = []
    for file in filenames:
        # print('loading file:',os.path.join(path, file))
        # start=timer()
        vid = np.array(load_func(os.path.join(path, file), is_train=is_train))
        # end=timer()
        # print(file,' - time loading:',end-start)
        if len(vid.shape) is not 4:
            raise ValueError('Video '+file+' has wrong shape: '+str(vid.shape))
        # print(file+': vid.shape in load_vids is', vid.shape)

        videos.append(vid)

    return videos

def load_vids_opencv(filenames, path, is_train=False):
    return load_vids_from_path(filenames, path, load_func='opencv', is_train=is_train)

def load_vids_skvideo(filenames, path, is_train=False):
    return load_vids_from_path(filenames, path, load_func='skvideo', is_train=is_train)

def load_vids_imageio(filenames, path, is_train=False):
    return load_vids_from_path(filenames, path, load_func='imageio', is_train=is_train)

def load_hmdb_npy_rgb(filenames, path, is_train=False):
    return load_hmdb_npy(filenames, type='rgb', hmdb_npy_path = path, is_train=is_train)

def load_hmdb_npy_flow(filenames, path, is_train=False):
    return load_hmdb_npy(filenames, type='flow', hmdb_npy_path = path, is_train=is_train)


def load_single_frame_as_video(folders_with_frames, path,
                is_train=False, frames_to_get=cfg._NUM_FRAMES,
                return_index = False):
    '''Loads a single frame from folder with frames and expands it to frames_to_get frames'''

    first_pass=True
    for i,f in enumerate(folders_with_frames):
        frames = np.load(os.path.join(path, f))
        # print('frames.shape',frames.shape)
        if first_pass:
            h = frames.shape[1]
            w = frames.shape[2]
            ch = frames.shape[3]
            batch = np.zeros((len(folders_with_frames), frames_to_get, h, w, ch))
            first_pass=False
        batch[i]= np.repeat(frames[0][np.newaxis, :,:,:], frames_to_get, axis=0)

    return batch



def load_frames(folders_with_frames, path,
                is_train=False, frames_to_get=cfg._NUM_FRAMES,
                return_fixed_frames=True, return_index=False,
                remove_suffix=False, ex_on_empty_folder=True, stride=2):
    '''Loads frames, considering that all frames in a given folder have the same width and height.'''

    batch=[]
    start_frames=[]

    if remove_suffix:
        if folders_with_frames[0].endswith('.mp4'):
            folders_with_frames = [f[:-4] for f in folders_with_frames]

    for j in range(0, len(folders_with_frames), stride):
        dirpath = os.path.join(path,folders_with_frames[j])
        if not os.path.isdir(dirpath):
            if ex_on_empty_folder:
                raise ValueError('Folder '+os.path.join(path,f)+' does not exist')
            else:
                batch.append(np.zeros((frames_to_get, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, 3)))
                continue

        first_pass=True
        frames = sorted([n for n in os.listdir(dirpath) if (n.endswith('.jpg') or n.endswith('.png') or n.endswith('.jpeg'))])
        if not frames:
            if ex_on_empty_folder:
                raise ValueError('Folder '+dirpath+' has zero frames in it')
            else:
                batch.append(np.zeros((frames_to_get, cfg._IMAGE_SIZE, cfg._IMAGE_SIZE, 3)))
                continue

        n_frames = len(frames)

        if not frames_to_get:
            frames_to_get = n_frames

        if n_frames <= frames_to_get*stride:
            
            for i in range(0, n_frames, stride):
                img = image.load_img(os.path.join(path, f, frames[i]))
                img = image.img_to_array(img)
                if first_pass:
                    rows,cols,channels = img.shape
                    imgs = np.zeros((n_frames, rows, cols, channels))
                    first_pass=False
                imgs[i]=img

            if is_train or return_fixed_frames:
                imgs = time_crop(imgs,frames_to_get)

            start_frames.append(0)
            batch.append(imgs)

        else:
            if is_train:
                start_f = np.random.randint(n_frames-frames_to_get)
            else:
                start_f = n_frames//2-(frames_to_get*stride)//2

            for i in range(0, frames_to_get*stride, stride):
                idx = start_f+i
                img = image.load_img(os.path.join(path, f, frames[idx]))
                img = image.img_to_array(img)
                if first_pass:
                    rows,cols,channels = img.shape
                    imgs = np.zeros((frames_to_get, rows, cols, channels))
                    first_pass=False
                imgs[i]=img

            start_frames.append(start_f)
            batch.append(imgs)
    if return_index:
        return batch, start_frames
    return batch


def load_frames_and_bbox(folders_with_frames, path,
                is_train=False, frames_to_get=cfg._NUM_FRAMES,
                return_fixed_frames=True):
    ''' Loads both frames and bboxes, and returns botheleents in a list.
    Made for the RoiPooling model, where a ground truth bbox is now used
    to know what slice of the feature map to take.

    Inputs:
    -------
    folders_with_frames: list of str
        List of all folders to be processed where frames are located.
        usually has size of a batch. Example: ['climb/vid1', 'run/vid2']
    path: str
        path to dataset
    is_train; bool
        Whether we are in training phase or not
    frames_to_get: int
        Number of frames to get from the video, or to reloop the vid if smaller.
    return_fixed_frames: bool
        Whether to always return frames-to-get frames or to be more lenient (smaller vids are temporally untouched)

    Outputs:
    --------
    batch: list of numpy arrays corresponding to the videos in folders_with_frames


    '''

    frame_batch, start_frames = load_frames(folders_with_frames, path,
                    is_train, frames_to_get,
                    return_fixed_frames, return_index=True)


    bbox_batch = load_bboxes(folders_with_frames,path,start_frames,
                            frames_to_get, return_fixed_frames)

    return [frame_batch, bbox_batch]

def load_bboxes(video_names,path,start_frames,
                frames_to_get=cfg._NUM_FRAMES,
                return_fixed_frames=True):

    for i,v in enumerate(video_names):
        bbox=np.load(os.path.join(path, v, 'bboxes.npy'))
        bbox[:,2] = bbox[:,2]-bbox[:,0]
        bbox[:,3] = bbox[:,3]-bbox[:,1]
        n_frames = bbox.shape[0]
        new_bbox = np.zeros(frames_to_get, 4)
        if bbox.shape[0] <= frames_to_get:
            if not return_fixed_frames:
                batch.append(bbox)
                continue
            for j in frames_to_get:
                new_bbox[j] = bbox[j%n_frames]
        else:
            new_bbox = bbox[start_frames[i]:start_frames[i]+frames_to_get]

        batch.append(new_bbox)

    return batch


def load_crops_raw(folders_with_frames, path, is_train=False):
    '''Loads frames, assuming that they're crops. Crops might not have the same width and height from one frame to the other.'''

    DESIRED_SIZE = cfg._IMAGE_SIZE
    CHANNELS = 3
    vids=[]

    for f in folders_with_frames:
        crops = sorted([n for n in os.listdir(os.path.join(path,f)) if (n.endswith('.jpg') or n.endswith('.png') or n.endswith('.jpeg'))])
        if not crops:
            raise ValueError('Folder '+os.path.join(path,f)+' has zero frames in it')
        frames=np.zeros((len(crops), DESIRED_SIZE, DESIRED_SIZE, CHANNELS), dtype=np.uint8)
        for i,c in enumerate(crops):
            frames[i]=resize_crop(os.path.join(path,f,c), DESIRED_SIZE)
        vids.append(frames)
    return vids

def load_crops_fixed(folders_with_frames, path,
                    is_train=False, frames_to_get=cfg._NUM_FRAMES,
                    return_fixed_frames=True):
    '''load crops, but always the amount indicated in frames-to-get, not more or less.

    Inputs:
    -------
    folders_with_frames: list of str
        List of all folders to be processed where frames are located.
        usually has size of a batch. Example: ['climb/vid1', 'run/vid2']
    path: str
        path to dataset
    is_train; bool
        Whether we are in training phase or not
    frames_to_get: int
        Number of frames to get from the video, or to reloop the vid if smaller.
    return_fixed_frames: bool
        Whether to always return frames-to-get frames or to be more lenient (smaller vids are temporally untouched)

    Returns:
    --------
    batch: list of numpy arrays
        Batch of sequnces composed of crops (which have not been necessarily resized).

    '''

    DESIRED_SIZE = cfg._IMAGE_SIZE
    CHANNELS = 3
    batch=[]

    for f in folders_with_frames:
        crops = sorted([n for n in os.listdir(os.path.join(path,f)) if (n.endswith('.jpg') or n.endswith('.png') or n.endswith('.jpeg'))])
        if not crops:
            raise ValueError('Folder '+os.path.join(path,f)+' has zero frames in it')
        n_frames = len(crops)


        if n_frames <= frames_to_get:
            frames=np.zeros((n_frames, DESIRED_SIZE, DESIRED_SIZE, CHANNELS), dtype=np.uint8)
            for i in range(n_frames):
                frames[i]=resize_crop(os.path.join(path,f,crops[i]), DESIRED_SIZE)

            if is_train or return_fixed_frames:
                frames = time_crop(frames,frames_to_get)

        else:
            if is_train:
                start_f = np.random.randint(n_frames-frames_to_get)
            else:
                start_f = n_frames//2-frames_to_get//2
            frames=np.zeros((frames_to_get, DESIRED_SIZE, DESIRED_SIZE, CHANNELS), dtype=np.uint8)
            for i in range(frames_to_get):
                idx = start_f+i
                frames[i]=resize_crop(os.path.join(path,f,crops[idx]), DESIRED_SIZE)

        batch.append(frames)

    return batch


def load_crops_in_streams(stream_folders, path, streams_to_concat=4, is_train=True):

    CROP_SIZE = cfg._IMAGE_SIZE
    FRAMES_PER_STREAM = 16
    CHANNELS = 3
    EXTENSION = 'extend10'

    vids = []
    for f in stream_folders:
        streams = sorted(os.listdir(os.path.join(path, f, EXTENSION)))
        # print('len streams:',len(streams))
        init = 0
        if len(streams)<=streams_to_concat:
            streams_to_concat=len(streams)
        elif is_train:
            init = np.random.randint(len(streams)-streams_to_concat)


        frames=np.zeros((FRAMES_PER_STREAM*streams_to_concat, CROP_SIZE, CROP_SIZE, CHANNELS))
        i=0
        for s in streams[init:init+streams_to_concat]:
            crops = sorted(os.listdir(os.path.join(path,f,EXTENSION,s)))
            if len(crops)>16:
                crops = crops[:16]
            for c in crops:
                frames[i]=resize_crop(os.path.join(path,f,EXTENSION,s,c), CROP_SIZE)
                i+=1
        vids.append(frames)

    return vids

def load_crops_raw_and_stream(folders, path, streams_to_concat=4, is_train=True):

    if not folders:
        raise ValueError('No folders to load received in load_crops_raw_and_stream function')

    raw_folders = []
    stream_folders = []
    for f in folders:
        # print('folder:',f)
        is_stream = False
        files = os.listdir(os.path.join(path, f))
        for file in files:
            if 'extend' in file:
                is_stream = True
                break
            elif '.jpg' in file or '.png' in file:
                break
        if is_stream:
            stream_folders.append(f)
            # print('appended to stream')
        else:
            raw_folders.append(f)
            # print('appended to raw')

    if len(stream_folders)+len(raw_folders)==0:
        raise ValueError('Folder length sums to zero. INFO DUMP\
        \nFolders: %s\nstream_folders: %s\nraw_folders: %s\n'
        % (str(folders), str(raw_folders), str(stream_folders)))

    # print('length stream, raw folder names:', len(stream_folders), len(raw_folders))
    vids_raw = load_crops_fixed(raw_folders, path=path, is_train=is_train)
    vids_stream = load_crops_in_streams(stream_folders, path=path, streams_to_concat=streams_to_concat, is_train=is_train)
    vids_concat = vids_raw+vids_stream
    # print('len(vids_raw, vids_stream)',len(vids_raw), len(vids_stream))

    if len(vids_raw)+ len(vids_stream)==0:
        raise ValueError('Loaded video sizes sums to zero. INFO DUMP\
        \nFolders: %s\nstream_folders: %s\nraw_folders: %s \nlen(vids_raw): %s\nlen(vids_stream): %s\
        \nvids_raw: '+str(vids_raw)+'\nvids_stream:'+str(vids_stream)
        % (str(folders), str(raw_folders), str(stream_folders), str(len(vids_raw)),str(len(vids_stream))))


    return vids_concat


def resize_crop(crop, desired_size=cfg._IMAGE_SIZE):

    # Open
    im = Image.open(crop)
    old_size = im.size  # old_size[0] is in (width, height) format

    # Generate image ratio
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # Resize
    im = im.resize(new_size, Image.ANTIALIAS)

    # Create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im


# previously called load_labels
def load_labels_video(filenames, label_dictionary, dataset_path=None, label_pos_in_string=-2):

    labels = np.zeros((len(filenames),len(label_dictionary)), dtype=np.int16)
    i=0
    for file in filenames:
        classname = file.split('/')[label_pos_in_string]
        labels[i, label_dictionary[classname] ] = 1
        i+=1

    return labels

def load_labels_mem(filenames, str2label_dict, dataset_path=None, label_pos_in_string=None, remove_slash=True):

    labels = np.zeros(len(filenames))
    for i,file in enumerate(filenames):
        if '/' in file and remove_slash:
            file = file.split('/')[-1]
        # print('FILE IN LOAD LABELS MEM:',file)
        labels[i] = str2label_dict[file] if isinstance(str2label_dict[file], float) else str2label_dict[file][0]


    return labels

def load_labels_mem_alpha(filenames, str2label_dict, dataset_path=None, label_pos_in_string=None):

#     print('filenames in load labels mem alpha',filenames)
    labels = np.zeros((len(filenames), 2))
    for i,file in enumerate(filenames):
        # print('FILE IN LOAD LABELS MEM:',file)
        labels[i] = str2label_dict[file]

    return labels

def load_labels_as_list(filenames, str2label_dict, dataset_path=None, label_pos_in_string=None):

    labels = []
    for i, file in enumerate(filenames):
        labels.append( str2label_dict[file] )
    return np.array(labels)

def load_labels_frame(filenames, str2label_dict, dataset_path=None, label_pos_in_string=None):
    labels=[]
    for file in filenames:
        # The numpy file loaded here must be a matrix of frames x n_classes.
        # For each frame, there has to be a corresponding one-hot encoded vector indicating the correct class
        labels.append(np.load(os.path.join(dataset_path+'_aux', 'frame_labels', file+'.npy')))
    return labels

def augment_imgaug(X, augment):
    seq = get_imgaug_sequence(augment)
    return augment_video_batch(X, seq)

def augment_raw(X, augment):
    return augment_video_batch(X, augment)


#### THREADSAFE
import threading
class threadsafe_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
    def __iter__(self):
        return self
    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


#######################
### GENERATOR CLASS ###
#######################

class VideoSeqGenerator(keras.utils.Sequence):

    '''Generates video data for Keras models. Extends keras Sequence class,
    making it work nicely with fit_generator and evaluate_generator, as well as
    multiprocessing.

    '''

    def __init__(self,
                txt_file=None,
                batch_size=1,
                files=None,
                label_csv="../../../nfs_share/datasets/HMDB51/videos_aux/labels.csv",
                str2label_dict=None,
                augment=None,
                load_labels_func = 'per_video',
                load_func='hmdb_npy_rgb',
                preprocess_func=None,
                augment_func=None,
                dataset_path=None,
                shuffle=True,
                is_train=True,
                verbose=1,
                use_imgaug=True,
                return_labels=True,
                remove_excess_files=True,
                has_label_col=False,
                it_per_epoch=None,
                sample_weights=None,
                input_captions=None,
                len_vocab=cfg.LEN_VOCAB):

        'Initialization'

        self.txt_file = txt_file
        self.batch_size = batch_size
        self.files = files
        self.augment = augment
        self.load_func = load_func
        self.preprocess_func = preprocess_func
        self.augment_func = augment_func
        self.load_labels_func = load_labels_func
        self.dataset_path = dataset_path
        self.label_csv = label_csv
        self.shuffle = shuffle
        self.is_train = is_train
        self.verbose = verbose
        self.use_imgaug = use_imgaug
        self.return_labels = return_labels
        self.times=[timer()]
        self.sample_weights=sample_weights
        self.input_captions=input_captions

        self.it_per_epoch = it_per_epoch
        self.epoch_counter = 0
        self.count = 0
        self.len_vocab = len_vocab

        load_func_dict = {  'vids_opencv': load_vids_opencv,
                            'vids_imageio': load_vids_imageio,
                            'vids_skvideo': load_vids_skvideo,
                            'npy_rgb': load_hmdb_npy_rgb,
                            'npy_flow': load_hmdb_npy_flow,
                            'crops_raw': load_crops_raw,
                            'crops_fixed': load_crops_fixed,
                            'crops_stream': load_crops_in_streams,
                            'crops_mixed': load_crops_raw_and_stream,
                            'frames_raw': load_frames,
                            'single_frame': load_single_frame_as_video}

        preprocess_func_dict = { 'i3d_rgb': preprocess_i3d_rgb,
                                 'i3d_flow': preprocess_i3d_flow,
                                 'fast_rgb': preprocess_rgb_fast,
                                 'fast_flow': preprocess_flow_fast,
                                 'cap_fast_rgb': preprocess_rgb_fast_with_caption,
                                 'cap_fast_flow': preprocess_flow_fast_with_caption,
                                  }

        augment_func_dict = {'augment_imgaug': augment_imgaug,
                            'augment_raw': augment_raw}

        load_labels_dict = {'per_video': load_labels_video,
                            'per_frame': load_labels_frame,
                            'mem': load_labels_mem,
                            'mem_alpha': load_labels_mem_alpha,
                            'as_list': load_labels_as_list}

        ## Reading files to load from txt file
        # start=timer()
        if has_label_col:
            if self.files is None:
                self.files, self.label_array = read_videofile_txt(self.txt_file, ret_label_array=has_label_col)
            else:
                self.label_array = [f[1] for f in self.files]
                self.files = [f[0] for f in self.files]
        else:
            if self.files is None:
                self.files = read_videofile_txt(self.txt_file)
            self.label_array = None



        self.full_file_list = self.files
        self.num_files = len(self.full_file_list)

        ## If limiting number of iterations per epoch
        if self.it_per_epoch and self.it_per_epoch*self.batch_size > self.num_files:
            print('Caution, it_per_epoch is bigger than the amount of iterations a normal epoch takes. \
                    Disregarding parameter - epochs will have %d iterations, as normal.' % self.num_files//self.batch_size)
            self.it_per_epoch = None

        ## Checking if batch size is multiple of num files
        excess_files = self.num_files % self.batch_size
        if excess_files != 0:
            print('Caution: batch_size (%d) is not multiple of given dataset length (%d). This can cause issues with multigpu training.' % (self.batch_size, self.num_files) )
            if remove_excess_files:
                print('Truncating given files: removing last %d elements' % excess_files)
                self.files = self.files[:-excess_files]
                self.num_files = len(self.files)
                print('New num_files:', self.num_files)


        ## Defining load function to use
        if isinstance(self.load_func, str):
            self.load_func = load_func_dict[self.load_func]

        ## Defining preprocess function to use
        if isinstance(self.preprocess_func, str):
            self.preprocess_func = preprocess_func_dict[self.preprocess_func]

        ## Defining augment function to use
        if isinstance(self.augment_func, str):
            self.augment_func = augment_func_dict[self.augment_func]


        ## Defining label function to use
        if isinstance(self.load_labels_func, str):
            self.load_labels_func = load_labels_dict[self.load_labels_func]


        ## Shuffling files and setting list of files to use if it_per_epoch is not None
        self.on_epoch_end()

        ## Build dictionary to switch from string label to numbered label
        if self.return_labels:
            if str2label_dict is None:
                print('Building stringtolabel')
                self.str2label_dict = build_str2label(label_csv)
            else:
                self.str2label_dict = str2label_dict
            self._original_str2label_dict = self.str2label_dict.copy()

        if self.verbose:
            print('Calling VideoSeqGenerator. Batch size: ',self.batch_size,
            '. Number of files received:',self.num_files,'. Augmentation: ',self.augment)


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.it_per_epoch:
            return self.it_per_epoch
#         if not self.is_train:
#             return np.ceil(self.num_files / self.batch_size)
#         else:
        return self.num_files // self.batch_size


    def __getitem__(self, index):
        'Generate one batch of data'

        start = timer()

        ###DEBUG
#         print('\n\nIndex: '+str(index)+' - Files/folders to load in this call:',self.files[index*self.batch_size:(index+1)*self.batch_size])

        files_to_use = self.files[index*self.batch_size:(index+1)*self.batch_size]

        if len(files_to_use)==0:
            raise ValueError('Slice of files has no elements. INFO DUMP:\nself.batch_size: %d\
            \nindex: %d\nlen(self.files): %d\nself.files[index*self.batch_size]: %s' % (self.batch_size, index, len(self.files), '-'))

        kwargs = {}
        if self.input_captions is not None:
            kwargs['idx_cap'] = np.random.randint(len(self.input_captions[files_to_use[0]]))
            kwargs['idx_seq'] = {n:np.random.randint(len(self.input_captions[n][kwargs['idx_cap']])) for n in files_to_use}
            kwargs['input_captions'] = self.input_captions
            kwargs['len_vocab'] = self.len_vocab

        ## LOAD DATA
        X = self.load_func(files_to_use,
                           self.dataset_path,
                           is_train=self.is_train,
                           **kwargs)


        if self.return_labels:
            Y = self.load_labels_func(files_to_use,
                                        self.str2label_dict,
                                        self.label_array,
                                        **kwargs)


        if self.sample_weights is not None:
            W = self.sample_weights[index*self.batch_size:(index+1)*self.batch_size]

        t1=timer()

        ## PREPROCESS DATA
        if self.preprocess_func is not None:
            X = self.preprocess_func(X, is_train=self.is_train)

        t2=timer()

        ## AUGMENT DATA
        if self.augment_func is None:
            if self.augment is not None:
                if self.use_imgaug:
                    X = augment_imgaug(X, self.augment)
                else:
                    X = augment_raw(X, self.augment)


        else:
            X = self.augment_func(X, self.augment)

        # Always do random flipping.
        if self.is_train:
            if np.random.choice([False, True]):
                if type(X)==list and len(X)==2:
                    X[0] = np.flip(X[0], axis=3)
                else:
                    X = np.flip(X, axis=3)

        t3=timer()


        if not (self.count % cfg._FREQ_OF_TIME_PRINTS) and self.verbose:
            print('\nTotal time on this batch:', t3-start,
            ' - it:',self.count, '- idx:',index)
            print('Time loading:', t1-start)
            print('Time preprocessing:', t2-t1)
            print('Time augmenting:', t3-t2)

        ## DEBUG
#         plot_frames(X[0], is_01image=True, frames_to_show=4)
#         print('Y',Y)

        self.count+=1
        if self.return_labels:
            if self.sample_weights is not None:
                return X, Y, [W]+[np.ones(len(W))]*(len(Y)-1)
            else:
                return X, Y
        else:
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        if self.it_per_epoch:
            if self.epoch_counter == 0:
                if self.shuffle:
                    random.shuffle(self.full_file_list)
            virt_idx = (self.epoch_counter+1)*self.it_per_epoch*self.batch_size
            if virt_idx >= self.num_files:
                self.files = self.full_file_list[(self.epoch_counter*self.it_per_epoch*self.batch_size)%self.num_files:]
                self.files.extend(self.full_file_list[:virt_idx%self.num_files])
                self.epoch_counter = 0
            else:
                self.files = self.full_file_list[self.epoch_counter*self.it_per_epoch*self.batch_size:virt_idx]
                self.epoch_counter+=1
        else:
            if self.shuffle:
                if self.sample_weights is None:
                    random.shuffle(self.files)
                else:
                    idxs = list(range(len(self.files)))
                    np.random.shuffle(idxs)
                    self.files = self.files[idxs]
                    self.sample_weights = self.sample_weights[idxs]

        # print(f'Files to see in this epoch: {len(self.files)} \
        # (it_per_epoch = {self.it_per_epoch}, batch_size = {self.batch_size}, epoch_counter = {self.epoch_counter})')


class VideoCapGenerator(VideoSeqGenerator):
    pass


#####################
### AUGMENTATION ####
#####################

def get_imgaug_sequence(augment):
    # TODO: Modify raw imgaug code to play nicely with videos, mainly take batches
    # of vids and process them all at once in matrix manner. Might improve speed.

    seq = iaa.Sequential()

    AUG_NUM = 11
    act_array = np.random.rand(AUG_NUM) >0.5
    p = np.random.rand(AUG_NUM)

    for a in augment:
        if a == 'ch_shift' and act_array[0]:
#             print(a, p[0])
            seq.add(iaa.Multiply((0.5+p[0], 0.65+p[0]), per_channel=1))
        elif a == 'br_shift' and act_array[1]:
#             print(a, p[1])
            seq.add(iaa.Multiply((0.8+p[1]*.6, 0.9+p[1]*.6)))
        elif a == 'edge' and act_array[2]:
#             print(a, p[2])
            seq.add(iaa.Sharpen(alpha=p[2]/2+.1, lightness=p[2]/2+.6))
        elif a == 'dropout' and act_array[3]:
#             print(a, p[3])
            seq.add(iaa.CoarseDropout(np.random.rand()*.15, size_percent=(p[3]*.2+.005)))
        elif a == 'crop' and act_array[4]:
#             print(a, p[4])
            seq.add(iaa.Crop(percent=(0.05, .15))) # crop images from each side by 0 to 16px (randomly chosen)
        elif a == 'flip' and act_array[5]:
#             print(a, p[5])
            seq.add(iaa.Fliplr(1.0))
        elif a == 'stretch' and act_array[6] :
#             print(a, p[6])
            seq.add(iaa.Affine(scale={'x':(1.2,1.5),'y':(1.2,1.5)}))
        elif a == 'crop_pad' and act_array[7]:
#             print(a, p[7])
            seq.add(iaa.CropAndPad(
            # This is: top, right, bot, left. Negative = crop, pos = pad
            percent=((0, -0.1), (p[7]*0.3+0.1, p[7]*0.3+0.05) , (0, -0.1), (p[7]*0.3+0.1, p[7]*0.3+0.05))))
        elif a == 'shear' and act_array[8]:
#             print(a, p[8])
            seq.add(iaa.Affine(shear=(p[8]*-10, p[8]*10)))
        elif a == 'cutout' and act_array[9]:
#             print(a, p[9])
            seq.add(iaa.Lambda(cutout_img, heatmap_func, keypoint_func))
        elif a == 'simplex' and act_array[10]:
#             print(a, p[10])
            seq.add(iaa.SimplexNoiseAlpha(
                first=iaa.Multiply(0), sigmoid_thresh=iap.Normal(8.0, 3.0),
                per_channel=False))


    return seq


def augment_video_batch(batch, augment):
    if isinstance(augment, iaa.Augmenter):
        for i in range(len(batch)):
            if np.random.choice([True,False]):
                batch[i] = augment.augment_images(batch[i])
    else:
        for i in range(len(batch)):
            if 'crop' in augment:
                batch[i] = spatial_crop(batch[i])
            if 'flip' in augment:
                batch[i] = random_flip(batch[i])
            if 'edge' in augment:
                batch[i] = edge_enhancement(batch[i])
            if 'stretch' in augment:
                batch[i] = stretch_video(batch[i])
            if 'ch_shift' in augment:
                batch[i] = channel_shift(batch[i])
            if 'br_shift' in augment:
                batch[i] = brightness_shift(batch[i])
            if 'cutout' in augment:
                batch[i] = cutout_video(batch[i])
            if 'rotate' in augment:
                batch[i] = rotate_video(batch[i])
            if 'shear' in augment:
                batch[i] = shear(batch[i])

    return batch





def stretch_video():
    pass

def heatmap_func(hm_on_images, random_state, parents, hooks):
    return hm_on_images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


def cutout_img(images, random_state, parents, hooks):
    fill_value = 0 #if np.random.rand()<0.5 else -1

    for img in images:
        H,W,C = img.shape
        cutout_size = np.random.randint(H*0.18)+int(H*0.1)
        x = np.random.randint(W-cutout_size+1)
        y = np.random.randint(H-cutout_size+1)
        img[y:y+cutout_size,x:x+cutout_size,:] = fill_value
    return images

def cutout_video(vid, cutout_percent_height=0.35, prob = 0.5):

    r = np.random.rand()
    if r<prob:
        # print('Cutout triggered')
        F,H,W,C = vid.shape
        cutout_size = int(cutout_percent_height*H)
        if cutout_size >= W:
            cutout_size = W//2

        fill_value = 0 #if np.random.rand()<0.5 else -1

        # In 50% of the cases, generate uniform cutout accross frames.
        if r<prob/2:
            x = np.random.randint(W-cutout_size+1)
            y = np.random.randint(H-cutout_size+1)

            vid[:,y:y+cutout_size,x:x+cutout_size,:] = fill_value

        # In the other 50%, randomize the cutout for every frame
        else:
            for i in range(F):
                x = np.random.randint(W-cutout_size+1)
                y = np.random.randint(H-cutout_size+1)

                vid[i,y:y+cutout_size,x:x+cutout_size,:] = fill_value

    return vid

def channel_shift(vid, filter):
    raise NotImplementedError()

def brightness_shift():
    raise NotImplementedError()

def random_flip(vid, prob=0.5):
    if np.random.rand()<prob:
        # print('Flip triggered')
        F,H,W,C = vid.shape
        for i in range(F):
            for j in range(C):
                vid[i,:,:,j] = np.fliplr(vid[i,:,:,j])

    return vid

def edge_enhancement(vid, prob=0.5, method = 'sobel'):

    if np.random.rand()<prob:
        # print('Edge enhancement triggered')
        if method =='sobel':
            @adapt_rgb(each_channel)
            def sobel_each(image):
                if isinstance(image[0][0], np.uint8):
                    return image//2+sobel(image)
                return np.clip((image*0.5+sobel(image)+0.5)/2, -1, 1)
            edge_filt = sobel_each
        elif method == 'roberts':
            edge_filt = roberts
        elif method == 'scharr':
            edge_filt = scharr
        elif method == 'prewitt':
            edge_filt = prewitt
        else:
            raise ValueError('Unknown edge enhancemenet method: '+method)

        for i in range(len(vid)):
            vid[i] = edge_filt(vid[i])

    return vid

def pca_enhancement():
    raise NotImplementedError()


def build_hmdb_textfiles(hmdb_splitfiles_dir,  output_dir='', split_n=1):
    """ Builds textfiles tailored to HMDB dataset required for the keras generator video_generator.

        Args
        ----
        hmdb_splitfiles_dir: string
            Directory where the splitfiles are located (downloaded from hmdb website)
        output_dir: string
            directory to save the txt files in
        split_n: int
            split number to generate files for (has to be 1,2 or 3)

        Raises
        -----
        ValueError if split_n is not 1 2 or 3

    """
    if split_n >=4 or split_n <=0:
        raise ValueError('HMDB51 has only 3 splits - split_n must be 1, 2 or 3.')

    txt_names = os.listdir(hmdb_splitfiles_dir)
    train_vids=[]
    test_vids=[]
    for name in txt_names:
        if 'split'+str(split_n) in name:
            print('looking in', name)
            classname = name.split('_test')[0]
            with open(os.path.join(hmdb_splitfiles_dir,name),'r') as f:
                lines=f.readlines()
            for l in lines:
                if l[-3]=='1':
                    train_vids.append(classname+'/'+l[:-4])
                elif l[-3]=='2':
                    test_vids.append(classname+'/'+l[:-4])

            print('len(train_vids)',len(train_vids))
            print('len(test_vids)',len(test_vids))

    # Saving txts
    with open(os.path.join(output_dir, 'train_split0'+str(split_n)+'.txt'), 'w') as f:
        for v in train_vids:
            f.write("%s\n" % v)

    # Saving test txts
    with open(os.path.join(output_dir, 'test_split0'+str(split_n)+'.txt'), 'w') as f:
        for v in test_vids:
            f.write("%s\n" % v)



def read_videofile_txt(txt_path, ret_label_array=False):
    files=[]
    if ret_label_array:
        label_array=[]
    with open(txt_path, 'r') as f:
        for line in f:
            # print(line)
            filename = line[:-1].split(',')[0].split(' ')[0]
            files.append(filename)
            if ret_label_array:
                label_array.append(line.split(',')[1])
    if ret_label_array:
        return files, label_array
    else:
        return files


# THE ABOVE FUNCTION COULD BE REPLACED BY SIMPLY [l[:-1] for l in open(txt_path, 'r')]
def read_txt_oneliner(txt_path):
    return [l[:-1] for l in open(txt_path, 'r')]


def show_N_results(model, txt_file, N=5, show_top_5=True, idxs=[0,16,32,48,-1], l2s_dict={}, augment = None,
                    dataset_path=None, csv_path=None, prep_func=None, load_func='hmdb_npy_rgb',
                    func_gen = True, shuffle=False, use_imgaug=True):
    if func_gen:
        vid_gen = video_generator(txt_file, dataset_path=dataset_path,
                                batch_size=1, augment = augment, shuffle=shuffle, is_train=False,
                                label_csv=csv_path, load_func=load_func, preprocess_func=prep_func,
                                use_imgaug=use_imgaug)
    else:
        vid_gen = VideoSeqGenerator(txt_file, dataset_path=dataset_path,
                                batch_size=1, augment = augment, shuffle=shuffle, is_train=False,
                                label_csv=csv_path, load_func=load_func, preprocess_func=prep_func,
                                use_imgaug=use_imgaug)



    if N is None:
        N = len([l for l in open(txt_file)])
    y_true=[]
    y_pred=[]
    is_in_top_5=0
    for i in range(N):
        if func_gen:
            batch = next(vid_gen)
        else:
            batch = vid_gen[i]
        y_true.append(np.argmax(batch[1],axis=1)[0])
        prediction = model.predict(batch[0])
        y_pred.append(np.argmax(prediction, axis=1)[0])
        top_5_preds = np.argsort(prediction[0])[-5:]
        if y_true[i] in top_5_preds:
            is_in_top_5+=1

        if l2s_dict:
            str_pred = l2s_dict[y_pred[i]]
            str_true = l2s_dict[y_true[i]]
        else:
            str_pred=str_true='-'
        caption = 'Predicted: '+str_pred+' / '+str(y_pred[i])+' - True: '+str_true+' / '+str(y_true[i])

        # print('np.min(batch[0][0])',np.min(batch[0][0]))
        # print('np.max(batch[0][0])',np.max(batch[0][0]))


        plot_frames(batch[0][0], title=caption, idxs =idxs)
        if show_top_5:
            print('Top 5 predictions:')
            for idx in top_5_preds[::-1]:
                print(l2s_dict[idx],' with prob %.3f' % prediction[0][idx])
            print()


    print('Accuracy in results shown:')
    print('TOP 1:', np.mean(np.equal(y_true,y_pred)))
    print('TOP 5:', is_in_top_5/N )








### VIDEO GENERATOR AS FUNCTION - DEPRECATED
# @threadsafe_generator
# def video_generator(txt_file=None,
#                     batch_size=1,
#                     files=None,
#                     label_csv="../../../datasets/HMDB51/videos_aux/labels.csv",
#                     augment=None,
#                     load_func='hmdb_npy_rgb',
#                     preprocess_func=None,
#                     augment_func=None,
#                     label_func='per_video',
#                     dataset_path=None,
#                     shuffle=True,
#                     is_train=True,
#                     verbose=1, use_imgaug=False,
#                     return_labels=True):

#     '''
#     DEPRECATED! Use videoSeqGenerator instead

#     Keras Generator for videos. Example flow:
#             # 0. Turn to numpy array
#             # 1. Resize
#             # 2. Rescale
#             # 3. Resample fps (optional)
#             # 4. Random time crop
#             # 5. Random space crop
#             # 6. Augmentation

#     The first lines define the function dictionaries for generator generalization.
#     If using a new dataset or architecture, add the corresponding load_func for the dataset
#     and the preprocess_func for the architecture to the dictionaries.

#     Args:
#     -----
#     txt_file: string
#         file containing the names of the vids to load, in format "classname/videoname.avi".
#         The list will be shuffled or not, depending on the flag.
#     batch_size: int
#         number of videos to load per batch.
#     label_csv: string
#         The path to the csv file containing tuples (class_name, numeric_label) for the current dataset
#     type: string
#         One of flow or rgb. selects which type of data should be loaded.
#         This assumes that flow data augmentationdata is available.
#     shuffle: boolean
#         wether to shuffle the incoming data list or not.
#     augment: list of strings
#         Array of augmentation strings. Currently supported: 'cutout', 'flip', 'edge'

#     Yields:
#     -------
#     (X,Y): tuple with batch of videos and corresponding labels.
#     '''


#     load_func_dict = {  'vids_opencv':load_vids_imageio,
#                         'vids_imageio': load_vids_imageio,
#                         'vids_skvideo': load_vids_skvideo,
#                         'npy_rgb': load_hmdb_npy_rgb,
#                         'npy_flow': load_hmdb_npy_flow,
#                         'crops_raw': load_crops_raw,
#                         'crops_fixed': load_crops_fixed,
#                         'crops_stream': load_crops_in_streams,
#                         'crops_mixed': load_crops_raw_and_stream,
#                         'frames_raw': load_frames }

#     preprocess_func_dict = { 'i3d_rgb': preprocess_i3d_rgb,
#                              'i3d_flow': preprocess_i3d_flow,
#                              'fast_rgb': preprocess_fast}

#     augment_func_dict = {'augment_imgaug': augment_imgaug,
#                         'augment_raw': augment_raw}

#     load_labels_dict = {'per_video': load_labels_video,
#                         'per_frame': load_labels_frame}


#     ## Reading files to load from txt file
#     # start=timer()
#     if files == None:
#         files=read_videofile_txt(txt_file)

#     ## Defining load function to use
#     if isinstance(load_func, str):
#         load_func = load_func_dict[load_func]

#     ## Defining preprocess function to use
#     if isinstance(preprocess_func, str):
#         preprocess_func = preprocess_func_dict[preprocess_func]

#     ## Defining augment function to use
#     if isinstance(augment_func, str):
#         augment_func = augment_func_dict[augment_func]

#     ## Defining label function to use
#     if isinstance(load_labels_func, str):
#         load_labels_func = load_labels_dict[load_labels_func]

#     ## Shuffling files
#     L = len(files)
#     if shuffle:
#         random.shuffle(files)

#     ## This is called once at the beggining of the training, so the shuffle happens once, at that point.
#     if verbose:
#         print('Calling video_generator. Batch size: ',batch_size,'. Number of files received:',L,'. Augmentation: ',augment)

#     ## Build dictionary to switch from string label to numbered label
#     if return_labels:
#         str2label_dict = build_str2label(label_csv)


#     ## This line is just to make the generator infinite, keras needs that
#     while True:

#         ## Define starting idx for batch slicing
#         batch_start = 0
#         batch_end = batch_size
#         c=1

#         ## Loop over the txt file while there are enough unseen files for one more batch
#         while batch_start < L:
#             ## LOAD DATA
#             limit = min(batch_end, L)
#             # print('STEP',c,' - yielding', limit-batch_start,'videos.')
#             start = timer()
#             ## load_func can return either a numpy array of shape (batch, frames, h, w, c)
#             ## or a list of videos with different n_frames, widths and heights. preprocess_func has to work accordingly.
#             if dataset_path is not None:
#                 X = load_func(files[batch_start:limit], dataset_path, is_train=is_train)
#             else:
#                 X = load_func(files[batch_start:limit], is_train=is_train)
#             if return_labels:
#                 Y = load_labels_func(files[batch_start:limit], str2label_dict, dataset_path)

#             ## AUGMENT DATA
#             if augment_func is None:
#                 if augment is not None:
#                     if use_imgaug:
#                         X = augment_imgaug(X, augment)
#                     else:
#                         X = augment_raw(X, augment)
#             else:
#                 X = augment_func(X, augment)

#             ## PREPROCESS DATA
#             if preprocess_func is not None:
#                 X = preprocess_func(X, is_train=is_train)

#             t1 = timer()
#             if not c % cfg._FREQ_OF_TIME_PRINTS:
#                 print('Time elapsed loading, augmenting and preprocessing:', t1-start, ' - it:',c)

#             ## YIELD DATA
#             if return_labels:
#                 yield X,Y #a tuple with two numpy arrays with batch_size samples
#             else:
#                 yield X
#             ## Increasing idxs for next batch
#             batch_start += batch_size
#             batch_end += batch_size
#             c+=1
