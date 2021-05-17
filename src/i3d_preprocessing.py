#!/usr/bin/env python
'''
Preprocessing for training and inference on DeepMind's Inception-v1 Inflated 3D CNN for action recognition.
The model is introduced in:
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
'''

__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"


## Imports
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from vid_utils import load_video_opencv


def compare_frames(frames1, frames2, names = ['before', 'after'], is_optical_flow=False, is_scaled=True, idxs=[0]):
    ''' Compares two set of frames processed for the i3d model. Works on flow and rgb frames'''

    for idx in idxs:
        if not is_optical_flow:
            fr1 = (frames1[idx]+1)/2
            fr2 = (frames2[idx]+1)/2
        else:
            fr1 = np.zeros((frames1[idx].shape[0],frames1[idx].shape[1],3))
            fr1[:,:,:2] = frames1[idx]
            fr1+=0.5
            fr2 = np.zeros((frames2[idx].shape[0],frames2[idx].shape[1],3))
            fr2[:,:,:2] = frames2[idx]
            fr2+=0.5

        print('Plotting first frames')
        plt.figure(figsize=[10,10])
        plt.subplot(1,2,1)
        plt.imshow(fr1)
        plt.title(names[0])
        plt.subplot(1,2,2)
        plt.imshow(fr2)
        plt.title(names[1])
        plt.show()


def compute_optical_flow(frames):
    ''' Computes optical flow with the TV-L1 algorithm.'''

    flow_frames=[]
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    for i in tqdm(range(len(frames)-1)):
        prvs = frames[i]
        next = frames[i+1]
        hsv = np.zeros((prvs.shape[0],prvs.shape[1],3))
        hsv[...,1] = 255

        # Calculating TV-L1 optical flow
        flow = optical_flow.calc(prvs, next, None)

        flow_frames.append(flow)

    return np.array(flow_frames)


def preprocess_and_save(video_dir, video_name, out_dir='../data', type='joint'):
    ''' Preprocesses video to make it adhere to the i3d needs, and saves the
    resulting frames as npy array.'''

    frames = load_video_opencv(os.path.join(video_dir, video_name))

    # Convert color to RGB.
    cvt_frames = convert_rgb(frames)

    # Sample at 25 fps.
    resa_frames = resample_video(cvt_frames, fps=25)

    # Resize, smallest dim 256 pixels, bilinear interpolation
    resi_frames = video_resize(resa_frames)

    if type in ['rgb','joint']:
        preprocess_rgb_and_save(resi_frames, npy_dir=out_dir, video_name=video_name)
    if type in ['flow', 'joint']:
        preprocess_flow_and_save(resi_frames, npy_dir=out_dir, video_name=video_name)
    print(video_name,'processed, numpy files saved.')



def preprocess_rgb_and_save(frames, npy_dir='../data', video_name='vid'):
    ## RGB Stream

    # Rescale between -1 and 1
    resc_frames = video_rescale(frames)

    # During training, randomly select a 224x224 crop. During test, take center 224x224 crop.
    crop_frames = video_spatial_crop(resc_frames, size=224, random=False, vid_name=video_name)

    # making sure that we're saving as float32, and removing last frame to match dimensionality of flow
    crop_frames = np.array(crop_frames[:-1,...],dtype=np.float32)

    # Save as .npy to run through evaluate script.
    np.save(os.path.join(npy_dir,video_name[:-4]+'_rgb.npy'),np.expand_dims(crop_frames,axis=0))


def preprocess_flow_and_save(frames, npy_dir='../data', video_name='vid'):
    ## Optical Flow Stream

    # Convert to grayscale
    gray_frames = []
    for frame in frames:
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    gray_frames = np.array(gray_frames)

    # TV-L1 optical flow algorithm
    print('Computing optical flow...')
    flow_frames = compute_optical_flow(gray_frames)

    # Pixel values truncated to [-20,20], then rescaled to -1,1
    resc_flow_frames = video_truncate_and_rescale(flow_frames)

    # Only use first 2 output dims.
    resc_flow_frames = resc_flow_frames[:,:,:,:2]

    # Apply same cropping as RGB.
    crop_flow_frames = video_spatial_crop(resc_flow_frames, size=224, random=False, vid_name=video_name)

    # making sure that we're saving as float32
    crop_flow_frames = np.array(crop_flow_frames,dtype=np.float32)

    # Save as .npy to run through evaluate script.
    np.save(os.path.join(npy_dir,video_name[:-4]+'_flow.npy'), np.expand_dims(crop_flow_frames,axis=0))



def crops2npy(crop_folder, output_folder):
    ''' Transforms crops obtained by previous nvidia intern into
    npy files that can be used by i3d. Saves the npys in folders corresponding
    to their classnames.'''

    folders = os.listdir(crop_folder)

    for fol in folders:
        in_folds = os.listdir(fol)

        if len(in_folds) != 3:
            print('WARNING: folders != 3')
            print(in_folds)

        transform_into_npy()







if __name__ == "__main__":

    dataset = 'HMDB51'
    data_dir = '../../../datasets/'+dataset

    class_list = os.listdir(data_dir)

    already_processed = ['brush_hair', 'catch', 'chew','climb','dive','drink','fall_floor','fencing',
    'flic_flac','handstand','hit', 'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'shoot_ball',
    'shoot_bow', 'sit', 'smoke', 'swing_baseball', 'throw', 'turn', 'walk']

    for cl in already_processed:
        class_list.remove(cl)

    for cl in class_list:
        print('Processing',cl, '...')
        out_dir = os.path.join('../data/',dataset,cl)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        video_list = os.listdir(os.path.join(data_dir,cl))

        for vid in video_list[m:]:
            try:
                preprocess(os.path.join(data_dir,cl), vid, out_dir = out_dir, type='flow')
            except Exception as err:
                print('Exception caught for video:', os.path.join(cl,vid))
                print('Error was:', err)
                print('Moving on...')
