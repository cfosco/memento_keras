# Imports
import os
import cv2
import sys
sys.path.append('../src')
from generator import load_vids_from_path, load_frames
import argparse
from timeit import default_timer as timer
import numpy as np

def test_loading_functions(idx=0, num_files=4):
    dataset = '../../../datasets/Kinetics'
    files=[f[:-1] for f in open('../../../datasets/Kinetics_aux/train_split01.txt', 'r')]
    tries=5

    # vids = files[idx:idx+num_files]
    # frame_folders=[]
    # for f in vids:
    #     splits =f.split('/')
    #     frame_folders.append(splits[0]+'_frames'+'/'+splits[1]+'/'+splits[2][:-4])

    frame_folders = ['train_frames/abseiling/0pP5PNGEDi0_000159_000169',
                 'train_frames/abseiling/1Qj6Mu_CpnM_000042_000052',
                 'train_frames/abseiling/2SRJgIEtZVs_000239_000249',
                 'train_frames/abseiling/3D7uhXK_6-M_000086_000096']

    vids = ['train/abseiling/0pP5PNGEDi0_000159_000169.mp4',
         'train/abseiling/1Qj6Mu_CpnM_000042_000052.mp4',
         'train/abseiling/2SRJgIEtZVs_000239_000249.mp4',
         'train/abseiling/3D7uhXK_6-M_000086_000096.mp4']

    times_vids = []
    times_frames=[]
    for i in range(tries):
        print(i)
        start=timer()
        videos = load_vids_from_path(vids, dataset, 'opencv', is_train=False)
        end=timer()
        times_vids.append(end-start)
    for i in range(tries):
        print(i)
        start=timer()
        videos = load_frames(frame_folders, dataset, is_train=False)
        end=timer()
        times_frames.append(end-start)

    print('Avg time loading videos:', np.mean(times_vids), times_vids)
    print('Avg time loading frames:', np.mean(times_frames), times_frames)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## MAIN ARGUMENTS
    parser.add_argument('--idx',
        help='Index of files to load',
        type=int,
        default=0)
    parser.add_argument('--num_files',
        help='Number of files to load',
        type=int,
        default=4)

    args = parser.parse_args()

    test_loading_functions(args.idx, args.num_files)
