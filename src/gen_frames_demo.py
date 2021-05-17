#!/usr/bin/env python


import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm
import scipy.misc
from generator import read_videofile_txt
import os
import shutil
from generator import build_label2str
from predict_and_save_kitty import extract_bbox_for_line
import cv2

def save_vid_with_labels(kitty_folder, video_path, frame_output_folder, label_csv, id_format_colon=False, verbose=True):
    '''Saves a video frame by frame where a bounding box is drawn around the detected
    persons and an action label is provided.
    Needs a kitty folder where the kitty files have action labels.'''

    vid = imageio.get_reader(video_path,  'ffmpeg')

    total_frames=len(vid)

    kitty_files = sorted(os.listdir(kitty_folder))
    frame_idxs = [int(n.split('.')[0].split('_')[-1])-1 for n in kitty_files]

    if verbose:
        print('Video loaded, len frame_idxs:', len(frame_idxs), 'len vid:', len(vid))

    # Get label to string dict
    label2str_dict = build_label2str(label_csv)
    label2str_dict[-1] = 'undefined'

    print('label2str_dict',label2str_dict)

    if not os.path.exists(frame_output_folder):
        os.mkdir(frame_output_folder)

    for num in tqdm(range(total_frames)):
        # check for valid frame number
#         if  num >= 0 &  num <= totalFrames:
#             # set frame position
#             cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
#         ret, img = cap.read()
#         print('ret:',ret)
        img = vid.get_data(num)

        if num in frame_idxs:
            # Read txt file
            txt_line_list = read_videofile_txt(os.path.join(kitty_folder,kitty_files[num]))

#             print('txt_line_list:',txt_line_list)

            # Get all bboxes for this frame
            for j,line in enumerate(txt_line_list):
                # Extract id
                if id_format_colon:
                    id_ = int(float(line.split(' ')[0].split(':')[-1]))
                else:
                    id_ = int(float(line.split(' ')[1]))

                # Extract action label
                act_label = int(float(line.split(' ')[-1]))

                if act_label == -1:
                    font_size=0.5
                    font_color = (200,200,0)
                    bbox_color = (100,0,0)
                else:
                    font_size=0.8
                    font_color = (255,255,0)
                    bbox_color = (255,0,0)

                text_label = label2str_dict[act_label]
#                 print('text_label:', text_label)

                # Getting bbox
                crop, bbox = extract_bbox_for_line(line, img, idx_bbox=3, margin=0.0, show=False, debug=False, k=1.0)

                left, right, top, bottom = bbox

                cv2.rectangle(img, (left,top), (right,bottom), bbox_color, 2)
                cv2.putText(img, text_label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, lineType=cv2.LINE_AA)

#             if not num%10:
#                 plt.figure()
#                 plt.imshow(img)
#                 plt.show()

        imageio.imwrite(os.path.join(frame_output_folder, str(num)+'.jpg'), img)


if __name__ == "__main__":

    save_vid_with_labels('../../../p2_metropolis/tmp/cfosco/VIDEOS/KITTI_CFOSCOnyc_c0110_2/with_action_labels',
                    '../../../p2_metropolis/tmp/cfosco/VIDEOS/nyc_c0110_2.mp4',
                    frame_output_folder = '../tmp_vid', label_csv='../../../nfs_share/datasets/IVA_Videos/crops_mixed_aux/labels_5.csv')
