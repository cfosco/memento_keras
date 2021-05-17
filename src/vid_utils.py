'''
Utility script with plotting functions and other convenience methods for action recognition.
Some methods are made for running on jupyter notebooks.
'''
__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
# import skvideo.io
from random import shuffle
import i3d_config as cfg
import warnings

########################
## PLOTTING FUNCTIONS ##
########################

def plot_first_frames_batch(batch):
    """Takes a batch of (videos,labels) and plots the first, middle and end
    frame of each, in subplots. Works only for rgb images, not optical flow.

    Args
    ----
    batch: list of 2 numy arrays
        Contains the batch of videos and the corresponding labels.
        Can be obtained through video_generator. The videos are assumed to be
        scaled between -1 and 1.

    """
    i=0
    for vid in batch[0]:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow((vid[0]+1)/2)
        plt.subplot(1,3,2)
        plt.imshow((vid[32]+1)/2)
        plt.subplot(1,3,3)
        plt.imshow((vid[-1]+1)/2)
        plt.suptitle('Label for this video:'+str(batch[1][i]), y=0.8)
        plt.show()
        i+=1

def plot_frames(frames, title='', is_optical_flow=False,
                is_255image=False, is_01image=False, idxs=None, frames_to_show=3, suptitle_y=0.75):
    """ Plots frames indicated in idxs after rescaling them to 0-255.

    Args
    ----
    frames: numpy array
        Array of frames to show. Dims: (F,H,W,C)
    title: str
        title for the set of subplots
    is_optical_flow: Boolean
        if True, assumes that the input frames are 2-channel OF frames
        with values between -0.5 and 0.5
    is_255image: boolean
        if True, assumes that input is array of uint8 (values from 0 to 255)
    is_01image: Boolean
        if True, assumes that input is array of floats (decimal values between 0 and 1)
    idxs: list of ints
        list of frame indexes to plot. Overrides frames_to_show.
    frames_to_show:
        number of equidistant frames to show from video. Will always show first and last frame.
    """

    # print(np.max(frames))

    plt.figure(figsize=[16,6])
    c=1

    if idxs is None:
        idxs = [i*len(frames)//(frames_to_show-1) for i in range(frames_to_show-1)]
        idxs.append(-1)

    for idx in idxs:
        if not is_optical_flow:
            if not is_01image:
                if not is_255image:
                    fr1 = (frames[idx]+1)/2
                else:
                    fr1 = frames[idx]/255
            else:
                fr1 = frames[idx]
        else:
            fr1 = np.zeros((frames[idx].shape[0],frames[idx].shape[1],3))
            fr1[:,:,:2] = frames[idx]
            if not is_255image:
                fr1+=0.5
            else:
                fr1 /=255


        plt.subplot(1,len(idxs),c)
        plt.yticks([])
        plt.xticks([])
        plt.imshow(fr1)
        c+=1

    plt.tight_layout()
    # plt.subplots_adjust(top=0.85)
    plt.suptitle(title + ' min %.3f, max %.3f' % (np.min(fr1), np.max(fr1)), y=suptitle_y)
    plt.show()

def process_frames_for_display(frames):
    ''' This function processes the frame such that they can be easily displayed
    by functions like imshow. The input has to be a numpy array of preprocessed frames,
    for example an array of dimensions (F,H,W,C) with values rescaled between -1 and 1.
    '''

    pr_frames = []
    for frame in frames:
        if frame.shape[2] == 3:
            fr = (frame+1)*255/2
        else:
            fr = np.zeros((frame.shape[0],frame.shape[1],3))
            fr[:,:,:2] = frame
            fr+=0.5
            fr*=255
        pr_frames.append(fr)

    return np.array(pr_frames)



##################################
### FORMAT TRANSFORM FUNCTIONS ###
##################################

def npy2gif(input_file, output_file, fps=25, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)

    Args
    -----
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the given output filename has the .gif extension
    fname, _ = os.path.splitext(output_file)
    output_file = fname + '.gif'

    frames = np.load(input_file)[0,...]
    frames = process_frames_for_display(frames)

    print(frames.shape)

    # make the moviepy clip
    clip = ImageSequenceClip(list(frames), fps=fps).resize(scale)
    clip.write_gif(output_file, fps=fps)
    return clip


def crop2video(crop_folder_path, out_path='crop_video', width=224, height=224, fill_type='cst', fill_value=0,
                fit_larger_dim=True, pix_processing_func=None, channels=3, save_vid=False, save_npy=False):
    '''Transforms a set of image crops contained in `crop_folder_path` into a video
    by preprocessing, resizing and concatenating the frames. Each action is
    controllable through the input parameters.

    Args
    ----
    crop_folder_path: string
        The folder where the bitmap crops are located (must be in jpg, png or jpeg format)
    out_path: string
        path and filename to save the resulting video. The filename should be provided without extension,
        so as to benefit from the multiple saving options of this function. For example,
        when saving a numpy array, provide out_path='output_folder/filename_without_extension' and save_npy=True.
        The ".npy" extension will be automatically added.
    width: int
        The width of the output video
    height: int
        The height of the output video
    fill_type: one of 'cst', 'rand',
        How to fill the space around the crop, if any. The keyword 'cst' indicates constant value (fill_value),
        'rand' indicates random values between fill_value[0] and fill_value[1]
    fill_value: one of int, list of ints
        Value to fill the extra space of the crops, if any.
    fit_larger_dim: Boolean
        if true, the larer dimension of the crops will be fitted to the height or width of
        the video, while the smaller dimension ill be resized to maintain the aspect ratio.
        This will leave black spaces around the crop, which will be filled according to fill_type.
        If False, the smaller dim will be fitted to the height or width of the video, and so the
        image will be effecively re-cropped to fit the given width and heights.
    pix_processing_func: function
        Function to process the pixels of the loaded crops. If None, no function is applied.
        This will usually be a rescaling function or a common filter that the user wants to apply.
    channels: int
        number of channels of the output video. Defaults to 3.
    save_vid: boolean
        whether to save as a video or not. Default: False
    save_npy: boolean
        whether to save as a npy array or not. Default: False


    Returns
    -------
    video: numpy array
        matrix of dimensions (n_crops, height, width, channels) ocntained the processed and resized crop video.
    '''

    crop_names = os.listdir(crop_folder_path)

    if fill_type=='cst':
        video = np.ones((len(crop_names), height, width, channels))*fill_value
    elif fill_type=='rand':
        video = np.random.rand(len(crop_names), height, width, channels)*(fill_value[1]-fill_value[0])+fill_value[0]

    mid_h = height//2
    mid_w = width//2
    for i,c in enumerate(crop_names):
        crop = cv2.imread(os.path.join(crop_folder_path,c))

        if pix_processing_func:
            crop = pix_processing_func(crop)
        H,W,C =  crop.shape

        if W<H:
            new_h = height
            new_w = int(W*height/H)
        else:
            new_w = width
            new_h = int(H*width/W)

        resi_crop = cv2.resize(crop, (new_w, new_h))

        video[i,mid_h-new_h//2:mid_h-new_h//2+new_h,mid_w-new_w//2:mid_w-new_w//2+new_w,:C] = resi_crop

    if save_npy:
        np.save(out_path+'.npy', video)

    # TODO: build this function (openCV has some nice built in methods to do this)
    if save_vid:
        save_video(video)

    return video


def frames2video(frames, format='avi'):
    raise NotImplementedError()


def npy2float32(folder):
    classnames = os.listdir(folder)
    for c in classnames:
        filenames = os.listdir(os.path.join(folder, c))
        size_before = sum(os.path.getsize(os.path.join(folder,c,f)) for f in filenames)
        for f in filenames:
            # print('Processing',f)
            npy_file = np.load(os.path.join(folder,c,f))
            # print('Bytes before:',npy_file.nbytes)
            npy_file = np.array(npy_file, dtype=np.float32)
            # print('Bytes after:',npy_file.nbytes)
            np.save(os.path.join(folder,c,f), npy_file)
        size_after = sum(os.path.getsize(os.path.join(folder,c,f)) for f in filenames)
        print('Done with %s. Difference in size: %d' % (c, size_before-size_after))





#########################
### LOADING FUNCTIONS ###
#########################

def load_video_imageio(video_file, verbose=True):

    vid = imageio.get_reader(video_file,  'ffmpeg')
    frames=[]
    for image in vid.iter_data():
        frames.append(image)

    npframes = np.array(frames)
    print(npframes.shape)
    return npframes

def load_video_pyav(video_file, verbose=True):

    container = av.open(video_file)
    frames=[]
    for frame in container.decode():
        img = frame.to_image()  # PIL/Pillow image
        frames.append(np.array(img))  # numpy array
    npframes = np.array(frames)
    print(npframes.shape)
    return npframes

def load_video_skvideo(video_file, verbose=True):
    reader = skvideo.io.vreader(video_file)
    npframes=[]
    for frame in reader:
        npframes.append(frame)
    npframes = np.array(npframes)
    print('npframes.shape',npframes.shape)
    return npframes

def load_video_opencv(video_file,verbose=False, is_train=False):

    cap = cv2.VideoCapture(video_file)

    if verbose:
        print('video_file:', video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(video_file.split('/')[-1], 'loaded. \nwidth, height, n_frames, fps:',width, height, n_frames, fps)

    test_frames = []
    success=True
    # i=0
    while success:
        success,frame = cap.read()
        # i+=1
        if success:
            test_frames.append(frame)

    test_frames = np.array(test_frames)
    test_frames = convert_rgb(test_frames)

    if verbose:
        print('Frames appended in numpy array. Shape of test_frames:', test_frames.shape)

    return test_frames

def load_video_opencv_fixed_frames(video_file, verbose=True,
                                is_train=True,frames_to_get=cfg._NUM_FRAMES,
                                return_fixed_frames=cfg._RETURN_FIXED_FRAMES):
    cap = cv2.VideoCapture(video_file)

    vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channels = 3

    if vid_frames <= frames_to_get:
        vid=np.zeros((vid_frames, height, width, channels), dtype=np.uint8)

        for i in range(vid_frames):
            success,frame = cap.read()
            if success:
                vid[i] = frame

        if is_train or return_fixed_frames:
            vid = time_crop(vid, frames_to_get)

    else:
        vid = np.zeros((frames_to_get, height, width, channels), dtype=np.uint8)

        if is_train:
            f = np.random.randint(0, vid_frames-frames_to_get)
        else:
            f = vid_frames//2-frames_to_get//2

        cap.set(1,f)
        for i in range(frames_to_get):
            success,frame = cap.read()
            if success:
                vid[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return vid


####################################
### VIDEO MODIFICATION FUNCTIONS ###
####################################

def convert_rgb(frames):
    cvt_frames=[]

    for frame in frames:

        cvt_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return np.array(cvt_frames)

def time_crop(vid, N, is_train=True):
    # print('Frames of loaded video:', len(vid))
    vid_frames, H, W, C = vid.shape
    if vid_frames == N:
        return vid

    new_vid = np.zeros((N,H,W,C))

    if vid_frames > N:
        if is_train:
            idx = np.random.randint(vid_frames-N+1)
        else:
            idx =  vid_frames//2 - N//2
        new_vid = vid[idx:idx+N]
    elif vid_frames == 0:
        print('Warning: found video with 0 frames !!!')
        return new_vid
    else:
        n_loops = N//vid_frames
        remaining = N%vid_frames
        for n in range(n_loops):
            new_vid[vid_frames*n:vid_frames*(n+1)] = vid

        if remaining:
            new_vid[-remaining:] = vid[:remaining]

    return new_vid

def video_spatial_crop(frames, size=cfg._IMAGE_SIZE, randomize=False, vid_name=''):
    '''takes a square crop from input video of size `size`. If randomize = False,
    the center crop is taken. If not, the crop is random across the canvas.'''

    F,H,W,C = frames.shape

    if H <size:
        if H <= 10 or H==None:
            raise ValueError('Video has wrong height, or dimensions are incorrect')
        frames = np.concatenate((frames,np.zeros((F,size-H,W,C))),axis=1)
        H = size
        warning_string = 'Video %s : height = %d < size = %d' % (vid_name, H, size)
        print(warning_string)
        with open('preprocess_log.txt', 'a+') as f:
            f.write(warning_string)
    if W <size:
        if W <= 10 or W==None:
            raise ValueError('Video has wrong width, or dimensions are incorrect')
        frames = np.concatenate((frames,np.zeros((F,H,size-W,C))),axis=2)
        W = size
        warning_string = 'Video %s : width = %d < size = %d' % (vid_name, W, size)
        print(warning_string)
        with open('preprocess_log.txt', 'a+') as f:
            f.write(warning_string)

    if randomize:
        if W == size:
            x = 0
        else:
            x = np.random.randint(0,W - size)
        if H == size:
            y = 0
        else:
            y = np.random.randint(0,H - size)
    else:
        x = W//2 - size//2
        y =  H//2 - size//2

    cropped_frames = frames[:,y:y+size,x:x+size]
    return cropped_frames

def video_rescale(frames, mini=-1, maxi=1):
    if mini>= maxi:
        raise ValueError('Mini >= Maxi in video_rescale.')
    H,W,C =  frames[0].shape
    F = len(frames)
    resc_frames = np.zeros((F,H,W,C),dtype=np.float32)
    for i in range(F):
        resc_frames[i] = ((frames[i]/255.0)*(maxi-mini)+mini)

    return resc_frames

def video_resize(frames):

    NEW_SIZE = 256

    F, height , width , channels =  frames.shape

    if not F or not height or not width or not channels:
        warnings.warn('Frames received in video_resize have one or more zero dimensions: frames=[%d,%d,%d,%d]' % frames.shape)
        return np.zeros((F,cfg._IMAGE_SIZE,cfg._IMAGE_SIZE,channels))

    if height < width:
        if height == NEW_SIZE:
            return frames
        new_h=NEW_SIZE
        new_w=int(width*NEW_SIZE/height)
    else:
        if width == NEW_SIZE:
            return frames
        new_w=NEW_SIZE
        new_h = int(height*NEW_SIZE/width)

    resi_frames = np.zeros((F,new_h,new_w,channels))
    for i in range(F):
        resi_frames[i] = cv2.resize(frames[i], (new_w, new_h))

    return resi_frames

def video_resize_fixed(frames, new_w=544, new_h=960, return_scale=True):

    F,H,W,C = frames.shape

    if H == new_h and W == new_w:
        if return_scale_factor:
            return resi_frames, 1, 1
        return frames

    scale_x = new_w / W
    scale_y = new_h / H
    resi_frames = np.zeros((F,new_h,new_w,C))
    for i in range(F):
        resi_frames[i] = cv2.resize(frames[i], (new_w, new_h))

    if return_scale_factor:
        return resi_frames, scale_x, scale_y
    return resi_frames

def video_resample(frames, prev_fps = 30, fps=25):
    # TODO: finish this function. Not needed for the HMDB dataset,
    # but might be needed for more complex ones.
    return frames


def video_truncate_and_rescale(frames, thresh=20):
    '''Truncates input frames to [thresh,-thresh] and rescales to [-1,1] interval.'''

    return np.clip(frames, -thresh, thresh)/thresh

    # trunc_frames = np.copy(frames)
    # trunc_frames[frames>thresh]=thresh
    # trunc_frames[frames<-thresh]=-thresh
    # trunc_frames = trunc_frames/thresh
    # return trunc_frames



##### OTHER #####
from shutil import copytree

def move_crops(folder_in, folder_out):
    classes = os.listdir(folder_in)
    with open('../../../datasets/test_split01.txt', 'w+') as f:
        for c in classes:
            if 'intermediatedata' in os.listdir(os.path.join(folder_in, c)):
                bb_path = os.path.join(folder_in, c, 'intermediatedata','trackedbb')
                vids_with_crops = os.listdir(bb_path)
                for v in vids_with_crops:
                    crops = os.listdir(os.path.join(bb_path, v))
                    for crop in crops:
                        if '_r' in crop:
                            destination = os.path.join(folder_out,c,v+'_'+crop)
                            src = os.path.join(bb_path, v, crop)
                            # copytree(src,destination)
                            f.write(c+'/'+v+'_'+crop+'\n')



def write_txt_for_generator(data_folder, output_folder=None, train_test_split=0.7, file_prefix='', suffix=''):

    classnames = os.listdir(data_folder)

    if output_folder == None:
        output_folder = data_folder+'_aux'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    for classname in classnames:
        itemnames = os.listdir(os.path.join(data_folder, classname))
        # print(itemnames,'\n\n\n')
        shuffle(itemnames)
        # print(itemnames)
        if len(itemnames) <=1:
            print('Classname %s has one example or less. Omitting.' % classname)
            continue;
        lim = int(len(itemnames)*train_test_split)
        train_itemnames = itemnames[:lim]
        test_itemnames= itemnames[lim:]

        with open (os.path.join(data_folder+'_aux', 'train_split01'+suffix+'.txt'), 'a+') as file:
            for n in train_itemnames:
                file.write('%s%s/%s\n' % (file_prefix, classname, n))

        with open(os.path.join(data_folder+'_aux', 'test_split01'+suffix+'.txt'), 'a+') as file:
            for n in test_itemnames:
                file.write('%s%s/%s\n' % (file_prefix, classname, n))
