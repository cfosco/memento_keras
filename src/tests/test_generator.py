
'''Test file for the keras video generator. Needs HMDB and UCF to be available.'''

__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"


## Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from generator import video_generator, build_hmdb_textfiles
from vid_utils import plot_frames
from timeit import default_timer as timer


# These tests are currently meant to be run from a notebook. Go to "notebooks/2.0_video_generator_tests.ipynb to run them"

# Path to files (when running from notebook)
test_file = '../src/tests/generator_test_files_hmdb.txt'
test_file_ucf = '../src/tests/generator_test_files_ucf.txt'

hmdb_path = '../../../preprocessed_data/HMDB51'
hmdb_path_raw = '../../../datasets/HMDB51'
ucf_path_raw = '../../../datasets/UCF101'

hmdb_labels_path = '../../../datasets/HMDB51_aux/labels.csv'
ucf_labels_path = '../../../datasets/UCF101_aux/labels.csv'


### HMDB-based tests
def test_generator_rgb_sizes():

    gen = video_generator(test_file, dataset_path=hmdb_path, batch_size=4, load_func='hmdb_npy_rgb')
    batch = next(gen)


    print('len(batch)', len(batch))
    print('X.shape', batch[0].shape)
    print('Y.shape',batch[1].shape)

    assert(len(batch)==2)
    assert(batch[0].shape == (4, 64, 224, 224, 3))
    assert(batch[1].shape == (4,51))

def test_generator_flow_sizes():

    gen = video_generator(test_file, dataset_path=hmdb_path, batch_size=4, load_func='hmdb_npy_flow')
    batch = next(gen)

    print('len(batch)', len(batch))
    print('X.shape', batch[0].shape)
    print('Y.shape',batch[1].shape)

    assert(len(batch)==2)
    assert(batch[0].shape == (4, 64, 224, 224, 2))
    assert(batch[1].shape == (4,51))


def test_generator_rgb_class():

    start = timer()

    gen = video_generator(test_file, dataset_path=hmdb_path, batch_size=4, load_func='hmdb_npy_rgb')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)
    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=False)

    end= timer()
    print('Time spent in video gen test with no preprocessing:', end-start)

    assert(set(lab)==set([0,1,2,3]))

def test_generator_flow_class():

    gen =video_generator(test_file, dataset_path=hmdb_path,batch_size=4, load_func='hmdb_npy_flow')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)

    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=True)

    assert(set(lab)==set([0,1,2,3]))


def test_small_video_should_have_correct_dimensions():
    small_vid = 'kick_ball/Amazing_Soccer_#2_kick_ball_f_cm_np1_le_bad_2.avi'
    txt = '../src/tests/small_vid.txt'

    with open(txt, 'w+') as f:
        f.write(small_vid)

    gen = video_generator(txt, dataset_path=hmdb_path, batch_size=2)
    batch = next(gen)

    os.remove(txt)

    assert(batch[0][0].shape == (64,224,224,3))


def test_batch_size_bigger_than_txt_reuses_files():
    pass

def test_looping_works_correctly():
    small_vid = 'kick_ball/Amazing_Soccer_#2_kick_ball_f_cm_np1_le_bad_2.avi'
    txt = '../src/tests/small_vid.txt'

    with open(txt, 'w+') as f:
        f.write(small_vid)

    gen = video_generator(txt, dataset_path=hmdb_path, batch_size=1)
    batch = next(gen)

    vid = batch[0][0]

    print('mean of first frame:', vid[0].mean(), ' - mean of first looped frame:', vid[32].mean())

    assert(np.array_equal(vid[0],vid[32]))


## Load tests

def test_load_raw_hmdb_rgb_opencv():

    start = timer()
    gen = video_generator(test_file, dataset_path=hmdb_path_raw, batch_size=4, load_func='hmdb_raw_rgb', preprocess_func='i3d_rgb')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)
    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=False)

    end= timer()
    print('Time spent in opencv load + gen test:', end-start)

    assert(set(lab)==set([0,1,2,3]))


def test_load_raw_hmdb_rgb_imageio():
    start = timer()
    gen = video_generator(test_file, dataset_path=hmdb_path_raw, batch_size=4, load_func='hmdb_raw_rgb_imageio', preprocess_func='i3d_rgb')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)
    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=False)

    end= timer()
    print('Time spent in imageio load + gen test:', end-start)

    assert(set(lab)==set([0,1,2,3]))

def test_load_raw_hmdb_rgb_skvideo():
    start = timer()
    gen = video_generator(test_file, dataset_path=hmdb_path_raw, batch_size=4, load_func='hmdb_raw_rgb_skvideo', preprocess_func='i3d_rgb')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)
    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=False)

    end= timer()
    print('Time spent in skvideo load + gen test:', end-start)

    assert(set(lab)==set([0,1,2,3]))


#### UCF-based tests

def test_load_raw_ucf_rgb():
    gen = video_generator(test_file_ucf, dataset_path=ucf_path_raw, batch_size=4, label_csv=ucf_labels_path, load_func='ucf_raw_rgb', preprocess_func='i3d_rgb')
    batch = next(gen)

    lab = np.argmax(batch[1],axis=1)
    print(batch[1])

    plot_frames(batch[0][0], title='Label for this video:'+str(lab[0]), is_optical_flow=False)

    assert(set(lab)==set([14,39,57,1]))
