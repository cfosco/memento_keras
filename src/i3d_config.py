'''Defaults for i3d training with the i3d_keras_training script.
These are the defaults used if no parameter is specified through the command line.
To correctly train i3d, you can either set all parameters on this file and
just run i3d_keras_training.py, or send parameters through the command line call.
Paremeters sent that way will overwrite the ones defined here.'''

# Data specific parameters
_DATASET = None
_DATA_PATH = None
_INIT_CKPT = None #'../../../ckpt/best/weights_hmdb_rgb_ep50_valloss1.12_acc72.11.hdf5'
_PREP_FUNC = None
_LOAD_FUNC = None
_NUM_CLASSES = None
_SPLIT = 1
_TYPE = 'rgb'

# Train specific parameters
_EPOCHS = 50
_STEPS = None
_VAL_STEPS = None
_BATCH_SIZE_TRAIN = 10
_LR = 0.0001
_SAVE_CKPT = '../../../ckpt'
_WORKERS_TRAIN = 8
_USE_MP_TRAIN = 0
_MAX_QUEUE_TRAIN = 20
_AUGMENT = "cutout,flip"
_OPTIMIZER = 'Adam'

# Eval specific parameters
_BATCH_SIZE_EVAL = 10
_WORKERS_EVAL = 8
_USE_MP_EVAL = 0
_MAX_QUEUE_EVAL = 20
_RESULT_OUTPUT_PATH = '../results'
_TXT_PATH = None
_VAL_TXT_PATH = None
_LABELS_PATH = None


# Constants for nets
_IMAGE_SIZE = 224
_NUM_FRAMES = 42

# Other defaults
_PATH_HMDB = '../../../datasets/HMDB51/preproc_videos'
_PATH_UCF = '../../../datasets/UCF101'
_VERBOSE = 1
_GPUS = 4
_CLASSES = None
_INTERNAL_I3D_CKPT_COSMOS = '../../../ckpt/i3d_best/weights_HMDB51_all_split1_rgb_ep50_valloss1.12_valacc72.11.hdf5'
_INTERNAL_I3D_CKPT_LOCAL= '../../../nfs_share/ckpt/i3d_best/weights_HMDB51_all_split1_rgb_ep50_valloss1.12_valacc72.11.hdf5'
_USE_PRETRAINED = True
_DOWNSAMPLE_F = None
_DL_WEIGHTS = False
_DROPOUT_PROB = 0.3
_REG_FACTOR = 1e-4
_FREEZE_OLD = False
_RETURN_FIXED_FRAMES = True
_FREQ_OF_TIME_PRINTS = 500
_RUN_LOCALLY = False
_ROI_POOL_LAYER = 5
_DETECTOR_CKPT_LOCAL = '../../../main_share/ar_main/ckpt/detector/resnet18_3class_people_face_bag_phase1_union.hdf5'
_DETECTOR_CKPT_COSMOS = '../../../ckpt/detector/resnet18_3class_people_face_bag_phase1_union.hdf5'

# CAPTION CONFIG
MAX_CAP_LEN = 50
LEN_VOCAB =  2988 #3873
_VOCAB_PATH = '../../memento_data/vocab_3.json'
