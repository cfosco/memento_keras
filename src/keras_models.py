'''Keras models for action recognition.
Contains full models, blocks and auxiliary functions.
'''
__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"


# Imports
import warnings
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras import layers
from keras.layers import Activation, Add
from keras.layers import Dense, Flatten
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D, Conv2D, LSTM
from keras.layers import MaxPooling3D, MaxPooling2D
from keras.layers import AveragePooling3D, AveragePooling2D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D
from keras.layers import TimeDistributed
from keras.layers import Embedding, Concatenate, Bidirectional
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.regularizers import l2
from keras.regularizers import l1
from keras.models import load_model
from keras.initializers import Constant
# from keras_contrib.applications.resnet import ResNet18
from keras.applications import ResNet50
from resnet3d import Resnet3DBuilder
import i3d_config as cfg
from attentive_lstm import AttentionLSTM

# import sys
# sys.path.append('../keras-utility-layer-collection/')
# from kulc.attention import ExternalAttentionRNNWrapper


# from resnet import ResNet


WEIGHTS_NAME = ['rgb_kinetics_only', 'flow_kinetics_only', 'rgb_imagenet_and_kinetics', 'flow_imagenet_and_kinetics']

# path to pretrained models with top (classification layer)
WEIGHTS_PATH = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels.h5'
}

# path to pretrained models with no top (no classification layer)
WEIGHTS_PATH_NO_TOP = {
    'rgb_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_kinetics_only' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_kinetics_only_tf_dim_ordering_tf_kernels_no_top.h5',
    'rgb_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/rgb_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5',
    'flow_imagenet_and_kinetics' : 'https://github.com/dlpbc/keras-kinetics-i3d/releases/download/v0.2/flow_inception_i3d_imagenet_and_kinetics_tf_dim_ordering_tf_kernels_no_top.h5'
}


###############################
##### BUILDING FUNCTIONS ######
###############################


def build_model_multigpu(ckpt=cfg._INIT_CKPT,
                         type=cfg._TYPE,
                         image_size=cfg._IMAGE_SIZE,
                         num_frames=cfg._NUM_FRAMES,
                         num_classes=cfg._NUM_CLASSES,
                         use_pretrained=True,
                         gpus=cfg._GPUS,
                         per_label=False,
                         verbose=True,
                         download_weights=True,
                         model_type='i3d',
                         downsample_factor=None,
                         run_locally=cfg._RUN_LOCALLY,
                         dropout_prob=cfg._DROPOUT_PROB,
                         final_activation='softmax',
                         show_internal_summary=False,
                         rescale=True,
                         embedding_matrix=None,
                         output_biases=None,
                         return_states=False):

    if model_type == 'i3d_per_label':
        build_func = build_i3d_per_label_custom
    elif model_type == 'i3d':
        build_func = build_i3d_custom
    elif model_type == 'i3d_rec':
        build_func = build_i3d_recurrent_head(pool_video_repr=True)
    elif model_type == 'i3d_rec_cap':
        build_func = build_i3d_recurrent_head(pool_video_repr=True, use_captions=True, output_biases=output_biases)
    elif model_type == 'i3d_rec_unpooled':
        build_func = build_i3d_recurrent_head(pool_video_repr=False)
    elif model_type == 'i3d_emb':
        build_func = build_i3d_sentence_emb(pool_video_repr=True)
    elif model_type == 'i3d_emb_unpooled':
        build_func = build_i3d_sentence_emb(pool_video_repr=False)
    elif model_type == 'i3d_cap_words_late':
        build_func = build_i3d_captioning(output_type='word', fusion='late', cap_module=captioning_module, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_cap_words_early':
        build_func = build_i3d_captioning(output_type='word', fusion='early', cap_module=captioning_module, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_cap_sentence_late':
        build_func = build_i3d_captioning(output_type='sentence', fusion='late', cap_module=captioning_module, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_cap_sentence_early':
        build_func = build_i3d_captioning(output_type='sentence', fusion='early', cap_module=captioning_module, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_cap_sentence_early_att':
        build_func = build_i3d_captioning(output_type='sentence', fusion='early', cap_module=captioning_module_attention, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_cap_sentence_early_extradense':
        build_func = build_i3d_captioning(output_type='sentence', fusion='early', cap_module=captioning_module, output_biases=output_biases, return_states=return_states, additional_dense=True)
    elif model_type == 'i3d_cap_sentence_early_bidirectional':
        build_func = build_i3d_captioning(output_type='sentence', fusion='early', cap_module=captioning_module_bidirectional, output_biases=output_biases, return_states=return_states)
    elif model_type == 'i3d_raw':
        build_func = build_i3d_raw
    elif model_type == 'dilated_2d3d':
        build_func = build_dilated_2d3d
    elif model_type =='dilated_i3d':
        build_func = build_dilated_i3d
    elif model_type == 'mict_i3d':
        build_func = build_mict_i3d
    elif model_type == 'resnet50_lstm':
        build_func = build_resnet50_lstm
    elif model_type == 'resnet18_lstm':
        build_func = build_resnet18_lstm
    elif model_type == 'resnet50_3d':
        build_func = build_resnet50_3d
    elif model_type == 'resnet18_3d':
        build_func = build_resnet18_3d
    elif model_type == 'tcn':
        build_func = build_tcn
    else:
        raise ValueError('Unknown model type: '+model_type)

    # print(ckpt, type, image_size, num_frames, num_classes, gpus, per_label)
    if gpus is not None and gpus >1:
        print('Building multi-gpu model with %d gpus' % (gpus) )

        with tf.device('/cpu:0'):
            full_model = build_func(ckpt=ckpt,
                                    type=type,
                                    image_size=image_size,
                                    num_frames=num_frames,
                                    num_classes=num_classes,
                                    use_pretrained=use_pretrained,
                                    verbose=verbose,
                                    download_weights=download_weights,
                                    downsample_factor=downsample_factor,
                                    run_locally=run_locally,
                                    dropout_prob=dropout_prob,
                                    final_activation=final_activation,
                                    show_internal_summary=show_internal_summary,
                                    embedding_matrix=embedding_matrix)

        full_model = multi_gpu_model(full_model, gpus=gpus)

    else:
        print('Working with single GPU')
        full_model = build_func(ckpt=ckpt,
                                type=type,
                                image_size=image_size,
                                num_frames=num_frames,
                                num_classes=num_classes,
                                verbose=verbose,
                                download_weights=download_weights,
                                downsample_factor=downsample_factor,
                                run_locally=run_locally,
                                dropout_prob=dropout_prob,
                                final_activation=final_activation,
                                show_internal_summary=show_internal_summary,
                                embedding_matrix=embedding_matrix)

    return full_model



def build_i3d_custom(ckpt=cfg._INIT_CKPT,
                     type=cfg._TYPE,
                     image_size=cfg._IMAGE_SIZE,
                     num_frames=cfg._NUM_FRAMES,
                     num_classes=cfg._NUM_CLASSES,
                     use_pretrained=True,
                     verbose=True,
                     download_weights=True,
                     downsample_factor=None,
                     dropout_prob=cfg._DROPOUT_PROB,
                     run_locally=cfg._RUN_LOCALLY,
                     final_activation='softmax',
                     show_internal_summary=False,
                     rescale=True,
                     embedding_matrix=None):

    weights=None
    if type == 'rgb':
        if use_pretrained and ckpt is None:
            weights = 'rgb_imagenet_and_kinetics'
        channels = 3
        resc_func = lambda x: x/255.0 *2 - 1
    elif type =='flow':
        if use_pretrained and ckpt is None:
            weights = 'flow_imagenet_and_kinetics'
        channels = 2
        resc_func = lambda x: K.clip(x,-20.0,20.0)/20.0
    else:
        raise ValueError('unknown network type')


    # We build an i3d for Kinetics (400 out classes) to correctly load the pretrained weights
    rgb_model = Inception_Inflated3d(include_top=True,
                                    weights=weights,
                                    input_shape=(num_frames, image_size, image_size, channels),
                                    download_weights=download_weights,
                                    dropout_prob=dropout_prob,
                                    downsample_factor=downsample_factor,
                                    classes=400)

    if show_internal_summary: rgb_model.summary()

    # Removing layers that we're not going to use
    rgb_model.layers.pop()
    rgb_model.layers.pop()
    rgb_model.layers.pop()
    rgb_model.outputs = [rgb_model.layers[-1].output]
    rgb_model.layers[-1].outbound_nodes = []


    # We define a new input layer
    in_layer = Input(shape=(num_frames, image_size, image_size, channels), name='i3d_input')

    # Adding rescaling layer
    if rescale:
        resc = Lambda(resc_func, name='rescale_lambda') (in_layer)  #tf.math.multiply(tf.math.divide(x,255.0),2.0)-1.0) (in_layer)

    x = rgb_model(resc)
    if verbose:
        print('x.shape before last conv',x.shape)

    # Connect the input to the last blocks of the net, which starts with one conv3d
    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    if verbose:
        print('x.shape after last conv',x.shape)

    # then reshape to (n_frames_remaining, classes) just to get rid of the two 1,1 dummy dimensions
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, num_classes), name='reshape')(x)

    # then mean over frames, which gives us the logits (that we have to average ith the flow logits in case of having a flow stream)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False), name='frame_pool_lambda')(x) #output_shape=lambda s: (s[0], s[2])

    # Finally the softmax to have a proba dist over the classes
    out = Activation(final_activation, name='prediction')(x)

    # Build the full model by creating a new one that takes the input we created and connects it to the out we defined above
    full_model = Model(in_layer, out, name='i3d_full')
    if verbose:
        full_model.summary()

    if ckpt is not None:
        full_model.load_weights(ckpt)

    return full_model







def build_i3d_per_label_custom(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None, run_locally=cfg._RUN_LOCALLY):

    if type == 'rgb':
        weights = 'rgb_imagenet_and_kinetics'
        channels = 3
    elif type =='flow':
        weights = 'flow_imagenet_and_kinetics'
        channels = 2
    else:
        raise ValueError('unknown network type')

    if ckpt is not None:
        weights=None

    # We build an i3d for Kinetics (400 out classes) to correctly load the pretrained weights
    rgb_model = Inception_Inflated3d(include_top= True,
                                    weights= weights,
                                    input_shape= (num_frames, image_size, image_size, channels),
                                    download_weights=download_weights,
                                    classes= 400)

    # Removing layers that we're not going to use
    rgb_model.layers.pop()
    rgb_model.layers.pop()
    rgb_model.layers.pop()
    rgb_model.outputs = [rgb_model.layers[-1].output]
    rgb_model.layers[-1].outbound_nodes = []


    # We define a new input layer
    in_layer = Input(shape=(num_frames, image_size, image_size, channels))
    x = rgb_model(in_layer)
    if verbose:
        print('x.shape before last conv',x.shape)

    # Connect the input to the last blocks of the net, which starts with one conv3d
    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    ## AT THIS POINT, x is LOGITS

    if verbose:
        print('x.shape after last conv',x.shape)

    # then reshape to (n_frames_remaining, classes) just to get rid of the two 1,1 dummy dimensions
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, 1, num_classes))(x)

    # here, instead of getting the mean over frames, we interpolate the final logits to
    # fit the original number of frames. We get a dense per-frame output
    x = Lambda(lambda x: tf.squeeze(tf.image.resize_images(x, [num_frames, 1]), [2]),
               output_shape=lambda s: (s[0], num_frames, s[3]))(x)

    if verbose:
        print('pre-softmax shape for per_label:', x.shape)

    # Finally the softmax, to have a proba dist over the classes
    out = Activation('relu', name='prediction')(x)

    # Build the full model by creating a new one that takes the input
    # we created and connects it to the out node we defined above
    full_model = Model(in_layer, out, name='i3d_full')
    if verbose:
        full_model.summary()

    if ckpt is not None:
        full_model.load_weights(ckpt)

    return full_model

def build_i3d_raw(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES,
                    use_pretrained=False, verbose=True, download_weights=True,
                    dropout_prob=0.2,downsample_factor=None, run_locally=cfg._RUN_LOCALLY):

    weights=None
    if type == 'rgb':
        if use_pretrained and ckpt is None:
            weights = 'rgb_imagenet_and_kinetics'
        channels = 3
    elif type =='flow':
        if use_pretrained and ckpt is None:
            weights = 'flow_imagenet_and_kinetics'
        channels = 2
    else:
        raise ValueError('unknown network type')


    # We build an i3d for Kinetics to correctly load the pretrained weights
    rgb_model = Inception_Inflated3d(include_top=False,
                                    weights=weights,
                                    input_shape=(num_frames, image_size, image_size, channels),
                                    download_weights=download_weights,
                                    downsample_factor=downsample_factor,
                                    classes= num_classes)

    # We define a new input layer
    in_layer = Input(shape=(num_frames, image_size, image_size, channels))
    x = rgb_model(in_layer)
    x = Dropout(dropout_prob)(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    print('x.shape after last conv:', x.shape)

    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, num_classes))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)

    out = Activation('softmax', name='prediction')(x)

    # Build the full model by creating a new one that takes the input we created and connects it to the out we defined above
    full_model = Model(in_layer, out, name='i3d_full')
    if verbose:
        full_model.summary()

    if ckpt is not None:
        full_model.load_weights(ckpt)


    # print(full_model.layers[1].layers[1].name, full_model.layers[1].layers[1].get_weights())

    return full_model


def captioning_module(input, 
                      embedding_matrix, 
                      initial_state=None,
                      embedding_trainable=False, 
                      units=512, 
                      output_type='word',
                      return_embedding=False, 
                      return_states=False):

    if output_type=='word':
        ret_seq = False
        out_layer = Dense(units, activation = 'relu', name='cap_words_dense_out')
    elif output_type=='sentence':
        ret_seq = True
        out_layer = TimeDistributed(Dense(units, activation = 'relu', name='cap_words_dense_out'), name='td_cap_words_dense_out')

    if type(initial_state)!= list and initial_state.shape[1] != units:
        state_h = Dense(units, activation='relu', name='initial_state_dense_h')(initial_state)
        state_c = Dense(units, activation='relu', name='initial_state_dense_c')(initial_state)
        initial_state = [None, [state_h, state_c]]

    emb = Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=embedding_trainable,
                    mask_zero=True,
                    name='embedding_caption')
    emb_out = emb(input)
    x, state_h1, state_c1 = LSTM(units, return_sequences=True, name='cap_words_lstm1', return_state=True)(emb_out, initial_state=initial_state[0])
    x, state_h2, state_c2 = LSTM(units, return_sequences=ret_seq, name='cap_words_lstm2', return_state=True)(x, initial_state=initial_state[1])
    output = out_layer(x)

    if return_states:
        return output, [state_h1, state_c1, state_h2, state_c2]
    if return_embedding:
        return output, emb_out
    else:
        return output
    
def captioning_module_attention(input, 
                      embedding_matrix, 
                      initial_state=None,
                      embedding_trainable=False, 
                      units=512, 
                      output_type='word',
                      return_embedding=False, 
                      return_states=False):

    if output_type=='word':
        ret_seq = False
        out_layer = Dense(units, activation = 'relu', name='cap_words_dense_out')
    elif output_type=='sentence':
        ret_seq = True
        out_layer = TimeDistributed(Dense(units, activation = 'relu', name='cap_words_dense_out'), name='td_cap_words_dense_out')

    print("input shape", input.shape)
    print("initial_state before dense",initial_state.shape)
    if type(initial_state)!= list and initial_state.shape[1] != units:
        state_h = Dense(units, activation='relu', name='initial_state_dense_h')(initial_state)
        state_c = Dense(units, activation='relu', name='initial_state_dense_c')(initial_state)
        print("state_h.shape",state_h.shape)
        initial_state = [None, [state_h, state_c]]

    emb = Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=embedding_trainable,
                    mask_zero=True,
                    name='embedding_caption')
    emb_out = emb(input)
    x, state_h1, state_c1 = AttentionLSTM(units, 
                                          return_sequences=True, 
                                          name='cap_words_lstm1', 
                                          return_state=True)([emb_out, state_c]) #, initial_state=initial_state[0]
    x, state_h2, state_c2 = AttentionLSTM(units, 
                                          return_sequences=ret_seq, 
                                          name='cap_words_lstm2', 
                                          return_state=True)([x, x]) #, initial_state=initial_state[1]
    output = out_layer(x)

    if return_states:
        return output, [state_h1, state_c1, state_h2, state_c2]
    if return_embedding:
        return output, emb_out
    else:
        return output   


def lag_input_module(x, lag_input):
    pass

def captioning_module_bidirectional(input, embedding_matrix, initial_state=None,
                            embedding_trainable=False, units=512, output_type='word', return_embedding=False, return_states=False):

    if output_type=='word':
        ret_seq = False
        out_layer = Dense(units, activation = 'relu', name='cap_words_dense_out')
    elif output_type=='sentence':
        ret_seq = True
        out_layer = TimeDistributed(Dense(units, activation = 'relu', name='cap_words_dense_out'), name='td_cap_words_dense_out')

    if type(initial_state)!= list and initial_state.shape[1] != units:
        state_h = Dense(units, activation='relu', name='initial_state_dense_h')(initial_state)
        state_c = Dense(units, activation='relu', name='initial_state_dense_c')(initial_state)
        initial_state = [None, [state_h, state_c, state_h, state_c]]

    emb = Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=embedding_trainable,
                    mask_zero=True,
                    name='embedding_caption')
    emb_out = emb(input)
    x, state_h1, state_c1 = Bidirectional(LSTM(units, return_sequences=True, name='cap_words_lstm1'), merge_mode='sum', return_state=True, name='bidir1')(emb_out, initial_state=initial_state[0])
    x, state_h2, state_c2 = Bidirectional(LSTM(units, return_sequences=ret_seq, name='cap_words_lstm2', return_state=True), merge_mode='sum', name='bidir2')(x, initial_state=initial_state[1])
    print('x.shape after Bidirectional', x.shape)
    output = out_layer(x)
    print('x.shape after out_layer', x.shape)

    if return_states:
        return output, [state_h1, state_c1, state_h2, state_c2]
    if return_embedding:
        return output, emb_out
    else:
        return output


def build_i3d_captioning(fusion = 'early', cap_module=captioning_module,
                        output_type='word', output_biases=None,
                         return_states=False,
                         additional_dense=False):

    def model_func(ckpt=cfg._INIT_CKPT,
                         type=cfg._TYPE,
                         image_size=cfg._IMAGE_SIZE,
                         num_frames=cfg._NUM_FRAMES,
                         num_classes=cfg._NUM_CLASSES,
                         use_pretrained=True,
                         verbose=True,
                         download_weights=True,
                         downsample_factor=None,
                         dropout_prob=cfg._DROPOUT_PROB,
                         run_locally=cfg._RUN_LOCALLY,
                         final_activation='softmax',
                         show_internal_summary=False,
                         rescale=True,
                         max_cap_len=50,
                         # size_vocab=1641,
                         embedding_matrix=None):

        weights=None
        if type == 'rgb':
            if use_pretrained and ckpt is None:
                weights = 'rgb_imagenet_and_kinetics'
            channels = 3
            resc_func = lambda x: x/127.5 - 1
        elif type =='flow':
            if use_pretrained and ckpt is None:
                weights = 'flow_imagenet_and_kinetics'
            channels = 2
            resc_func = lambda x: K.clip(x,-20.0,20.0)/20.0
        else:
            raise ValueError('unknown network type')


        # We build an i3d for Kinetics (400 out classes) to correctly load the pretrained weights
        rgb_model = Inception_Inflated3d(include_top=True,
                                        weights=weights,
                                        input_shape=(num_frames, image_size, image_size, channels),
                                        download_weights=download_weights,
                                        dropout_prob=dropout_prob,
                                        downsample_factor=downsample_factor,
                                        classes=400)

        if show_internal_summary: rgb_model.summary()

        # Removing layers that we're not going to use
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.outputs = [rgb_model.layers[-1].output]
        rgb_model.layers[-1].outbound_nodes = []

        # We define a new input layer
        in_video = Input(shape=(num_frames, image_size, image_size, channels))

        # Adding rescaling layer
        if rescale:
            resc = Lambda(resc_func) (in_video)

        video_out = rgb_model(resc)
        if verbose:
            print('video_out.shape (internal representation over 5 pseudo-frames)',video_out.shape)

        if additional_dense:
            x = Dense(512,name='extradense')(video_out)
            x = BatchNormalization(name='extradense_bn')(x)
            x = Activation('Mish',name='extradense_mish')(x)
            if verbose:
                print('shape after additional dense:',x.shape)

        # Connect the input to the last blocks of the net, which starts with one conv3d
        x = conv3d_bn(x if additional_dense else video_out, num_classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        if verbose:
            print('x.shape after last conv',x.shape)

        # then reshape to (n_frames_remaining, classes) just to get rid of the two 1,1 dummy dimensions
        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, num_classes))(x)

        # then mean over frames, which gives us the logits (that we have to average ith the flow logits in case of having a flow stream)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        # Finally the softmax to have a proba dist over the classes
        out_mem = Activation(final_activation, name='output_classes')(x)

        # Compute internal repr for captioning stream. n_b x 1024.
        video_repr = Lambda(lambda x: K.mean(K.squeeze(K.squeeze(x,axis=2),axis=2), axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[-1]))(video_out)

        ### CAPTIONING SUBMODULE:
        in_captions = Input(shape=(None,))

        ### STATE INPUTS (IF USING MODEL STATE BY STATE)
        if return_states:
            initial_state = Input(shape=(None, None))
        else:
            initial_state = video_repr

        if fusion == 'early':
            caption_repr = cap_module(in_captions, embedding_matrix, output_type=output_type, initial_state=initial_state, return_states=return_states)
            if return_states:
                caption_repr, states = caption_repr
            if output_type == 'sentence':
                out_captions = TimeDistributed(Dense(embedding_matrix.shape[0], activation = 'softmax', name = 'output_captions',
                                                bias_initializer=Constant(output_biases) if output_biases is not None else 'zeros'),
                                                name='td_output_captions')(caption_repr)
            else:
                out_captions = Dense(embedding_matrix.shape[0], activation = 'softmax', name = 'output_captions', bias_initializer=Constant(output_biases) if output_biases is not None else 'zeros')(caption_repr)

        else:
            caption_repr = cap_module(in_captions, embedding_matrix, output_type=output_type, intial_state=initial_state, return_states=return_states)
            if return_states:
                caption_repr, states = caption_repr
            if output_type == 'sentence':
                video_repr = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=max_cap_len, axis=1))(video_repr)
                # Concatenate video and caption representations
                conc = Concatenate(axis=-1)([caption_repr, video_repr])
                # Dense layers
                x = TimeDistributed(Dense(512, activation = 'relu', name='caption_stream_dense'))(conc)
                x = TimeDistributed(BatchNormalization(axis=-1, name='caption_stream_bn'))(x)

                # Final output, b_s x max_cap_len x len_vocab
                out_captions = TimeDistributed(Dense(embedding_matrix.shape[0], activation = 'softmax', name='output_captions', bias_initializer=Constant(output_biases) if output_biases is not None else 'zeros'),name='td_output_captions')(x)

            else:
                # Concatenate video and caption representations
                conc = Concatenate(axis=-1)([caption_repr, video_repr])
                # Dense layers
                x = Dense(512, activation = 'relu', name='caption_stream_dense')(conc)
                x = BatchNormalization(axis=-1, name='caption_stream_bn')(x)
                # Final output, b_s x len_vocab
                out_captions = Dense(embedding_matrix.shape[0], activation = 'softmax', name='output_captions', bias_initializer=Constant(output_biases) if output_biases is not None else 'zeros')(x)
        
        print("out_mem.shape, out_captions.shape", out_mem.shape, out_captions.shape)
        
        # Build the full model by creating a new one that takes the input we created and connects it to the out we defined above
        if return_states:
            outputs = [out_mem, out_captions, states]
            inputs = [in_video, in_captions, initial_state]
        else:
            outputs = [out_mem, out_captions]
            inputs = [in_video, in_captions]
        full_model = Model(inputs=inputs, outputs=outputs, name='i3d_captions')
        if verbose:
            full_model.summary()

        if ckpt is not None:
            full_model.load_weights(ckpt)

        return full_model

    return model_func




def build_i3d_sentence_emb(pool_video_repr=True):
    def build_model(ckpt=cfg._INIT_CKPT,
                         type=cfg._TYPE,
                         image_size=cfg._IMAGE_SIZE,
                         num_frames=cfg._NUM_FRAMES,
                         num_classes=cfg._NUM_CLASSES,
                         use_pretrained=True,
                         verbose=True,
                         download_weights=True,
                         downsample_factor=None,
                         dropout_prob=cfg._DROPOUT_PROB,
                         run_locally=cfg._RUN_LOCALLY,
                         final_activation='softmax',
                         show_internal_summary=False,
                         rescale=True,
                         embedding_matrix=None):

        weights=None
        if type == 'rgb':
            if use_pretrained and ckpt is None:
                weights = 'rgb_imagenet_and_kinetics'
            channels = 3
            resc_func = lambda x: x/255.0 *2 - 1
        elif type =='flow':
            if use_pretrained and ckpt is None:
                weights = 'flow_imagenet_and_kinetics'
            channels = 2
            resc_func = lambda x: K.clip(x,-20.0,20.0)/20.0
        else:
            raise ValueError('unknown network type')


        # We build an i3d for Kinetics (400 out classes) to correctly load the pretrained weights
        rgb_model = Inception_Inflated3d(include_top=True,
                                        weights=weights,
                                        input_shape=(num_frames, image_size, image_size, channels),
                                        download_weights=download_weights,
                                        dropout_prob=dropout_prob,
                                        downsample_factor=downsample_factor,
                                        classes=400)

        if show_internal_summary: rgb_model.summary()

        # Removing layers that we're not going to use
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.outputs = [rgb_model.layers[-1].output]
        rgb_model.layers[-1].outbound_nodes = []


        # We define a new input layer
        in_layer = Input(shape=(num_frames, image_size, image_size, channels))

        # Adding rescaling layer
        if rescale:
            resc = Lambda(resc_func) (in_layer)  #tf.math.multiply(tf.math.divide(x,255.0),2.0)-1.0) (in_layer)

        video_repr = rgb_model(resc)
        if verbose:
            print('video_repr shape before last conv',video_repr.shape)

        # Connect the input to the last blocks of the net, which starts with one conv3d
        x = conv3d_bn(video_repr, num_classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False,
                use_bn=False, name='Conv3d_6a_1x1')

        if verbose:
            print('x.shape after last conv',x.shape)

        # then reshape to (n_frames_remaining, classes) just to get rid of the two 1,1 dummy dimensions
        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, num_classes))(x)

        # then mean over frames, which gives us the logits (that we have to average ith the flow logits in case of having a flow stream)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        # Finally the softmax to have a proba dist over the classes
        out = Activation(final_activation, name='prediction')(x)


        ## OUTPUT FOR THE SENTENCE EMBEDDING
        if pool_video_repr:
            x = Reshape((int(video_repr.shape[1]), int(video_repr.shape[-1])))(video_repr)
            x = Lambda(lambda t: K.mean(t, axis=1, keepdims=True))(x)
        else:
            x = Reshape((int(video_repr.shape[1])*int(video_repr.shape[-1]),1))(video_repr)

        print('tensor shape before entering dense_emb_out',x.shape)
        emb_out = Dense(1024, activation='tanh', name='dense_emb_out')(x)

        # Build the full model by creating a new one that takes the input we created and connects it to the out we defined above
        full_model = Model(in_layer, [out, emb_out], name='i3d_full')
        if verbose:
            full_model.summary()

        if ckpt is not None:
            full_model.load_weights(ckpt)

        return full_model
    return build_model





def captioning_module_td(input, embedding_matrix, initial_state=None,
                            embedding_trainable=False, units=512, output_type='word',
                            return_embedding=False, return_states=False):

    if output_type=='word':
        ret_seq = False
        out_layer = TimeDistributed(Dense(units, activation = 'relu', name='cap_words_dense_out'))
    elif output_type=='sentence':
        ret_seq = True
        out_layer = TimeDistributed(TimeDistributed(Dense(units, activation = 'relu', name='cap_words_dense_out')))



    # Replicate input to match time dimention of initial_state
    input_td = Lambda(lambda x: K.repeat_elements(K.expand_dims(input, axis=1), rep=int(initial_state.shape[1]), axis=1)) (input)

    if type(initial_state)!= list or initial_state.shape[1] != units:
        state_h = TimeDistributed(Dense(units, activation='relu', name='initial_state_dense_h'))(initial_state)
        state_c = TimeDistributed(Dense(units, activation='relu', name='initial_state_dense_c'))(initial_state)
        print("shape state_h in captioning_module_td (should be bs x 8 x units):", state_h)
        initial_state = [state_h, state_c]

    print(embedding_matrix.shape)

    emb = TimeDistributed(Embedding(embedding_matrix.shape[0],
                    embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    trainable=embedding_trainable,
                    mask_zero=True))
    emb_out = emb(input_td)
    x, state_h1, state_c1 = TimeDistributed(LSTM(units, return_sequences=True, name='cap_words_lstm1', return_state=True)) (emb_out, initial_state=initial_state)
    # x, state_h2, state_c2 = LSTM(units, return_sequences=ret_seq, name='cap_words_lstm2', return_state=True)(x, initial_state=initial_state[1])
    output = out_layer(x)

    if return_embedding:
        return output, emb_out
    else:
        return output


def recurrent_module(x, n_time_preds):

    x = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, axis=1), rep=n_time_preds, axis=1))(x)
    x = LSTM(int(x.shape[-1]), return_sequences=True)(x)

    return x

def build_i3d_recurrent_head(pool_video_repr=True, n_time_preds=8, use_captions=False, output_biases=None, timestep_cap_input=0):
    def build_model(ckpt=cfg._INIT_CKPT,
                         type=cfg._TYPE,
                         image_size=cfg._IMAGE_SIZE,
                         num_frames=cfg._NUM_FRAMES,
                         num_classes=cfg._NUM_CLASSES,
                         use_pretrained=True,
                         verbose=True,
                         download_weights=True,
                         downsample_factor=None,
                         dropout_prob=cfg._DROPOUT_PROB,
                         run_locally=cfg._RUN_LOCALLY,
                         final_activation='softmax',
                         show_internal_summary=False,
                         rescale=True,
                         embedding_matrix=None):

        weights=None
        if type == 'rgb':
            if use_pretrained and ckpt is None:
                weights = 'rgb_imagenet_and_kinetics'
            channels = 3
            resc_func = lambda x: x/255.0 *2 - 1
        elif type =='flow':
            if use_pretrained and ckpt is None:
                weights = 'flow_imagenet_and_kinetics'
            channels = 2
            resc_func = lambda x: K.clip(x,-20.0,20.0)/20.0
        else:
            raise ValueError('unknown network type')


        # We build an i3d for Kinetics (400 out classes) to correctly load the pretrained weights
        rgb_model = Inception_Inflated3d(include_top=True,
                                        weights=weights,
                                        input_shape=(num_frames, image_size, image_size, channels),
                                        download_weights=download_weights,
                                        dropout_prob=dropout_prob,
                                        downsample_factor=downsample_factor,
                                        classes=400)

        if show_internal_summary: rgb_model.summary()

        # Removing layers that we're not going to use
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.layers.pop()
        rgb_model.outputs = [rgb_model.layers[-1].output]
        rgb_model.layers[-1].outbound_nodes = []


        # We define a new input layer
        in_layer = Input(shape=(num_frames, image_size, image_size, channels))

        # Adding rescaling layer
        if rescale:
            resc = Lambda(resc_func) (in_layer)  #tf.math.multiply(tf.math.divide(x,255.0),2.0)-1.0) (in_layer)

        video_repr = rgb_model(resc)
        if verbose:
            print('video_repr shape before last conv',video_repr.shape)

        if pool_video_repr:
            video_repr = Reshape((int(video_repr.shape[1]), int(video_repr.shape[-1])))(video_repr)
            video_repr = Lambda(lambda t: K.mean(t, axis=1, keepdims=False))(video_repr)
        else:
            video_repr = Reshape((int(video_repr.shape[1])*int(video_repr.shape[-1]),1))(video_repr)

        recurrent_feature_out = recurrent_module(video_repr, n_time_preds=n_time_preds)

        recurrent_mem_out = TimeDistributed(Dense(1, activation='relu'))(recurrent_feature_out)

        if use_captions:
            in_captions = Input(shape=(None,),name='cap_input')
            cap_initial_state = Lambda(lambda x: x[:,timestep_cap_input,:], name='rec_output_selection')(recurrent_feature_out)
            caption_repr = captioning_module(in_captions, embedding_matrix, output_type='sentence', initial_state=cap_initial_state, return_states=False)
            out_captions = TimeDistributed(Dense(embedding_matrix.shape[0],
                                    activation = 'softmax',
                                    name = 'output_captions',
                                    bias_initializer=Constant(output_biases) if output_biases is not None else 'zeros',
                                    ), name='td_output_captions')(caption_repr)

            input = [in_layer, in_captions]
            output = [recurrent_mem_out, out_captions]
        else:
            input = in_layer
            output = recurrent_mem_out
        full_model = Model(input, output, name='i3d_full')
        if verbose:
            full_model.summary()

        if ckpt is not None:
            full_model.load_weights(ckpt)

        return full_model
    return build_model




def build_dilated_2d3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None, run_locally=cfg._RUN_LOCALLY,
                    embedding_matrix=None):
    if type == 'rgb':
        channels = 3
    elif type =='flow':
        channels = 2
    else:
        raise ValueError('unknown network type')

    model = dilated_2d3d(include_top=True,
                    input_shape=(num_frames, image_size, image_size, channels),
                    downsample_factor=downsample_factor,
                    dropout_prob = 0.2,
                    use_inception=True,
                    classes= num_classes)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_dilated_i3d_pretrained(ckpt=cfg._INIT_CKPT,
                                type=cfg._TYPE,
                                image_size=cfg._IMAGE_SIZE,
                                num_frames=cfg._NUM_FRAMES,
                                num_classes=cfg._NUM_CLASSES,
                                use_pretrained=cfg._USE_PRETRAINED,
                                verbose=cfg._VERBOSE,
                                download_weights=cfg._DL_WEIGHTS,
                                dropout_prob=cfg._DROPOUT_PROB,
                                downsample_factor=cfg._DOWNSAMPLE_F,
                                freeze_old=cfg._FREEZE_OLD,
                                internal_i3d_ckpt=cfg._INTERNAL_I3D_CKPT_COSMOS,
                                run_locally = cfg._RUN_LOCALLY,
                                embedding_matrix=None):


    internal_num_classes = 51

    if run_locally:
        internal_i3d_ckpt = cfg._INTERNAL_I3D_CKPT_LOCAL
    else:
        internal_i3d_ckpt = cfg._INTERNAL_I3D_CKPT_COSMOS

    model  = build_i3d_raw(ckpt=internal_i3d_ckpt, type=type, image_size=image_size,
                        num_frames=num_frames, num_classes=internal_num_classes,
                        use_pretrained=use_pretrained, verbose=verbose, download_weights=download_weights,
                        dropout_prob=dropout_prob,downsample_factor=downsample_factor,
                        embedding_matrix=None)

    # print(model.layers[1].layers[1].name, model.layers[1].layers[1].get_weights())

    # We dissassemble the layers of i3d to insert the dilated layers
    layers = [l for l in model.layers[1].layers]

    print(layers[:5])

    for l in layers:
        if freeze_old:
            l.trainable = False

    if not downsample_factor :
        pdl = 3
        mp2_idx = 12
    else:
        pdl = 4
        mp2_idx = 13


    # We add a dilation block between the first 7x7x7 conv and the first maxpool
    x = dilation_block(layers[pdl].output, dil_dim=96, output_dim=64,name='dil_block1')

    # We stack back the layers from the first maxpool to the end of the second conv (1x1x1, 64f)
    for i in range(pdl+1, pdl+9):
        print('Stacking',layers[i].name)
        x = layers[i](x)

    # We add dilation after the maxpool that comes after the original 3x3x3 192 conv3d
    x = dilation_block(x, dil_dim=192, output_dim=192,name='dil_block2')

    # We now stack all the following layers, making sure that we
    # treat the inception modules correctly
    first_1, first_2, first_3, first_0 = [ True ] * 4
    for i in range(mp2_idx, len(layers)):
        # print('Processing layer:',layers[i].name)

        if 'Mixed' in layers[i].name:
            # print('Mixed layer, concatenating branches')
            x = keras.layers.concatenate(
                [branch_0, branch_1, branch_2, branch_3],
                axis=-1,
                name=layers[i].name)
            del branch_0, branch_1, branch_2, branch_3
            first_1, first_2, first_3, first_0 = [ True ] * 4

        else:
            info = layers[i].name.split('_')
            if len(info)<=3:
                # print('Maxpool or avg_pool layer, adding sequentially')
                # This is a maxpooling layer or the final global_avg_pool
                x = layers[i](x)
            else:
                if len(info[2]) == 3:
                    # print('Solitary convs, adding sequentially')
                    x = layers[i](x)
                else:
                    # print('Part of inception module, adding to corresp branch')
                    if info[2] == '0a':
                        if first_0:
                            branch_0 = layers[i](x)
                            first_0=False
                        else:
                            branch_0 = layers[i](branch_0)
                    elif info[2] == '1a':
                        if first_1:
                            branch_1 = layers[i](x)
                            first_1=False
                        else:
                            branch_1 = layers[i](branch_1)
                    elif info[2] == '1b':
                        branch_1 = layers[i](branch_1)
                    elif info[2] == '2a':
                        if first_2:
                            branch_2 = layers[i](x)
                            first_2=False
                        else:
                            branch_2 = layers[i](branch_2)
                    elif info[2] == '2b':
                        branch_2 = layers[i](branch_2)
                    elif info[2] == '3a':
                        if first_3:
                            branch_3 = layers[i](x)
                            first_3=False
                        else:
                            branch_3 = layers[i](branch_3)
                    elif info[2] == '3b':
                        branch_3 = layers[i](branch_3)

    print('Reconstruction of i3d finished')

    # We add the final layers (dropout+conv+softmax)
    x = Dropout(dropout_prob, name='final_dropout')(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    # print('x.shape after last conv:', x.shape)

    # Reshaping
    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, num_classes),name='final_reshape')(x)

    # Logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]), name='mean_time_dim')(x)

    out = Activation('softmax', name='prediction')(x)

    # Build the full model by creating a new one that takes the input we created and connects it to the out we defined above
    full_model = Model(model.layers[1].layers[0].input, out, name='dilated_i3d_full')
    if verbose:
        print('\n\nFULL MODEL SUMMARY')
        full_model.summary()

    if ckpt is not None:
        full_model.load_weights(ckpt)

    # print(full_model.layers[1].name, full_model.layers[1].get_weights())
    return full_model



def dilation_block(x, dil_dim=128, output_dim=128, channel_axis=-1, name='dil_block', use_bn=True):
    dil1 = conv3d_bn(x, dil_dim, 3,3,3, strides=(1,1,1), dilation_rate=(1,1,1),
                    use_bn=use_bn, padding='same', name=name+'_dil1_3x3')
    dil2 = conv3d_bn(x, dil_dim, 3,3,3, strides=(1,1,1), dilation_rate=(2,1,1),
                    use_bn=use_bn, padding='same', name=name+'_dil2_3x3')
    dil3 = conv3d_bn(x, dil_dim, 3,3,3, strides=(1,1,1), dilation_rate=(3,1,1),
                    use_bn=use_bn, padding='same', name=name+'_dil3_3x3')
    x = layers.concatenate([dil1,dil2,dil3],axis=channel_axis, name=name+'_concat')
    # We add a 1x1x1 conv for dimension matching, here to 192 filters
    x = conv3d_bn(x, output_dim, 1,1,1, use_bn=use_bn, padding='same', name=name+'_conv3d_dim_match')

    return x

def build_tcn(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None, run_locally=cfg._RUN_LOCALLY,
                    dropout_prob=cfg._DROPOUT_PROB,
                    embedding_matrix=None):

    from tcn import dilated_tcn
    from classification_models import ResNet18

    if type=='rgb':
        channels = 3
    elif type=='flow':
        channels = 2

    input = Input((num_frames, image_size, image_size, channels))

    x = downsample_block(input, downsample_factor)

    if downsample_factor:
        div = downsample_factor
    else:
        div =1



    if run_locally:
        resnet = keras.models.load_model('../ckpt/resnet_base/resnet18_imagenet_1000.h5')
    else:
        # resnet = keras.models.load_model('../ckpt/resnet_base/resnet18_imagenet_1000.h5')
        resnet = ResNet18(input_shape=(224,224,3), weights='imagenet', classes=1000)



    resnet.summary()

    x = TimeDistributed(resnet)(x)

    tcn = dilated_tcn(output_slice_index='last',
                        num_feat=1000, #output of resnet
                        num_classes=num_classes,
                        nb_filters=64,
                        kernel_size=8,
                        dilatations=[1, 2, 4, 8],
                        nb_stacks=8,
                        max_len=num_frames,
                        activation='norm_relu')

    x = tcn(x)

    model = Model(inputs=input, outputs=x)

    if ckpt:
        model.load_weights(ckpt)
    return model


def build_dilated_i3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None, run_locally=cfg._RUN_LOCALLY,
                    dropout_prob=cfg._DROPOUT_PROB,
                    embedding_matrix=None):
    if type == 'rgb':
        channels = 3
    elif type =='flow':
        channels = 2
    else:
        raise ValueError('unknown network type')

    model = dilated_i3d(include_top=True,
                    input_shape=(num_frames, image_size, image_size, channels),
                    downsample_factor=downsample_factor,
                    dropout_prob = 0.2,
                    classes= num_classes)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_mict_i3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB,
                    run_locally=cfg._RUN_LOCALLY,
                    embedding_matrix=None):
    if type == 'rgb':
        channels = 3
    elif type =='flow':
        channels = 2
    else:
        raise ValueError('unknown network type')

    model = mict_i3d(include_top=True,
                    use_inception=True,
                    downsample_factor=downsample_factor,
                    input_shape=(num_frames, image_size, image_size, channels),
                    dropout_prob = dropout_prob,
                    classes= num_classes)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_resnet50_lstm(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB, run_locally=cfg._RUN_LOCALLY,
                    embedding_matrix=None):
    if type == 'rgb':
        channels = 3
    elif type =='flow':
        channels = 2
    else:
        raise ValueError('unknown network type')

    model = resnet_lstm_model(resnet_layers=50,
                        input_shape=(num_frames, image_size, image_size, channels),
                        lstm_layers=1,
                        verbose=1,
                        num_classes=num_classes,
                        lstm_dim =512,
                        dropout_prob=dropout_prob,
                        downsample_factor=downsample_factor,
                        run_locally=run_locally)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_resnet18_lstm(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB, run_locally=cfg._RUN_LOCALLY):
    if type == 'rgb':
        channels = 3
    elif type =='flow':
        channels = 2
    else:
        raise ValueError('unknown network type')

    model = resnet_lstm_model(resnet_layers=18,
                        input_shape=(num_frames, image_size, image_size, channels),
                        lstm_layers=1,
                        verbose=1,
                        num_classes=num_classes,
                        lstm_dim =512,
                        dropout_prob=dropout_prob,
                        downsample_factor=downsample_factor,
                        run_locally=run_locally)

    if ckpt:
        model.load_weights(ckpt)

    return model

def _obtain_input_shape(input_shape,
                        default_frame_size,
                        min_frame_size,
                        default_num_frames,
                        min_num_frames,
                        data_format,
                        require_flatten,
                        weights=None):
    """Internal utility to compute/validate the model's input shape.
    (Adapted from `keras/applications/imagenet_utils.py`)

    # Arguments
        input_shape: either None (will return the default network input shape),
            or a user-provided shape to be validated.
        default_frame_size: default input frames(images) width/height for the model.
        min_frame_size: minimum input frames(images) width/height accepted by the model.
        default_num_frames: default input number of frames(images) for the model.
        min_num_frames: minimum input number of frames accepted by the model.
        data_format: image data format to use.
        require_flatten: whether the model is expected to
            be linked to a classifier via a Flatten layer.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
            If weights='kinetics_only' or weights=='imagenet_and_kinetics' then
            input channels must be equal to 3.

    # Returns
        An integer shape tuple (may include None entries).

    # Raises
        ValueError: in case of invalid argument values.
    """
    if weights != 'kinetics_only' and weights != 'imagenet_and_kinetics' and input_shape and len(input_shape) == 4:
        if data_format == 'channels_first':
            if input_shape[0] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[0]) + ' input channels.')
            default_shape = (input_shape[0], default_num_frames, default_frame_size, default_frame_size)
        else:
            if input_shape[-1] not in {1, 3}:
                warnings.warn(
                    'This model usually expects 1 or 3 input channels. '
                    'However, it was passed an input_shape with ' +
                    str(input_shape[-1]) + ' input channels.')
            default_shape = (default_num_frames, default_frame_size, default_frame_size, input_shape[-1])
    else:
        if data_format == 'channels_first':
            default_shape = (3, default_num_frames, default_frame_size, default_frame_size)
        else:
            default_shape = (default_num_frames, default_frame_size, default_frame_size, 3)
    if (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics') and require_flatten:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True` '
                                 'and loading `imagenet` weights, '
                                 '`input_shape` should be ' +
                                 str(default_shape) + '.')
        return default_shape

    if input_shape:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[0] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[1] is not None and input_shape[1] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[2] is not None and input_shape[2] < min_frame_size) or
                   (input_shape[3] is not None and input_shape[3] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
        else:
            if input_shape is not None:
                if len(input_shape) != 4:
                    raise ValueError(
                        '`input_shape` must be a tuple of four integers.')
                if input_shape[-1] != 3 and (weights == 'kinetics_only' or weights == 'imagenet_and_kinetics'):
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if input_shape[0] is not None and input_shape[0] < min_num_frames:
                    raise ValueError('Input number of frames must be at least ' +
                                     str(min_num_frames) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')

                if ((input_shape[1] is not None and input_shape[1] < min_frame_size) or
                   (input_shape[2] is not None and input_shape[2] < min_frame_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_frame_size) + 'x' + str(min_frame_size) + '; got '
                                     '`input_shape=' + str(input_shape) + '`')
    else:
        if require_flatten:
            input_shape = default_shape
        else:
            if data_format == 'channels_first':
                input_shape = (3, None, None, None)
            else:
                input_shape = (None, None, None, 3)
    if require_flatten:
        if None in input_shape:
            raise ValueError('If `include_top` is True, '
                             'you should specify a static `input_shape`. '
                             'Got `input_shape=' + str(input_shape) + '`')
    return input_shape


def act_and_bn(x, use_activation_fn, use_bn, act_before_bn, act_name=None, bn_name=None):
    if act_before_bn and use_activation_fn:
            x = Activation('relu', name=act_name)(x)

    if use_bn:
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = -1
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    if not act_before_bn and use_activation_fn:
            x = Activation('relu', name=act_name)(x)
    return x


def conv3d_bn(x,
              filters,
              num_frames,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1, 1),
              dilation_rate=(1, 1, 1),
              use_bias = False,
              use_activation_fn = True,
              use_bn = True,
              name=None,
              w_decay=None,
              act_before_bn=False):

    """Utility function to apply conv3d + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        bn_name = None
        conv_name = None
        act_name = None

    x = Conv3D(filters,
            (num_frames, num_row, num_col),
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            use_bias=use_bias,
            kernel_initializer='he_normal',
            kernel_regularizer=l2(w_decay) if w_decay else None,
            bias_regularizer=l2(w_decay) if w_decay else None,
            name=conv_name)(x)

    x = act_and_bn(x, use_activation_fn, use_bn, act_before_bn, act_name=act_name, bn_name=bn_name)

    # if act_before_bn and use_activation_fn:
    #         x = Activation('relu', name=name)(x)
    #
    # if use_bn:
    #     if K.image_data_format() == 'channels_first':
    #         bn_axis = 1
    #     else:
    #         bn_axis = -1
    #     x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    #
    # if not act_before_bn and use_activation_fn:
    #         x = Activation('relu', name=name)(x)

    return x



def conv2d_bn(x, filters,
            kernel_size,
            strides=(1, 1),
            dilation_rate=(1, 1),
            use_bias=True, # We could change this if too many parameters or not learning fast enough
            use_activation_fn = True,
            use_bn = True,
            padding='same',
            name=None,
            w_decay=None,
            act_before_bn=False):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        bn_name = None
        conv_name = None
        act_name = None

    x = Conv2D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
                use_bias=use_bias,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(w_decay) if w_decay else None,
                bias_regularizer=l2(w_decay) if w_decay else None,
                name=conv_name)(input)

    x = act_and_bn(x, use_activation_fn, use_bn, act_before_bn, act_name=act_name, bn_name=bn_name)

    # if use_activation_fn:
    #     x = Activation('relu', name=act_name)(x)
    #
    # if use_bn:
    #     if K.image_data_format() == 'channels_first':
    #         bn_axis = 1
    #     else:
    #         bn_axis = -1
    #     x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)

    return x

def downsample_block(input, downsample_factor):
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(input)

    else:
        x = input

    return x

def resnet_lstm_model(resnet_layers=18,
                    input_shape=(64,224,224,3),
                    lstm_layers=1,
                    verbose=1,
                    num_classes=5,
                    lstm_dim =512,
                    dropout_prob=cfg._DROPOUT_PROB,
                    downsample_factor=2,
                    run_locally=False):
    '''Generates a cnn+lstm model for action recognition with a resnet as base
    followed by 'lstm_layers' lstm layers. If a resnet hdf5 path is given, the
    resnet is loaded from there.'''

    input = Input(shape=input_shape)

    # Downsample temporally
    x = downsample_block(input, downsample_factor)
    # if downsample_factor:
    #     if K.image_data_format() == 'channels_first':
    #         x = Lambda(lambda t: t[:,:,::downsample_factor,...],
    #                    output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(input)
    #     else:
    #         x = Lambda(lambda t: t[:,::downsample_factor,...],
    #                    output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(input)
    #
    # else:
    #     x = input


    # currently no resnet layer modification is available
    if resnet_layers == 50:
        resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
    elif resnet_layers == 18:
        if run_locally:
            resnet = keras.models.load_model('../../../nfs_share/ckpt/resnet_base/resnet18.h5')
        else:
            resnet = keras.models.load_model('../../../ckpt/resnet_base/resnet18.h5')
    else:
        raise ValueError('Number of layers not supported')

    x = TimeDistributed(resnet)(x)

    if verbose:
        print('x.shape after timedist',x.shape)

    for i in range(lstm_layers):
        x = LSTM(lstm_dim, return_sequences=False, dropout=dropout_prob)(x)

    x = Dense(512)(x)
    x = act_and_bn(x, use_activation_fn=False, use_bn=True,
                    act_before_bn=True, act_name='act', bn_name='bn')
    x = Dropout(dropout_prob)(x)

    out = Dense(num_classes, activation='softmax')(x)
    # create model
    model = Model(input, out, name='resnet_lstm')

    return model

def se_resnet_lstm_model():
    pass



def MiCT_net_small(n_classes=51):

    input = Input(shape=(64, 224, 224, 3))

    x = mict_block_plain(input, name='mict1')
    x = mict_block_plain(x, filters_branch=128, name='mict2')
    print('shape after mict',x.shape)

    x = AveragePooling3D((2, 14, 14), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
    x = Dropout(0.5)(x)

    print('shape after avg',x.shape)

    x = conv3d_bn(x, n_classes, 1, 1, 1, padding='same',
            use_bias=True, use_activation_fn=False, use_bn=False,
            name='Conv3d_6a_1x1', act_before_bn=True)
    print('shape after conv3dbn',x.shape)


    num_frames_remaining = int(x.shape[1])
    x = Reshape((num_frames_remaining, n_classes))(x)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)

    x = Activation('softmax', name='prediction')(x)
    print('shape after act',x.shape)

    # create model
    model = Model(input, x, name='mict_test')

    return model

def MiCT_net(n_classes=51, dropout_prob=0.5, downsample_factor=2):

    # Input
    input = Input(shape=(64, 224, 224, 3))

    # Downsample temporally
    # Downsample temporally
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(img_input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(img_input)

    else:
        x = img_input

    # print('input.shape after downs',input.shape)
    # print('x.shape after downs',x.shape)


    # MiCT block 1
    x = mict_block_plain(x, filters_branch=64, filters_post=[64, 192], kernel_sizes_post=[1,3], name='mict1')
    x = Dropout(dropout_prob)(x)

    # MiCT block 2
    x = mict_block_inception(x, filters_branch=256, filters_post=[320, 576], name='mict2')
    x = Dropout(dropout_prob)(x)

    # MiCT block 3
    x = mict_block_inception(x, filters_branch=576, filters_post=[576, 608, 609, 1056], name='mict3')
    x = Dropout(dropout_prob)(x)

    # MiCT block 4
    x = mict_block_inception(x, filters_branch=1024, filters_post=[1024], name='mict4')
    x = Dropout(dropout_prob)(x)

    spatial_size = x.shape[2]
    x = AveragePooling3D((1, spatial_size, spatial_size), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

    print('shape after AvgPool3D',x.shape)
    # x = Reshape((int(x.shape[1]*x.shape[-1]),))(x)
    # print('shape after resha',x.shape, 'num_classes:', n_classes)
    # x = Flatten()(x)
    # print('shape after flatten',x.shape, 'num_classes:', n_classes)
    x = Dense(n_classes)(x)

    print('shape after dense',x.shape)

    x = Reshape((int(x.shape[1]), n_classes))(x)
    print('shape after Reshape',x.shape)

    # logits (raw scores for each class)
    x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
               output_shape=lambda s: (s[0], s[2]))(x)


    x = Activation('softmax', name='prediction')(x)

    # create model
    model = Model(input, x, name='mict_test')

    return model


def mict_i3d(include_top=True,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                downsample_factor=None,
                classes=400,
                use_inception=True,
                use_bn=True,
                filter_factor_br=0.7,
                filter_factor=0.7,
                k=1.3):
    '''Architecture for action recognition with dilated inputs. Based on i3d and inception v1,
    This architecture replaces i3d's 3d inception modules with mixed 2d/3d convolutional tubes.
    It also adds time-dilated 3d-convolutions as an input step.'''

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor



    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Downsample temporally
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(img_input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(img_input)

    else:
        x = img_input


    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(x, 64, 7, 7, 7, strides=(2, 2, 2), use_bn=use_bn, padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)

    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), use_bn=use_bn, padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # print('x.shape after downsa:', x.shape)

    # Mixed 3b
    x = mict_block_inception(x,
                                output_dim_2d=int(256*filter_factor_br),
                                output_dim_3d=int(256*filter_factor_br),
                                output_dim_inc2dpost=[int(256*filter_factor),int(256*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_3b',
                                act_before_bn=True)

    # Mixed 3c
    x = mict_block_inception(x,
                                output_dim_2d=int(480*filter_factor_br),
                                output_dim_3d=int(480*filter_factor_br),
                                output_dim_inc2dpost=[int(480*filter_factor),int(480*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_3c',
                                act_before_bn=True)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor_br),
                                output_dim_3d=int(512*filter_factor_br),
                                output_dim_inc2dpost=[int(512*filter_factor),int(512*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4b',
                                act_before_bn=True)

    # Mixed 4c
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor_br),
                                output_dim_3d=int(512*filter_factor_br),
                                output_dim_inc2dpost=[int(512*filter_factor),int(512*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4c',
                                act_before_bn=True)

    # Mixed 4d
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor_br),
                                output_dim_3d=int(512*filter_factor_br),
                                output_dim_inc2dpost=[int(512*filter_factor),int(512*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4d',
                                act_before_bn=True)

    # Mixed 4e
    x = mict_block_inception(x,
                                output_dim_2d=int(528*filter_factor_br),
                                output_dim_3d=int(528*filter_factor_br),
                                output_dim_inc2dpost=[int(528*filter_factor),int(528*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4e',
                                act_before_bn=True)

    # Mixed 4f
    x = mict_block_inception(x,
                                output_dim_2d=int(832*filter_factor_br),
                                output_dim_3d=int(832*filter_factor_br),
                                output_dim_inc2dpost=[int(832*filter_factor),int(832*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4f',
                                act_before_bn=True)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    x = mict_block_inception(x,
                                output_dim_2d=int(832*filter_factor_br),
                                output_dim_3d=int(832*filter_factor_br),
                                output_dim_inc2dpost=[int(832*filter_factor),int(832*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_5b',
                                act_before_bn=True)

    # Mixed 5c
    x = mict_block_inception(x,
                                output_dim_2d=int(1024*filter_factor_br),
                                output_dim_3d=int(1024*filter_factor_br),
                                output_dim_inc2dpost=[int(1024*filter_factor),int(1024*k*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_5c',
                                act_before_bn=True)

    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        print('x.shape after last convolution', x.shape)

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

    inputs = img_input
    # create model
    model = Model(inputs, x, name='dilated_2d3d')

    return model

def dilated_i3d(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                download_weights=True,
                downsample_factor=None,
                use_bn=True,
                classes=400):
    """Instantiates the time-dilated Inflated 3d inception v1 architecture.

    Optionally loads weights pre-trained
    on Kinetics. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input frame(image) size for this model is 224x224.

    # Arguments
        include_top: whether to include the the classification
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset only).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)` (with `channels_last` data format)
            or `(NUM_FRAMES, 3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
            NUM_FRAMES should be no smaller than 8. The authors used 64
            frames per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer.
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in WEIGHTS_NAME or weights is None or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or %s' %
                         str(WEIGHTS_NAME) + ' '
                         'or a valid path to a file containing `weights` values')

    if weights in WEIGHTS_NAME and include_top and classes != 400:
        raise ValueError('If using `weights` as one of these %s, with `include_top`'
                         ' as true, `classes` should be 400' % str(WEIGHTS_NAME))

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    # Downsample temporally
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(img_input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(img_input)

    else:
        x = img_input



    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

    # Dilation block
    # TODO: we could change these blocks to MiCT or Inception3D blocks
    dil1 = conv3d_bn(x, 64, 3,3,3, strides=(1,1,1), dilation_rate=(1,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil1_3x3')
    dil2 = conv3d_bn(x, 64, 3,3,3, strides=(1,1,1), dilation_rate=(2,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2_3x3')
    dil3 = conv3d_bn(x, 64, 3,3,3, strides=(1,1,1), dilation_rate=(3,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil3_3x3')
    # dil4 = conv3d_bn(x, 32, 3,3,3, strides=(1,1,1), dilation_rate=(4,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil4_3x3')
    x = layers.concatenate([dil1,dil2,dil3],axis=channel_axis, name='concat_dilation')


    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)

    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), use_bn=use_bn, padding='same', name='Conv3d_2b_1x1')

    dil1 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(1,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.1_3x3')
    dil2 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(2,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.2_3x3')
    dil3 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(3,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.3_3x3')
    x = layers.concatenate([dil1,dil2,dil3],axis=channel_axis, name='concat_dilation2')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)


    # print('x.shape after downsa:', x.shape)


    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')



    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        print('x.shape after last conv:', x.shape)

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

    inputs = img_input
    # create model
    model = Model(inputs, x, name='dilated_i3d')

    return model

def dilated_2d3d(include_top=True,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                downsample_factor=2,
                classes=400,
                use_inception=True,
                use_bn=True,
                filter_factor=0.6):

    '''Architecture for action recognition with dilated inputs. Based on i3d and inception v1,
    This architecture replaces i3d's 3d inception modules with mixed 2d/3d convolutional tubes.
    It also adds time-dilated 3d-convolutions as an input step.'''

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor



    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Downsample temporally
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(img_input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(img_input)

    else:
        x = img_input


    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(x, 64, 7, 7, 7, strides=(2, 2, 2), use_bn=use_bn, padding='same', name='Conv3d_1a_7x7')

    # Dilation block
    # TODO: we could change these blocks to MiCT or Inception3D blocks
    dil1 = conv3d_bn(x, 32, 3,3,3, strides=(1,1,1), dilation_rate=(1,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil1_3x3')
    dil2 = conv3d_bn(x, 32, 3,3,3, strides=(1,1,1), dilation_rate=(2,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2_3x3')
    dil3 = conv3d_bn(x, 32, 3,3,3, strides=(1,1,1), dilation_rate=(3,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil3_3x3')
    # dil4 = conv3d_bn(x, 32, 3,3,3, strides=(1,1,1), dilation_rate=(4,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil4_3x3')
    x = layers.concatenate([dil1,dil2,dil3],axis=channel_axis, name='concat_dilation')


    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)

    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), use_bn=use_bn, padding='same', name='Conv3d_2b_1x1')

    dil1 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(1,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.1_3x3')
    dil2 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(2,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.2_3x3')
    dil3 = conv3d_bn(x, 128, 3,3,3, strides=(1,1,1), dilation_rate=(3,1,1), use_bn=use_bn, padding='same', name='Conv3d_dil2.3_3x3')
    x = layers.concatenate([dil1,dil2,dil3],axis=channel_axis, name='concat_dilation2')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    x = mict_block_inception(x,
                                output_dim_2d=int(256*filter_factor),
                                output_dim_3d=int(256*filter_factor),
                                output_dim_inc2dpost=[int(256*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_3b',
                                act_before_bn=True)

    # Mixed 3c
    x = mict_block_inception(x,
                                output_dim_2d=int(480*filter_factor),
                                output_dim_3d=int(480*filter_factor),
                                output_dim_inc2dpost=[int(480*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_3c',
                                act_before_bn=True)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor),
                                output_dim_3d=int(512*filter_factor),
                                output_dim_inc2dpost=[int(512*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4b',
                                act_before_bn=True)

    # Mixed 4c
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor),
                                output_dim_3d=int(512*filter_factor),
                                output_dim_inc2dpost=[int(512*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4c',
                                act_before_bn=True)

    # Mixed 4d
    x = mict_block_inception(x,
                                output_dim_2d=int(512*filter_factor),
                                output_dim_3d=int(512*filter_factor),
                                output_dim_inc2dpost=[int(512*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4d',
                                act_before_bn=True)

    # Mixed 4e
    x = mict_block_inception(x,
                                output_dim_2d=int(528*filter_factor),
                                output_dim_3d=int(528*filter_factor),
                                output_dim_inc2dpost=[int(528*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4e',
                                act_before_bn=True)

    # Mixed 4f
    x = mict_block_inception(x,
                                output_dim_2d=int(832*filter_factor),
                                output_dim_3d=int(832*filter_factor),
                                output_dim_inc2dpost=[int(832*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_4f',
                                act_before_bn=True)

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    x = mict_block_inception(x,
                                output_dim_2d=int(832*filter_factor),
                                output_dim_3d=int(832*filter_factor),
                                output_dim_inc2dpost=[int(832*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_5b',
                                act_before_bn=True)

    # Mixed 5c
    x = mict_block_inception(x,
                                output_dim_2d=int(1024*filter_factor),
                                output_dim_3d=int(1024*filter_factor),
                                output_dim_inc2dpost=[int(1024*filter_factor)],
                                use_inception=use_inception,
                                use_bn=use_bn,
                                name='mict_5c',
                                act_before_bn=True)

    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        print('x.shape after last convolution', x.shape)

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)

    inputs = img_input
    # create model
    model = Model(inputs, x, name='dilated_2d3d')

    return model


def build_resnet18_3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB, run_locally=cfg._RUN_LOCALLY):

    if type=='rgb':
        channels = 3
    elif type=='flow':
        channels = 2

    input = Input((num_frames, image_size, image_size, channels))

    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(input)

        div = downsample_factor
    else:
        x = input
        div = 1

    resnet = Resnet3DBuilder.build_resnet_18(input_shape=(num_frames//div, image_size, image_size, channels), num_outputs=num_classes, reg_factor=cfg._REG_FACTOR)

    resnet.summary()

    out = resnet(x)

    model = Model(inputs=input, outputs=out)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_resnet50_3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB, run_locally=cfg._RUN_LOCALLY):

    if type=='rgb':
        channels = 3
    elif type=='flow':
        channels = 2

    input = Input((num_frames, image_size, image_size, channels))

    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]))(input)
        else:
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]))(input)

        div = downsample_factor
    else:
        x = input
        div = 1

    resnet = Resnet3DBuilder.build_resnet_50(input_shape=(num_frames//div, image_size, image_size, channels), num_outputs=num_classes, reg_factor=cfg._REG_FACTOR)

    resnet.summary()

    out = resnet(x)

    model = Model(inputs=input, outputs=out)

    if ckpt:
        model.load_weights(ckpt)

    return model

def build_roi_i3d(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES, verbose=True,
                    download_weights=True, downsample_factor=None,
                    dropout_prob=cfg._DROPOUT_PROB, run_locally=cfg._RUN_LOCALLY,
                    use_ground_truth_rois=True):
    '''End-to-end trainable resnet+i3d, where the i3d takes as input RoIs pooled rom the 6th layer of the resnet.
    The resnet should be a people detector network, and the person detections should be used as input here to extract the correct RoIs.
    Currently works for one RoI, corresponding to one person performing an action.
    The network takes as input the video to process + a Fx5x1 vector indicating, for each one of the F frames,
    the bounding box to extract in the format [channel, top, left, bottom, right]
    '''

    # CONSTANTS
    pool_size = 4
    num_rois = 1
    channels = 3

    det_input_h = 544
    det_input_w = 960
    downsample_factor =8
    batch_size=4

    image_h = 240
    image_w = 320

    # Detector needs 0-1 inputs
    if run_locally:
        detector = keras.models.load_model(cfg._DETECTOR_CKPT_LOCAL, compile=False)
    else:
        detector = keras.models.load_model(cfg._DETECTOR_CKPT_COSMOS,compile=False)

    # detector.layers.pop()
    # detector.layers.pop()

    # input1 = Input((channels, image_h, image_w))
    # out = detector(input1)
    # new_detector = Model(input=input1, output = out)
    detector.summary()


    video_input = Input((num_frames, det_input_h, det_input_w, channels))
    bbox_input = Input((num_frames, num_rois, 4)) # x,y,w,h

    print('bbox_input.shape',bbox_input.shape)

    # reshaped_input = TimeDistributed(Lambda(lambda x: tf.image.resize_images(x, [det_input_h, det_input_w]),
    #            output_shape=lambda s: (s[0], det_input_h, det_input_w, s[-1])))(video_input)

    reshaped_input = K.permute_dimensions(video_input, (0, 1, 4, 2, 3))

    print('reshaped_input.shape',reshaped_input.shape)

    # The output of this model should be a set of feature maps with shape H_fm, W_fm, n_feature_maps
    fmaps_model = Model(inputs=detector.input, outputs=detector.get_layer('add_3').output)

    # fmaps should have shape num_frames x H_fm x W_fm x n_feature_maps
    fmaps = TimeDistributed(fmaps_model)(reshaped_input)

    fmaps = K.permute_dimensions(fmaps, (0, 1, 3, 4, 2))

    print('fmaps.shape',fmaps.shape)
    n_feature_maps = fmaps.shape[4]


    # THIS HAS TO BE A LAMBDA LAYER
    def reshape_bbox(bbox):

        reshaped_bbox = K.repeat_elements(bbox,fmaps.shape[2], 2)
        reshaped_bbox = K.expand_dims(reshaped_bbox, -2)
        print('reshaped_bbox.shape',reshaped_bbox.shape)
        reshaped_bbox = K.repeat_elements(reshaped_bbox,fmaps.shape[3], -2)
        print('reshaped_bbox.shape',reshaped_bbox.shape)

        # bbox_reshaped = K.placeholder(shape=(fmaps.shape[0], fmaps.shape[1], fmaps.shape[2], fmaps.shape[3], 1))
        # # Reshaping bbox input to merge it with the fmap input. We need this to be able to use TimeDistributed
        # bbox_np_array = np.zeros((batch_size, fmaps.shape[1],fmaps.shape[2], fmaps.shape[3], 1))
        # print(bbox[0, 0, 0, :4])
        # for b in range(batch_size):
        #     for f in range(fmaps.shape[1]):
        #         bbox_np_array[b, f, :4, 0, 0] = bbox[b, f, 0, :4]/downsample_factor
        # K.set_value(bbox_reshaped, bbox_np_array)
        return reshaped_bbox

    print(K.dtype(fmaps[0,0,0,0]))
    # for t in x_unpacked:
    #     # do whatever
    #     result_tensor = t + 1
    #     processed.append(result_tensor)


    # bbox_reshaped = Lambda(reshape_bbox,
    #                 output_shape=lambda s: (s[0], s[1], int(fmaps.shape[2]), int(fmaps.shape[3]), s[3]))(bbox_input) #,output_size

    # print('bbox_resh shape',bbox_reshaped.shape)
    # print('fmaps.shape',fmaps.shape)
    #
    # merged_input = keras.layers.concatenate([fmaps, bbox_reshaped], axis = -1)

    # pool_list =[4] means that the RoiPooling layer will compute one fixed
    # length represnetation of size 4x4 by evenly subdividing the input feature map into a 4x4 grid
    from roi_pooling.roi_pooling_ops import roi_pooling
    def roi_pooling_with_size(tup):
        fmaps, bbox_input = tup
        return roi_pooling(fmaps, bbox_input, pool_size, pool_size)


    fmaps = K.permute_dimensions(fmaps, (1, 0, 2, 3, 4))
    bbox_input = K.permute_dimensions(bbox_input, (1, 0, 2, 3))

    # To run this line, I need to install roi_pooling correctly, but there's currently something wrong with my Ubuntu (it thinks Im running CUDA 7.5) preventing me to install it.
    # This should run fine on a computer that has no trace of CUDA 7.5
    pooled_fmaps = tf.map_fn(roi_pooling_with_size,(fmaps, bbox_input),dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    # pooled_fmaps = TimeDistributed(RoiPoolingConv(pool_size=pool_size, num_rois=num_rois))(merged_input)

    print('pooled_fmaps.shape',pooled_fmaps.shape)

    # Building i3d with feature map input. This model will
    # reshape the pooled feature maps to the i3d required channels with a 1x1 conv.
    i3d_roi = build_roi_i3d_fmap_input(num_frames = num_frames, image_size = pool_size,
                                        input_channels = n_feature_maps)

    # Feeding those images to the i3d with feature maps inputs.
    out = i3d_roi(pooled_fmaps)

    # building the full model.
    model = Model(inputs = [video_input, bbox_input], outputs = out)

    return model

def build_roi_i3d_fmap_input(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                    num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES,
                    use_pretrained=False, verbose=cfg._VERBOSE, download_weights=cfg._DL_WEIGHTS,
                    dropout_prob=cfg._DROPOUT_PROB,downsample_factor=None, run_locally=cfg._RUN_LOCALLY,
                    input_channels=3):

    pooled_fmaps = Input((n_frames, image_size, image_size, input_channels))

    # We use a 1x1 conv to squeeze all the feature maps into 3 channels,
    # to emulate a normal image for i3d.
    fmap_as_img = TimeDistributed(Conv2D(3,kernel_size=1,activation = 'relu'))(pooled_fmaps)

    i3d = build_i3d_raw(ckpt=cfg._INIT_CKPT, type=cfg._TYPE, image_size=cfg._IMAGE_SIZE,
                        num_frames=cfg._NUM_FRAMES, num_classes=cfg._NUM_CLASSES,
                        use_pretrained=False, verbose=cfg._VERBOSE, download_weights=cfg._DL_WEIGHTS,
                        dropout_prob=cfg._DROPOUT_PROB,downsample_factor=None, run_locally=cfg._RUN_LOCALLY)

    out = i3d(fmap_as_img)

    full_model = Model(inputs = pooled_fmaps, outputs= out)

    return full_model



##########################
### Keras video blocks ###
##########################

def mict_block_inception(x,
                        output_dim_2d = 64,
                        output_dim_3d = 64,
                        output_dim_inc2dpost = [64, 192],
                        use_inception=False,
                        name=None,
                        use_bn=True,
                        act_before_bn=True,
                        strides3d = (1,1,1),
                        dilation_rate = (1,1,1),
                        w_decay=None):
    '''Based on Zhou et al, MiCT Paper by Microsoft research:
    https://www.microsoft.com/en-us/research/uploads/prod/2018/05/Zhou_MiCT_Mixed_3D2D_CVPR_2018_paper.pdf

    This block aims to replace traditional 3D convolution blocks by a combination
    of 2D and 3D convolutions that capture spatio-temporal relationships more efficient.
    With this block, networks can be built with less 3D convs but higher accuracy
    than traditional C3D-inspired architectures.

    This particular block uses inception modules instead of traditional 2D convs.
    The 3D blocks can be instantiated as inception modules as well, or as normal 3d blocks.

    '''

    if use_inception:
        branch3d = Inception3DFig5(w_decay=w_decay,
                                    strides = strides3d,
                                    output_dim=output_dim_3d,
                                    use_bn=use_bn,
                                    name=name+'_inc3d',
                                    dilation_rate=dilation_rate,
                                    pooling='avg')(x)
    else:
        branch3d = conv3d_bn(x,
                    output_dim_3d, 3, 3, 3,
                    dilation_rate=dilation_rate,
                    strides=strides3d,
                    use_bn=use_bn,
                    padding='same',
                    name=name+'_conv3d_branch', act_before_bn=act_before_bn)


    # Downsample temporally
    if strides3d[0]>1:
        x = Lambda(lambda x: x[:,::strides3d[0],...],
                   output_shape=lambda s: (s[0], s[1]//strides3d[0], s[2], s[3], s[4]))(x)

    # print('x.shape after ds',x.shape)

    branch2d = TimeDistributedInceptionFig5(w_decay=w_decay, output_dim=output_dim_2d, use_bn=use_bn, name=name+'_inc2d_branch')(x)

    x = Add()([branch2d, branch3d])

    for i in range(len(output_dim_inc2dpost)):
        x = TimeDistributedInceptionFig5(w_decay=w_decay, output_dim=output_dim_inc2dpost[i], use_bn=use_bn, name=name+'_inc2d_'+str(output_dim_inc2dpost[i]))(x)

    # print('shape of tensor at end of mict block named',name,':', x.shape)

    return x




def mict_block_plain(x,
                    filters_branch = 64,
                    filters_post = [64, 192],
                    kernel_sizes_post = [1,3],
                    use_inception=False,
                    use_bn=True,
                    name=None,
                    act_before_bn=True):

    '''
    Based on Zhou et al, MiCT Paper by Microsoft research:
    https://www.microsoft.com/en-us/research/uploads/prod/2018/05/Zhou_MiCT_Mixed_3D2D_CVPR_2018_paper.pdf

    This block aims to replace traditional 3D convolution blocks by a combination
    of 2D and 3D convolutions that capture spatio-temporal relationships more efficient.
    With this block, networks can be built with less 3D convs but higher accuracy
    than traditional C3D-inspired architectures.

    The idea is to first apply 3D conv to a set of frames, while passing each of those frames
    through a 2D conv layer (in parallel), and then summing up the two resulting feature cubes.
    It matches dimensionally, because the dimensions of the feature cube outputed
    by the 3D conv is n_frames x h x c x n_kernels3d, while the multiple 2d convs output exactly
    n_frames feature maps of size h x c x n_kernels2d. kernels2d and kernels3d must be the same here
    to be able to sum those two feature cubes.

    We then pass the intermediate feature cubes through 2D convolutions that also operate per frame
    2d conv to obtain n_frames sets of n_kernels feature maps.
    '''

    print('x.shape',x.shape)
    branch3d = conv3d_bn(x,
                filters_branch, 3, 7, 7,
                strides=(1,2,2),
                use_bn=use_bn,
                padding='same',
                name=name+'_conv3d_branch', act_before_bn=act_before_bn)
    branch3d = MaxPooling3D(strides=(1,2,2), padding='same')(branch3d)
    print('x.shape',x.shape)

    print('branch3d.shape',branch3d.shape)
    branch2d = TimeDistributed(Conv2D(filters_branch,
                kernel_size=7,
                strides=(2, 2),
                padding='same',
                dilation_rate=(1,1),
                use_bias=True,
                activation = 'relu',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-5),
                bias_regularizer=l2(5e-5),
                name=name+'_conv2d_branch'))(x)
    if use_bn:
        branch2d = TimeDistributed(BatchNormalization())(branch2d)

    branch2d = TimeDistributed(MaxPooling2D(strides=(2,2)))(branch2d)

    print('branch2d size, branch3d size:', branch2d.shape, branch3d.shape)
    x = Add()([branch2d, branch3d])

    for i in range(len(filters_post)):
        x = TimeDistributed(
            Conv2D(filters_post[i],
                        kernel_size=kernel_sizes_post[i],
                        strides=(1,1),
                        padding='same',
                        dilation_rate=(1,1),
                        use_bias=True,
                        activation = 'relu',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(5e-5),
                        bias_regularizer=l2(5e-5),
                        name=name+'_conv2d_post'))(x)

    print('shape of tensor at end of mict block named',name,':', x.shape)

    return x


def TimeDistributedConvBN(
            filters,
            kernel_size,
            strides=(1, 1),
            dilation_rate=(1, 1),
            use_bias=True, # We could change this if too many parameters or not learning fast enough
            use_activation_fn = True,
            use_bn = True,
            padding='same',
            name=None,
            w_decay=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_co'
    else:
        bn_name = None
        conv_name = None

    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = -1

    def f(input):
        x = TimeDistributed(Conv2D(filters,
                    kernel_size,
                    strides=strides,
                    padding=padding,
                    dilation_rate=dilation_rate,
                    use_bias=use_bias,
                    activation = 'relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(w_decay) if w_decay else None,
                    bias_regularizer=l2(w_decay) if w_decay else None,
                    name=conv_name))(input)


        x = TimeDistributed(BatchNormalization(axis=bn_axis, name=bn_name))(x)

        return x
    return f


# This inc. module also corresponds to Inception-v4 A
def TimeDistributedInceptionFig5(w_decay, output_dim, name=None, use_bn=True):

    tower_out_dim = output_dim//4

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    def f(input):

        # Tower A
        conv_a1 = TimeDistributedConvBN(tower_out_dim//2, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tda1')(input)
        conv_a2 = TimeDistributedConvBN(tower_out_dim//2, 3, w_decay=w_decay, use_bn=use_bn,name=name+'_tda2')(conv_a1)
        conv_a3 = TimeDistributedConvBN(tower_out_dim, 3, w_decay=w_decay, use_bn=use_bn,name=name+'_tda3')(conv_a2)
        # Tower B
        conv_b1 = TimeDistributedConvBN(tower_out_dim//2, 1, w_decay=w_decay, use_bn=use_bn,name=name+'_tdb1')(input)
        conv_b2 = TimeDistributedConvBN(tower_out_dim, 3, w_decay=w_decay, use_bn=use_bn,name=name+'_tdb2')(conv_b1)
        # Tower C
        pool_c1 = TimeDistributed(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name=name+'_tdc1'))(input)
        conv_c2 = TimeDistributedConvBN(tower_out_dim, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdc2')(pool_c1)
        # Tower D
        conv_d1 = TimeDistributedConvBN(tower_out_dim, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdd1')(input)
        return layers.concatenate([conv_a3, conv_b2, conv_c2, conv_d1], axis=c_axis, name=name+'_cat')

    return f

# This inc. module also corresponds to Inception-v4 B
def TimeDistributedInceptionFig6(w_decay, output_dim, name=None, use_bn=True):

    tower_out_dim = output_dim//4
    # print('tower_out_dim',tower_out_dim)

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    def f(input):

        # Tower A
        conv_a1 = TimeDistributedConvBN(int(tower_out_dim*.75), 1, w_decay=w_decay, use_bn=use_bn,name=name+'_tda1')(input)
        conv_a2 = TimeDistributedConvBN(int(tower_out_dim*.75), (1,7), w_decay=w_decay, use_bn=use_bn, name=name+'_tda2')(conv_a1)
        conv_a3 = TimeDistributedConvBN(int(tower_out_dim*.875), (7,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tda3')(conv_a2)
        conv_a4 = TimeDistributedConvBN(int(tower_out_dim*.875), (1,7), w_decay=w_decay, use_bn=use_bn, name=name+'_tda4')(conv_a3)
        conv_a5 = TimeDistributedConvBN(tower_out_dim, (7,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tda5')(conv_a4)

        # Tower B
        conv_b1 = TimeDistributedConvBN(int(tower_out_dim*.75), 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdb1')(input)
        conv_b2 = TimeDistributedConvBN(int(tower_out_dim*.875), (1,7), w_decay=w_decay, use_bn=use_bn, name=name+'_tdb2')(conv_b1)
        conv_b3 = TimeDistributedConvBN(tower_out_dim, (7,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tdb3')(conv_b2)

        # Tower C
        pool_c1 = TimeDistributed(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name=name+'_tdc1'))(input)
        conv_c2 = TimeDistributedConvBN(tower_out_dim//2, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdc2')(pool_c1)
        # Tower D
        conv_d1 = TimeDistributedConvBN(int(tower_out_dim*1.5), 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdd1')(input)

        return layers.concatenate([conv_a5, conv_b3, conv_c2, conv_d1], axis=c_axis, name=name+'_cat')

    return f



def TimeDistributedInceptionFig7(w_decay, output_dim, name=None, use_bn=True):

    tower_out_dim = output_dim//6
    # print('tower_out_dim',tower_out_dim)

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    def f(input):

        # Tower A
        conv_a1 = TimeDistributedConvBN(int(tower_out_dim*1.5), 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdc1')(input)
        conv_a2 = TimeDistributedConvBN(int(tower_out_dim*1.75), (1,3), w_decay=w_decay, use_bn=use_bn, name=name+'_tdc2')(conv_a1)
        conv_a3 = TimeDistributedConvBN(tower_out_dim*2, (3,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tdc3')(conv_a2)
        conv_a4a = TimeDistributedConvBN(tower_out_dim, (1,3), w_decay=w_decay, use_bn=use_bn, name=name+'_tdc2')(conv_a3)
        conv_a4b = TimeDistributedConvBN(tower_out_dim, (3,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tdc3')(conv_a3)
        # Tower B
        conv_b1 = TimeDistributedConvBN(int(tower_out_dim*1.5), 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdb1')(input)
        conv_b2a = TimeDistributedConvBN(tower_out_dim, (1,3), w_decay=w_decay, use_bn=use_bn, name=name+'_tdb2a')(conv_b1)
        conv_b2b = TimeDistributedConvBN(tower_out_dim, (3,1), w_decay=w_decay, use_bn=use_bn, name=name+'_tdb2b')(conv_b1)
        # Tower C
        pool_c1 = TimeDistributed(AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name=name+'_tdc1'))(input)
        conv_c2 = TimeDistributedConvBN(tower_out_dim, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdc2')(pool_c1)
        # Tower D
        conv_d1 = TimeDistributedConvBN(tower_out_dim, 1, w_decay=w_decay, use_bn=use_bn, name=name+'_tdd1')(input)
        return layers.concatenate([conv_a4a, conva4b, conv_b2a, conv_b2b, conv_c2, conv_d1], axis=c_axis, name=name+'_cat')

    return f


def Inception3DFig5(w_decay, output_dim, name=None, strides=(1,1,1),
                    dilation_rate = (1,1,1), use_bn=True, pooling='avg'):
    '''Inception block with 3d Convolutions, inspired by Fig5 of the original inception paper.'''

    tower_out_dim = output_dim//4
    # print('tower_out_dim',tower_out_dim)

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    if pooling == 'avg':
        Pooling = AveragePooling3D
    else:
        Pooling = MaxPooling3D

    def f(x):

        branch_0 = conv3d_bn(x, tower_out_dim, 1, 1, 1, strides = strides, use_bn=use_bn, padding='same', name=name+'_co3d_0a_1x1x1')

        branch_1 = conv3d_bn(x, tower_out_dim//2, 1, 1, 1, strides = strides, use_bn=use_bn, padding='same', name=name+'_co3d_1a_1x1x1')
        branch_1 = conv3d_bn(branch_1, tower_out_dim//2, 3, 3, 3, dilation_rate=dilation_rate, use_bn=use_bn, padding='same', name=name+'_co3d_1b_3x3x3')
        branch_1 = conv3d_bn(branch_1, tower_out_dim, 3, 3, 3, padding='same', use_bn=use_bn, name=name+'_co3d_1c_3x3x3')

        branch_2 = conv3d_bn(x, tower_out_dim//2, 1, 1, 1,  strides = strides, use_bn=use_bn, padding='same', name=name+'_co3d_2a_1x1x1')
        branch_2 = conv3d_bn(branch_2, tower_out_dim, 3, 3, 3, dilation_rate=dilation_rate, use_bn=use_bn, padding='same', name=name+'_co3d_2b_3x3')

        branch_3 = Pooling((3, 3, 3), strides = strides, padding='same', name=name+'_pool_3a_3x3x3')(x)
        branch_3 = conv3d_bn(branch_3, tower_out_dim, 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_3b_1x1x1')

        branch3d_concat = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=c_axis,
            name=name+'_cat')

        return branch3d_concat

    return f

def Inception3DFig6(w_decay, output_dim, name=None, pooling='avg', use_bn=True):
    '''Inception block with 3d Convolutions, inspired by Fig6 of the original inception paper.'''

    tower_out_dim = output_dim//4

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    if pooling == 'avg':
        Pooling = AveragePooling3D
    else:
        Pooling = MaxPooling3D

    def f(x):

        branch_0 = conv3d_bn(x, int(tower_out_dim*1.5), 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_0a_1x1x1')

        branch_1 = conv3d_bn(x, int(tower_out_dim*.75), 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2a_1x1x1')
        branch_1 = conv3d_bn(branch_1, int(tower_out_dim*.75), 1, 1, 7, use_bn=use_bn, padding='same', name=name+'_co3d_2b_1x1x7')
        branch_1 = conv3d_bn(branch_1, int(tower_out_dim*.75), 1, 7, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2c_1x7x1')
        branch_1 = conv3d_bn(branch_1, int(tower_out_dim*.875), 7, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2d_7x1x1')
        branch_1 = conv3d_bn(branch_1, int(tower_out_dim*.875), 1, 1, 7, use_bn=use_bn, padding='same', name=name+'_co3d_2e_1x1x7')
        branch_1 = conv3d_bn(branch_1, int(tower_out_dim*.875), 1, 7, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2f_1x7x1')
        branch_1 = conv3d_bn(branch_1, tower_out_dim, 7, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2g_7x1x1')

        branch_2 = conv3d_bn(x, int(tower_out_dim*.625), 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2a_1x1x1')
        branch_2 = conv3d_bn(branch_2, int(tower_out_dim*.75), 1, 1, 7, use_bn=use_bn, padding='same', name=name+'_co3d_2b_1x1x7')
        branch_2 = conv3d_bn(branch_2, int(tower_out_dim*.875), 1, 7, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2c_1x7x1')
        branch_2 = conv3d_bn(branch_2, tower_out_dim, 7, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2c_7x1x1')

        branch_3 = Pooling((3, 3, 3), strides=(1, 1, 1), padding='same', name=name+'_pool_3a_3x3')(x)
        branch_3 = conv3d_bn(branch_3, tower_out_dim//2, 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_3d_3b_1x1')

        branch3d_concat = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=c_axis,
            name=name+'_cat')

        return branch3d_concat

    return f


def Inception3DFig7(w_decay, output_dim, name=None, pooling='avg', use_bn=True):
    '''Inception block with 3d Convolutions, inspired by Fig7 of the original inception paper.'''

    tower_out_dim = output_dim//8

    if K.image_data_format() == 'channels_first':
        c_axis = 1
    else:
        c_axis = -1

    if pooling == 'avg':
        Pooling = AveragePooling3D
    else:
        Pooling = MaxPooling3D

    def f(x):

        branch_0 = conv3d_bn(x, tower_out_dim, 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_0a_1x1x1')

        branch_1 = conv3d_bn(x, int(tower_out_dim*1.5), 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2a_1x1x1')
        branch_1 = conv3d_bn(branch_1, tower_out_dim, 1, 1, 3, use_bn=use_bn, padding='same', name=name+'_co3d_2b_1x1x3')
        branch_1 = conv3d_bn(branch_1, tower_out_dim, 1, 3, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2c_1x3x1')
        branch_1 = conv3d_bn(branch_1, tower_out_dim, 3, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2d_3x1x1')
        branch_1a = conv3d_bn(branch_1, tower_out_dim, 1, 1, 3, use_bn=use_bn, padding='same', name=name+'_co3d_2e_1x1x3')
        branch_1b = conv3d_bn(branch_1, tower_out_dim, 1, 3, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2f_1x3x1')
        branch_1c = conv3d_bn(branch_1, tower_out_dim, 3, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2g_3x1x1')

        branch_2 = conv3d_bn(x, int(tower_out_dim*1.5), 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2a_1x1x1')
        branch_2a = conv3d_bn(branch_2, tower_out_dim, 1, 1, 3, use_bn=use_bn, padding='same', name=name+'_co3d_2b_1x1x3')
        branch_2b = conv3d_bn(branch_2, tower_out_dim, 1, 3, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2c_1x3x1')
        branch_2c = conv3d_bn(branch_2, tower_out_dim, 3, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_2d_3x1x1')

        branch_3 = Pooling((3, 3, 3), strides=(1, 1, 1), padding='same', name=name+'_pool_3a_3x3')(x)
        branch_3 = conv3d_bn(branch_3, tower_out_dim, 1, 1, 1, use_bn=use_bn, padding='same', name=name+'_co3d_3b_1x1')

        branch3d_concat = layers.concatenate(
            [branch_0, branch_1a, branch_1b, branch_1c, branch_2a, branch_2b, branch_2c, branch_3],
            axis=c_axis,
            name=name+'_cat')

        return branch3d_concat

    return f



##########################
### i3d model ###
##########################

# """Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
#
# The model is introduced in:
#
# Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
# Joao Carreira, Andrew Zisserman
# https://arxiv.org/abs/1705.07750v1
# """


def Inception_Inflated3d(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                dropout_prob=0.0,
                endpoint_logit=True,
                download_weights=True,
                downsample_factor=None,
                verbose=True,
                classes=400):
    """Instantiates the Inflated 3D Inception v1 architecture.

    Optionally loads weights pre-trained
    on Kinetics. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input frame(image) size for this model is 224x224.

    # Arguments
        include_top: whether to include the the classification
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or 'kinetics_only' (pre-training on Kinetics dataset only).
            or 'imagenet_and_kinetics' (pre-training on ImageNet and Kinetics datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(NUM_FRAMES, 224, 224, 3)` (with `channels_last` data format)
            or `(NUM_FRAMES, 3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
            NUM_FRAMES should be no smaller than 8. The authors used 64
            frames per example for training and testing on kinetics dataset
            Also, Width and height should be no smaller than 32.
            E.g. `(64, 150, 150, 3)` would be one valid value.
        dropout_prob: optional, dropout probability applied in dropout layer
            after global average pooling layer.
            0.0 means no dropout is applied, 1.0 means dropout is applied to all features.
            Note: Since Dropout is applied just before the classification
            layer, it is only useful when `include_top` is set to True.
        endpoint_logit: (boolean) optional. If True, the model's forward pass
            will end at producing logits. Otherwise, softmax is applied after producing
            the logits to produce the class probabilities prediction. Setting this parameter
            to True is particularly useful when you want to combine results of rgb model
            and optical flow model.
            - `True` end model forward pass at logit output
            - `False` go further after logit to produce softmax predictions
            Note: This parameter is only useful when `include_top` is set to True.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in WEIGHTS_NAME or weights is None or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or %s' %
                         str(WEIGHTS_NAME) + ' '
                         'or a valid path to a file containing `weights` values')

    if weights in WEIGHTS_NAME and include_top and classes != 400:
        raise ValueError('If using `weights` as one of these %s, with `include_top`'
                         ' as true, `classes` should be 400' % str(WEIGHTS_NAME))

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_frame_size=224,
        min_frame_size=32,
        default_num_frames=64,
        min_num_frames=8,
        data_format=K.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 4

    # Downsample temporally
    if downsample_factor:
        if K.image_data_format() == 'channels_first':
            print('Adding temporal downsample layer with factor %d in chanels_first mode' % downsample_factor)
            x = Lambda(lambda t: t[:,:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1], s[2]//downsample_factor, s[3], s[4]),name='temp_downsample')(img_input)
        else:
            print('Adding temporal downsample layer with factor %d in chanels_last mode' % downsample_factor)
            x = Lambda(lambda t: t[:,::downsample_factor,...],
                       output_shape=lambda s: (s[0], s[1]//downsample_factor, s[2], s[3], s[4]),name='temp_downsample')(img_input)

    else:
        x = img_input


    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(x, 64, 7, 7, 7, strides=(2, 2, 2), padding='same', name='Conv3d_1a_7x7')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, 1, 1, 1, strides=(1, 1, 1), padding='same', name='Conv3d_2b_1x1')
    x = conv3d_bn(x, 192, 3, 3, 3, strides=(1, 1, 1), padding='same', name='Conv3d_2c_3x3')

    # Downsampling (spatial only)
    x = MaxPooling3D((1, 3, 3), strides=(1, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)


    print('x.shape after downsa:', x.shape)
    print(K.image_data_format())


    # Mixed 3b
    branch_0 = conv3d_bn(x, 64, 1, 1, 1, padding='same', name='Conv3d_3b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_3b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 128, 3, 3, 3, padding='same', name='Conv3d_3b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_3b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 32, 3, 3, 3, padding='same', name='Conv3d_3b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 32, 1, 1, 1, padding='same', name='Conv3d_3b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')

    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_3c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 192, 3, 3, 3, padding='same', name='Conv3d_3c_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_3c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 96, 3, 3, 3, padding='same', name='Conv3d_3c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_3c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_3c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_4a_3x3')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_4b_0a_1x1')

    branch_1 = conv3d_bn(x, 96, 1, 1, 1, padding='same', name='Conv3d_4b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 208, 3, 3, 3, padding='same', name='Conv3d_4b_1b_3x3')

    branch_2 = conv3d_bn(x, 16, 1, 1, 1, padding='same', name='Conv3d_4b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 48, 3, 3, 3, padding='same', name='Conv3d_4b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4c_0a_1x1')

    branch_1 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 224, 3, 3, 3, padding='same', name='Conv3d_4c_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    # Mixed 4d
    branch_0 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_0a_1x1')

    branch_1 = conv3d_bn(x, 128, 1, 1, 1, padding='same', name='Conv3d_4d_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 256, 3, 3, 3, padding='same', name='Conv3d_4d_1b_3x3')

    branch_2 = conv3d_bn(x, 24, 1, 1, 1, padding='same', name='Conv3d_4d_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4d_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4d_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4d_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn(x, 112, 1, 1, 1, padding='same', name='Conv3d_4e_0a_1x1')

    branch_1 = conv3d_bn(x, 144, 1, 1, 1, padding='same', name='Conv3d_4e_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 288, 3, 3, 3, padding='same', name='Conv3d_4e_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4e_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 64, 3, 3, 3, padding='same', name='Conv3d_4e_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4e_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 64, 1, 1, 1, padding='same', name='Conv3d_4e_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_4f_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_4f_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_4f_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_4f_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_4f_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_4f_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_4f_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')


    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='MaxPool2d_5a_2x2')(x)

    # Mixed 5b
    branch_0 = conv3d_bn(x, 256, 1, 1, 1, padding='same', name='Conv3d_5b_0a_1x1')

    branch_1 = conv3d_bn(x, 160, 1, 1, 1, padding='same', name='Conv3d_5b_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 320, 3, 3, 3, padding='same', name='Conv3d_5b_1b_3x3')

    branch_2 = conv3d_bn(x, 32, 1, 1, 1, padding='same', name='Conv3d_5b_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5b_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5b_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5b_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, 1, 1, 1, padding='same', name='Conv3d_5c_0a_1x1')

    branch_1 = conv3d_bn(x, 192, 1, 1, 1, padding='same', name='Conv3d_5c_1a_1x1')
    branch_1 = conv3d_bn(branch_1, 384, 3, 3, 3, padding='same', name='Conv3d_5c_1b_3x3')

    branch_2 = conv3d_bn(x, 48, 1, 1, 1, padding='same', name='Conv3d_5c_2a_1x1')
    branch_2 = conv3d_bn(branch_2, 128, 3, 3, 3, padding='same', name='Conv3d_5c_2b_3x3')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same', name='MaxPool2d_5c_3a_3x3')(x)
    branch_3 = conv3d_bn(branch_3, 128, 1, 1, 1, padding='same', name='Conv3d_5c_3b_1x1')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')


    if include_top:
        # Classification block
        x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)
        x = Dropout(dropout_prob)(x)

        x = conv3d_bn(x, classes, 1, 1, 1, padding='same',
                use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

        print('x.shape after last conv:', x.shape)

        num_frames_remaining = int(x.shape[1])
        x = Reshape((num_frames_remaining, classes))(x)

        # logits (raw scores for each class)
        x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                   output_shape=lambda s: (s[0], s[2]))(x)

        if not endpoint_logit:
            x = Activation('softmax', name='prediction')(x)
    else:
        h = int(x.shape[2])
        w = int(x.shape[3])
        x = AveragePooling3D((2, h, w), strides=(1, 1, 1), padding='valid', name='global_avg_pool')(x)



    inputs = img_input
    # create model
    model = Model(inputs, x, name='i3d_inception')

    # load weights
    if weights in WEIGHTS_NAME and download_weights:
        if weights == WEIGHTS_NAME[0]:   # rgb_kinetics_only
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_kinetics_only']
                model_name = 'i3d_inception_rgb_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[1]: # flow_kinetics_only
            if include_top:
                weights_url = WEIGHTS_PATH['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_kinetics_only']
                model_name = 'i3d_inception_flow_kinetics_only_no_top.h5'

        elif weights == WEIGHTS_NAME[2]: # rgb_imagenet_and_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['rgb_imagenet_and_kinetics']
                model_name = 'i3d_inception_rgb_imagenet_and_kinetics_no_top.h5'

        elif weights == WEIGHTS_NAME[3]: # flow_imagenet_and_kinetics
            if include_top:
                weights_url = WEIGHTS_PATH['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics.h5'
            else:
                weights_url = WEIGHTS_PATH_NO_TOP['flow_imagenet_and_kinetics']
                model_name = 'i3d_inception_flow_imagenet_and_kinetics_no_top.h5'

        downloaded_weights_path = get_file(model_name, weights_url, cache_subdir='models')

        print('Loading downloaded weights into model:',weights, 'from path:', downloaded_weights_path)
        model.load_weights(downloaded_weights_path)

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your keras config '
                          'at ~/.keras/keras.json.')

    elif weights is not None:
        print('Loading own weights into model:',weights)
        model.load_weights(weights)

    return model


###########################
### KERAS CUSTOM LAYERS ###
###########################

class RoiPoolingConv(keras.layers.Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **kwargs):

        self.dim_ordering = K.image_dim_ordering()
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            # self.nb_channels = input_shape[0][1]
            self.nb_channels = input_shape[1]

        elif self.dim_ordering == 'tf':
            # self.nb_channels = input_shape[0][3]
            self.nb_channels = input_shape[3]


    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            return None, self.num_rois, self.nb_channels, self.pool_size, self.pool_size
        else:
            return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        # assert (len(x) == 2)

        img = x[...,:-4]
        rois = x[...,0,:1,-4:]

        print('img.shape',img.shape)
        print('rois.shape',rois.shape)
        # img = x[0]
        # rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            if self.dim_ordering == 'th':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        dx = K.maximum(1, x2 - x1)
                        x2 = x1 + dx

                        dy = K.maximum(1, y2 - y1)
                        y2 = y1 + dy

                        new_shape = [input_shape[0], input_shape[1],
                                     y2 - y1, x2 - x1]

                        x_crop = img[:, :, y1:y2, x1:x2]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(2, 3))
                        outputs.append(pooled_val)

            elif self.dim_ordering == 'tf':
                for jy in range(num_pool_regions):
                    for ix in range(num_pool_regions):
                        x1 = x + ix * row_length
                        x2 = x1 + row_length
                        y1 = y + jy * col_length
                        y2 = y1 + col_length

                        x1 = K.cast(x1, 'int32')
                        x2 = K.cast(x2, 'int32')
                        y1 = K.cast(y1, 'int32')
                        y2 = K.cast(y2, 'int32')

                        new_shape = [input_shape[0], y2 - y1,
                                     x2 - x1, input_shape[3]]
                        x_crop = img[:, y1:y2, x1:x2, :]
                        xm = K.reshape(x_crop, new_shape)
                        pooled_val = K.max(xm, axis=(1, 2))
                        outputs.append(pooled_val)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        if self.dim_ordering == 'th':
            final_output = K.permute_dimensions(final_output, (0, 1, 4, 2, 3))
        else:
            final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output



class RoiPoolingConvOneInput(RoiPoolingConv):


    def __init__(self, pool_size, num_rois, **kwargs):
        print('Init RoiPoolingConvOneInput')
        super(RoiPoolingConvOneInput, self).__init__(pool_size, num_rois, **kwargs)

    def call(self, x, mask=None):
        # The input X is a set of F feature maps concatenated with bbox coordinates.
        # Shape of x: batch_size x frames x H x W x channels+1
        # Original shape of bbox batch: batch_size x frames x 4
        # Warped shape of bbox batch: batch_size x frames x H x W x 4
        print('Calling RoiPoolingConvOneInput')
        # fmaps = x[...,:-1]
        # bboxes = x[...,0,0,:]



        print('bboxes.shape after slicing:', bboxes.shape)

        # dual_input = [fmaps, bboxes]
        dual_input = [x,x]

        super(RoiPoolingConvOneInput, self).call(dual_input)





# Import Necessary Modules.
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})
