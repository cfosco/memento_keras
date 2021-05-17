#!/usr/bin/env python
'''File with necessary functions to evaluate the keras i3d model.'''
__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"

# Imports
import sys
import os
import numpy as np
import keras.backend as K
from keras.optimizers import SGD
from keras_training import process_dataset_arg, top_2_accuracy, instantiate_model
from keras_models import build_model_multigpu
from generator import video_generator, VideoSeqGenerator, build_label2str, read_videofile_txt
from timeit import default_timer as timer
import i3d_config as cfg
import argparse
import tensorflow as tf


def eval_model(model, val_txt_loc, batch_size=10, max_queue_size=10, workers=1,
                use_mp=False, verbose=1, classes_to_process=None, func_gen=False,
                **gen_params):
    '''Main evaluation function. The model must be compiled.'''

    start = timer()

    files = read_videofile_txt(val_txt_loc)
    if classes_to_process and classes_to_process != 'all':
        files = restrict_to_desired_classes(files, classes_to_process)

    classnames = [f.split('/')[0] for f in files]
    classnames = np.unique(classnames)
    n_classes = len(classnames)

    num_val_vids = len(files)
    if func_gen:
        vid_gen = video_generator(files=files, batch_size=batch_size, **gen_params)
    else:
        vid_gen = VideoSeqGenerator(files=files, batch_size=batch_size, **gen_params)



    print('num_val_vids//batch_size',num_val_vids//batch_size)

    start = timer()
    metrics = model.evaluate_generator(vid_gen,
                                       steps=num_val_vids//batch_size,
                                       max_queue_size=max_queue_size,
                                       workers=workers,
                                       use_multiprocessing=use_mp)
    end = timer()
    time_eval = end-start
    if verbose:
        print('Time elapsed evaluating:', time_eval)
        print('Metrics on',num_val_vids,'videos. Loss:',metrics[0],' - Accuracy:',metrics[1], '- Top 2 Acc:', metrics[2])

    metric_dict = {}
    for i,m in enumerate(metrics):
        metric_dict[model.metrics_names[i]] = m

    # print('metric_dict',metric_dict)
    return metric_dict, n_classes ,num_val_vids, time_eval



def eval_i3d_over_splits(split_ckpts, val_txt_loc, num_classes, splits_to_test=[1,2,3],
                        batch_size=10, verbose = True,
                        max_queue_size=10, workers=1, use_mp=False, **gen_params):
    ''' Evaluation over multiple splits. important to send an equal amount of
    ckpts as number of splits to be tested.'''

    if len(split_ckpts) != len(splits_to_test):
        raise ValueError('Received a different amount of ckpts than splits to test. len(split_ckpts): '+str(len(split_ckpts)))

    metric_dict = {}
    for split in splits_to_test:
        not_evaluated=True
        for ckpt in split_ckpts:
            if ('split'+str(split) in ckpt) or ('split0'+str(split) in ckpt):
                print('Evaluating split'+str(split))
                metric_dict[split] = eval_i3d(ckpt, val_txt_loc, num_classes, batch_size=batch_size, verbose=verbose,
                                                max_queue_size=max_queue_size,workers=workers,use_mp=use_mp,**gen_params)
                not_evaluated=False
        if not_evaluated:
            print('Didnt find split substring in given checkpoints, evaluating with ckpt number',split)
            metric_dict[split] = eval_i3d(split_ckpts[split], val_txt_loc)

    metric_lists=[]
    for k,v in metric_dict:
        metrics=[]
        for c,w in v:
            metrics.append(w)
        metric_lists.append(metrics)
    # print('metric_dict:', metric_dict)
    print('Average Loss:',np.mean([m[0] for m in metric_lists]),
          ' - Average Accuracy:',np.mean([m[1] for m in metric_lists]),
          ' - Average Top 5 acc:',np.mean([m[2] for m in metric_lists]))
    return metric_dict


def eval_per_class(model, val_txt_loc, batch_size=10, max_queue_size=10, workers=1,
                    use_mp=False, verbose=1, classes_to_process=None, func_gen=False,  **gen_params):


    files=read_videofile_txt(val_txt_loc)
    if classes_to_process and classes_to_process != 'all':
        files = restrict_to_desired_classes(files, classes_to_process)

    file_dict={}
    for f in files:
        classname = f.split('/')[-2]
        if classname not in file_dict:
            file_dict[classname]=[]
        file_dict[classname].append(f)

    n_classes = len(file_dict)
    if verbose:
        print(n_classes,'classes to evaluate.')


    metric_dict_per_class={}
    metric_dict_full = {}
    for name in model.metrics_names:
        metric_dict_full[name] = 0
    for k,v in file_dict.items():
        if verbose:
            print('Evaluating',k, '. # videos:', len(v))
        start = timer()

        if func_gen:
            vid_gen = video_generator(files=v, batch_size=batch_size, **gen_params)
        else:
            vid_gen = VideoSeqGenerator(files=v, batch_size=batch_size, **gen_params)
        metrics = model.evaluate_generator(vid_gen, steps=len(v)//batch_size,
                        max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_mp)

        metric_dict_per_class[k]=metrics
        end = timer()
        if verbose:
            print('Time elapsed on',k,':', end-start)
            print('Metrics on ',len(v),'videos of ',k,': Loss:',metrics[0],
                  ' - Accuracy:',metrics[1], ' - Top 2 acc:', metrics[2])

        for i,m in enumerate(metrics):
            metric_dict_full[model.metrics_names[i]] += len(v)*m

    print('Metrics on full dataset (%d videos):' % len(files))
    for metric, value in metric_dict_full.items():
        metric_dict_full[metric] = value / len(files)
        print(metric, value/len(files))

    return metric_dict_per_class, metric_dict_full, n_classes, len(files)


def restrict_to_desired_classes(files, classes_to_process):
    if isinstance(classes_to_process, str) and classes_to_process:
        classes_to_process = read_videofile_txt(classes_to_process)
    files = [f for f in files if f.split('/')[0] in classes_to_process]
    return files


def metrics2csv(metric_dict, output_path, verbose = True):
    '''Writes metrics obtained from eval_i3d, eval_over_splits in a
    csv format.

    Inputs
    ------
    results: dict
        dictionary where keys are class id or general id, and values are a list of metrics,
        usually loss, accuracy and top_5_accuracy.
    '''

    with open(output_path, 'w+') as f:
        for k,v in metric_dict.items():
            if verbose:
                print('%s, %.4f, %.4f' % (k,v[1],v[2]))
            f.write('%s, %.4f, %.4f\n' % (k,v[1],v[2]))

def save_results(metric_dict, metrics_per_class=None, main_log_path='../../../main_eval_log.txt', id_str='model1', verbose=True):
    '''Appends metrics to the main log file.'''
    # print('metric dict in save_results', metric_dict)
    with open(main_log_path, 'a+') as f:
        f.write('%s:\n\t' % id_str)
        for k,v in sorted(metric_dict.items()):
            if verbose:
                print(k,v)
            f.write('%s:%.4f, ' % (k, v))
        if metrics_per_class:
            f.write('\n\t')
            for k,v in sorted(metrics_per_class.items()):
                f.write('%s:[%.3f,%.3f], ' % (k,v[1],v[2]))
        f.write('\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ## MAIN ARGUMENTS
    parser.add_argument('-eval_ckpt',
        help='Path to checkpoint.',type=str)

    parser.add_argument('--dataset',
        help='Name of dataset. Alternatively, the user can pass the num_classes, \
        path to data, prep and load funcs separately instead of setting this parameter.',
        type=str,
        default=cfg._DATASET)

    parser.add_argument('--verbose',
        help='Verbosity value.',
        type=int,
        default=cfg._VERBOSE)

    parser.add_argument('--gpus',
        help='Number of GPUs to use. If 1 or None, no multi_gpu model is instanciated.',
        type=int,
        default=cfg._GPUS)

    parser.add_argument('--model_type',
    help='Model type to evluate. Must match the given ckpt.',
    type=str, default='i3d')

    # TODO: Add argument to do per class calculations


    ## EVALUATE_GENERATOR SPECIFIC ARGS
    parser.add_argument('--batch_size',
        help='Batch size, fed to the evaluate_generator function',
        type=int,
        default=cfg._BATCH_SIZE_EVAL)
    parser.add_argument('--max_queue',
        help='Max queue size. Fed to evaluate_generator function.',
        type=int,
        default=cfg._MAX_QUEUE_EVAL)
    parser.add_argument('--workers',
        help='Number of workers for evaluate_generator',
        type=int,
        default=cfg._WORKERS_EVAL)
    parser.add_argument('--use_mp',
        help='Use multiprocessing or not. Send 0 or 1.',
        type=int,
        default=cfg._USE_MP_EVAL)

    ## NEW DATASET ARGS
    ## These arguments are only used if --dataset is not provided or not recognized
    parser.add_argument('--data_path',
        help='Path to the training folder. In this folder, videos should be in subfolders per classname: classname/videoname',
        type=str, default=cfg._DATA_PATH)

    parser.add_argument('--num_classes',
        help='Number of classes for the given dataset',
        type=int, default=cfg._NUM_CLASSES)

    parser.add_argument('--prep_func',
        help='Preprocessing function',
        type=str,
        default=cfg._PREP_FUNC)

    parser.add_argument('--load_func',
        help='Load Function',
        type=str,
        default=cfg._LOAD_FUNC)

    ## OTHER OPTIONAL ARGS
    parser.add_argument('--txt_path',
        help='Path to txt containing paths of videos to evaluate, in format classname/videoname',
        type=str, default=cfg._VAL_TXT_PATH)

    parser.add_argument('--split',
        help='Split number. Optional. If txt_path is defined, this arg is overrided.',
        type=int,
        default=cfg._SPLIT)

    parser.add_argument('--labels_path',
    help='Path to csv containing label names and their corresponding values',
    type=str, default=cfg._LABELS_PATH)

    parser.add_argument('--save_results',
    help='Whether to save results or not. Appends results to the main \
    eval log file: "../../../main_eval_log.txt"',
    type=int, default=1)

    parser.add_argument('--type',
    help='Type of net to load. one of [flow, rgb]',
    type=str, default=cfg._TYPE)

    parser.add_argument('--classes_to_process',
    help='Txt with list of classes to process, separated by newline. If nothing is provided, \
    all classes found in dataset folder will be evaluated.',
    type=str, default=cfg._CLASSES)

    parser.add_argument('--do_per_class',
    help='Whether to report per-class accuracy or not.',
    type=int, default=1)

    parser.add_argument('--dl_w',
    help='Whether to download imagenet default weights or not. Turn this off only if you know what youre doing, \
    e.g. if you are sending a multigpu checkpoint.',
    type=int, default=1)

    parser.add_argument('--downsample',
    help='Int factor by which to downsample temporally the input videos before running them through the model.',
    type=int, default=None)

    parser.add_argument('--remove_excess',
    help='Automatically removes some of given files to make sure that data is multiple of batch size.',
    type=int, default=0)

    parser.add_argument('--run_locally',
    help='Indicates that the script will be run on local machine or notebook instead of SaturnV. \
    This makes the program search for certain internal chekpoints in the correct locations.',
    type=int, default=0)

    # Parse arguments
    args = parser.parse_args()

    args = process_dataset_arg(args.dataset, args, is_train=False)


    print('Args:')
    for arg in vars(args):
        print (arg, getattr(args, arg))

    gen_params = dict(dataset_path=args.data_path,
                      label_csv=args.labels_path,
                      load_func=args.load_func,
                      preprocess_func=args.prep_func,
                      remove_excess_files=args.remove_excess,
                      shuffle=False, is_train=False)

    model = instantiate_model(args.model_type,
                                        ckpt = args.eval_ckpt, type=args.type,
                                        num_classes=args.num_classes,
                                        downsample_factor=args.downsample,
                                        verbose=args.verbose, gpus=args.gpus,
                                        run_locally=args.run_locally)


    metric_dict_per_class=None
    if args.do_per_class:
        start = timer()
        metric_dict_per_class, metric_dict_full, num_classes_analyzed, num_vids_analyzed = eval_per_class(model, args.txt_path,
                batch_size=args.batch_size, max_queue_size=args.max_queue,
                workers=args.workers, use_mp=args.use_mp, verbose=args.verbose,
                classes_to_process=args.classes_to_process, **gen_params )
        end = timer()
        time_eval = end-start
    else:
        metric_dict_full, num_classes_analyzed, num_vids_analyzed, time_eval = eval_model(model, args.txt_path,
                batch_size=args.batch_size, max_queue_size=args.max_queue,
                workers=args.workers, use_mp=args.use_mp, verbose=args.verbose,
                classes_to_process=args.classes_to_process, **gen_params )


    if args.save_results:
        print('Appending evaluation results to main log')
        id_str = "%s - classes %d - videos %d - split %d - batch_size %d - gpus %d - workers %d - EVAL TIME: %.3f" % \
            (args.eval_ckpt, num_classes_analyzed, num_vids_analyzed, args.split, args.batch_size, args.gpus, args.workers, time_eval)
        save_results(metric_dict_full, metric_dict_per_class, id_str=id_str, main_log_path='../../../main_eval_log.txt')
