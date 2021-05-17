'''
Utility methods for running jobs, changing parameters, and sending multiple
jobs at the same time to the dgx server.
'''
__author__ = "Camilo Fosco"
__email__ = "cfosco@nvidia.com"

import os
import numpy as np
import sys
import subprocess



JSON_TEMPLATE1 = \
'{\
    "jobDefinition": {\
        "name": "%s",\
        "description": "Processing i3d",\
        "clusterId": %d,\
        "dockerImage": "nvidian_general/iva:cfosco-action_recognition",\
        "jobType": "BATCH",\
        "command": "%s",\
        "resources": {\
            "gpus": %d,\
            "systemMemory": %d,\
            "cpuCores": %d\
        },\
        "jobDataLocations": [\
            {\
                "mountPoint": "/home/cfosco/nfs_share",\
                "protocol": "NFSV3",\
                "sharePath": "/export/iva_action_recognition.cosmos192",\
                "shareHost": "dcg-zfs-04.nvidia.com"\
            }\
        ],\
        "portMappings": []\
    }\
}'


COMMAND_TEMPLATE1 = \
"cd nfs_share/action_recognition/camilo_workspace/src;\
python3 keras_eval.py \
-eval_ckpt %s \
--dataset %s --model_type %s --split %d --verbose %d --num_classes %d \
--batch_size %d --gpus %d --workers %d --use_mp %d --do_per_class %d --type %s %s "

def evaluate_all_ckpts(ckpt_folder, ckpt_loc, req_str='HMDB51', model_type = 'i3d', split=1, dataset ='HMDB51', num_classes=51, batch_size=5,
                        workers=10, use_mp=0, verbose=1, min_epoch=0, cpus=12, gpus=4, ram=64, cluster_id=425, do_per_class=0, extra_cmd = '',
                        classes_to_process=None):
    if classes_to_process:
        extra_cmd += ' --classes_to_process %s ' % classes_to_process


    ckpts = [f for f in os.listdir(ckpt_folder) if '.hdf5' in f]
    names=[]
    for ck in ckpts:
        ep = int(ck.split('_ep')[1].split('_')[0])
        if ep >= min_epoch:
            if req_str in ck:
                if 'flow' in ck:
                    type = 'flow'
                elif 'rgb' in ck:
                    type = 'rgb'
                print('Checkpoint',ck,'fulfills requirements. Starting eval job...')
                cmd = create_command(os.path.join(ckpt_loc, ck), dataset=dataset, model_type=model_type,
                                    split=split, verbose=verbose, batch_size=batch_size, num_classes=num_classes,
                                    gpus=gpus, workers=workers, use_mp=use_mp, type=type, extra_cmd=extra_cmd, do_per_class=do_per_class)
                name = 'ckpt_'+req_str+'_ep'+str(ep)+'_eval'
                json_name = save_json(JSON_TEMPLATE1, cmd, name, cpus=cpus,
                                        gpus=gpus, ram=ram, cluster_id=cluster_id)
                out=subprocess.check_output(['dgx','job','submit','-f',json_name])
                for l in out.split('\n'):
                    if 'Id:' in l:
                        print(l)
                        break
                names.append(json_name)
    delete_tmp_jsons(names)

def create_command(ckpt, dataset='HMDB51', model_type='i3d', num_classes=51, split=1, verbose=1, batch_size=5, gpus=4, workers=10, use_mp=0, type='rgb', do_per_class=0, extra_cmd=''):
    return COMMAND_TEMPLATE1 % (ckpt, dataset, model_type, split, verbose, num_classes, batch_size, gpus, workers, use_mp, do_per_class, type, extra_cmd)

def delete_tmp_jsons(names):
    for n in names:
        try:
            os.remove(n)
        except Exception as err:
            print(err)
            print('Moving on')

def save_json(template, cmd, name='ckpt_eval', cpus=12,gpus=4,ram=64, cluster_id=425):
    json_str = template % (name, cluster_id, cmd, gpus, ram, cpus)
    json_name = "%s_gpus%d_ram%d_cpus%d" % (name, gpus, ram, cpus)
    with open (json_name, 'w+') as f:
        f.write(json_str)

    return json_name

# def train_multiple(init_ckpt, param_dict_to_test):
#     for k,v in param_dict_to_test:
#         for param_value in v:
#             json_name = save_json(ck, TEMPLATE1, )
#             subprocess.call(['dgx','job','submit','-f',json_name])
