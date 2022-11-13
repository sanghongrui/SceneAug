#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   arguments.py
@Time    :   2022/10/09 16:39:05
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Base_Agent Arguments')
    # General Arguments
    parser.add_argument(
        '--exp-config', 
        choices=["configs/experiment_config/ppo_interaction_nav.yaml"], 
        default="configs/experiment_config/ppo_interaction_nav.yaml",
        help="experiment config file path."
    )

    parser.add_argument(
        '--agent-type', 
        choices=["base-rl"], 
        default='base-rl',
        help="agent type of the experiment."
    )

    parser.add_argument(
        '--run-type', 
        choices=["train", "eval"], 
        default='train',
        help="run type of the experiment (train or eval)."
    )

    parser.add_argument(
        '--gpu-idx', 
        choices=[0, 1, 2],
        type=int, 
        default=0,
        help="torch_gpu_id and igibson_gpu_id (default: 0)."
    )
    
    parser.add_argument(
        '--scene', 
        choices=["Rs_int", "Beechwood_0_int"], 
        default="Beechwood_0_int",
        help="train scenes (default: Beechwood_0_int)."
    )
    
    parser.add_argument(
        '--scene-aug-type', 
        choices=["random_none", "scene_random", "room_instance_random", "random_remove", "random_open"], 
        default="random_none",
        help="scene augmentation method (default: random_none)."
    )

    parser.add_argument(
        '--model-action-type', 
        choices=["base_only", "base_arm"], 
        default='base_only',
        help="agent type of the experiment (default: base_only)."
    )

    parser.add_argument(
        '--model-aug-type', 
        choices=["base", "scene", "room", "remove", "open"], 
        default='base',
        help="cpt type of the experiment (default: base)."
    )

    parser.add_argument(
        '--eval-type', 
        choices=['none', 'unseen_scene', 'unseen_states', 'unseen_layout_scene', 
                 'unseen_layout_remove', 'unseen_objects'
        ], 
        default='none',
        help="eval type(default: none)."
    )
    
    parser.add_argument(
        '--start-ckpt-index', 
        type=int, 
        default=50,
        help="eval start ckpt index (default: 50)."
    )
    
    # Add More
    
    # parse arguments
    args = parser.parse_args()
    
    # modify args
    
    # return args
    return args
    
if __name__ == "__main__":
    args = get_args()
    print(args)

    print("test")