#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   example.py
@Time    :   2022/07/18 15:42:29
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
import argparse
import time
import cv2
import numpy as np
from igibson.utils.utils import parse_config
from augmentation.envs.augmented_env import AugmentedEnv

def highlight_categories(scene, categories_list):
    """
    Highlight objects for all categories
    categories_list: List[str]
    """
    for c in categories_list:
        highlight_category(scene, c)

def highlight_category(scene, obj_category):
    """
    Highlight objects by one category
    obj_category: str
    """
    for obj in scene.objects_by_category[obj_category]:
        obj.highlight()

def color_map(floor_map, rect=None):
    view_map = np.reshape(floor_map, (floor_map.shape[0], -1, 1))
    color_map = np.repeat(view_map, 3, 2)
    
    if rect is not None:
        cv2.rectangle(color_map,  (rect[0][1], rect[0][0]), (rect[1][1], rect[1][0]), color=(0,0,255), thickness=1)
    
    return color_map

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="which config file to use [default: use yaml files in examples/configs]"
)
parser.add_argument(
    "--mode",
    choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
    default="headless",
    help="which mode for simulation (default: headless)",
)
parser.add_argument(
    "--scene_id",
    "-s",
    choices=["Rs_int", "Beechwood_0_int", "Ihlen_1_int", "Benevolence_1_int"],
    default="",
    help="which scene to load",
)
parser.add_argument(
    '--scene-aug-type', 
    choices=["random_none", "scene_random", "room_instance_random", "random_remove", "random_open"], 
    default="random_none",
    help="scene augmentation method (default: random_none)."
)

args = parser.parse_args()
args.config = "configs/fetch_interactive_nav.yaml"
args.mode = "headless"   #"gui_interactive"
args.scene_id = "Rs_int"
args.scene_aug_type = "random_open"
args.scene_id = args.scene_id if args.scene_id != "" else None

igibson_config = parse_config(args.config)
aug_config = parse_config(igibson_config['aug_config_file'])   
# modify augmentation config file
aug_config['name'] = args.scene_aug_type
igibson_config['aug_config_file'] = aug_config

env = AugmentedEnv(
    config_file=igibson_config,
    scene_id=args.scene_id,
    mode=args.mode,
    action_timestep=1.0 / 10.0, 
    physics_timestep=1.0 / 40.0,
    use_pb_gui=False
)

if args.mode == "gui_interactive":
    viewer = env.simulator.viewer
    if env.scene.scene_id == "Rs_int":
        # rs_int: living_room_0
        viewer.initial_pos = [-0.9, 0.2, 6.5]
        viewer.initial_view_direction = [0.0, -0.0, -1.0]
    if env.scene.scene_id == "Beechwood_0_int":
        viewer.initial_pos = [-6.3, -1.6, 8.6]
        viewer.initial_view_direction = [0.0, 0.1, -1.0]
    if env.scene.scene_id == "Ihlen_1_int":
        viewer.initial_pos = [-0.2, 4.6, 8.9]
        viewer.initial_view_direction = [0.0, 0.0, -1.0]
    if env.scene.scene_id == "Benevolence_1_int":
        viewer.initial_pos = [-2.2, -3.1, 7.9]
        viewer.initial_view_direction = [0.0, 0.0, -1.0]

for episode in range(100):
    
    env.reset()
    # cv2.imshow("floor_map", cv2.flip(cv2.resize(env.scene.floor_map[0], (512,512)), 0))
    # highlight_categories(env.scene, ["bottom_cabinet_no_top", "door", "fridge", "chest", "bottom_cabinet"])
    
    print("Episode: {}".format(episode))
    # print("-" * 80)
    # for i in range(10):  # 10 seconds
        
    #     action = env.action_space.sample()
    #     state, reward, done, _ = env.step(action)
    #     env.simulator.sync(force_sync=True)
        
    #     cv2.imshow("map", cv2.flip(cv2.resize(env.scene.floor_map[0],(512,512)), 0))
    #     print("\nStep: ", i)
    #     print("Reward: ", reward)
    #     if done:
    #         break
    # print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    # print("-" * 80 + "\n")

env.close()
