#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   efficiency_analysis.py
@Time    :   2023/01/29 21:53:22
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang@tongji.edu.cn
'''
import logging
import time

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

from augmentation.envs.augmented_interactive_indoor_scene import AugInteractiveIndoorScene

num_resets = 100
num_steps_per_reset = 1

def igibson_object_randomization(headless=True):
    """
    Example of randomization of the texture in a scene
    Loads Rs_int (interactive) and randomizes the texture of the objects
    """
    
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
         
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    load_object_categories = None  # Use the list to load a partial house

    for i in range(num_resets):
        print("Randomizing objects ", i)
        
        timer_start = time.time()
        scene = InteractiveIndoorScene(
            "Rs_int",
            texture_randomization=False,
            load_object_categories=load_object_categories,
            object_randomization=True,
            object_randomization_idx=i,
        )
        s.import_scene(scene)
        for _ in range(num_steps_per_reset):
            s.step()
        s.reload()
        timer_end = time.time()

        per_time = (timer_end - timer_start)*1000   # in millisecond format
        print('time: ', per_time)
        writer("./igibson_object_randomization.log", per_time)
    
    s.disconnect()


def igibson_texture_randomization(headless=True):
    """
    Example of randomization of the texture in a scene
    Loads Rs_int (interactive) and randomizes the texture of the objects
    """
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = InteractiveIndoorScene(
        "Rs_int",
        # load_object_categories=[],  # To load only the building. Fast
        texture_randomization=True,
        object_randomization=False,
    )
    s.import_scene(scene)

    for i in range(num_resets):
        print("Randomize texture", i)
        
        timer_start = time.time()
        scene.randomize_texture()
        for _ in range(num_steps_per_reset):
            s.step()
        timer_end = time.time()

        per_time = (timer_end - timer_start)*1000   # in millisecond format
        print('time: ', per_time)
        writer("./igibson_texture_randomization.log", per_time)
    s.disconnect()

def writer(file_path, per_time):
    with open(file_path, 'a') as fp:
        fp.write('\n' + str(per_time))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # igibson_object_randomization(headless=True)
    igibson_texture_randomization(headless=True)
