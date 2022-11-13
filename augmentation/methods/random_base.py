#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   random_open.py
@Time    :   2022/07/15 14:45:43
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
from abc import ABCMeta, abstractmethod
import numpy as np
from augmentation.envs.augmented_env import AugmentedEnv
from augmentation.envs.augmented_interactive_indoor_scene import AugInteractiveIndoorScene

class Augmentation_Base():
    """
    Base Scene Augmentation Method class
    """
    
    __metaclass__ = ABCMeta
    
    def __init__(
        self, 
        scene:AugInteractiveIndoorScene,
        config,
    ) -> None:
        
        self._name = 'random_base'
        
        self.config = {}
        self.config['name'] = config['name']
        self.config.update(config['base'])
        self.config.update(config[config['name']])
        
        self.scene = scene
        self.prob = self.config.get('prob', 0.5)

    def run(self, env:AugmentedEnv):
        """
        @description: method-specific  scene augmentation
        ---------
        @param  :
        -------
        @Returns  : None
        -------
        """ 
        if np.random.random() > self.prob:
            return
        
        # augment scene and update its floor_map accordingly.
        self.augmentation(env)
        
        # update florr_graph
        env.scene.build_graph_without_editing_trav_map(env.task.floor_num)
        
    @abstractmethod
    def augmentation(self, env:AugmentedEnv):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        raise NotImplementedError()