#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   random_open.py
@Time    :   2022/07/15 15:32:17
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
from common.baseline_registry import baseline_registry
from augmentation.methods.random_base import Augmentation_Base

@baseline_registry.register_augmentation_method(name="random_none")
class Random_None(Augmentation_Base):
    '''
    Random Open Scene Augmentation Method class.s
    '''
    def __init__(
        self, 
        scene,
        config,
    ) -> None:
        super(Random_None, self).__init__(scene, config)
        
        self._name = 'none'

    def augmentation(
        self, 
        env
    ) -> None:   
        """
        Random open objects in self.all_openable_categories
        """
        pass
    