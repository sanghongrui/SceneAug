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
from typing import List
from baseline_registry import baseline_registry
from augmentation.methods.random_base import Augmentation_Base

@baseline_registry.register_augmentation_method(name="random_open")
class Random_Open(Augmentation_Base):
    '''
    Random Open Scene Augmentation Method class.s
    '''
    def __init__(
        self, 
        scene,
        config,
    ) -> None:
        super(Random_Open, self).__init__(scene, config)
        
        self._name = 'random_open'
        self.all_openable_categories = self.config.get('include_categories')
        
    def augmentation(
        self, 
        env
    ) -> None:   
        """
        Random open objects in self.all_openable_categories
        """
        self.force_wakeup_by_categories(self.all_openable_categories)
        self.scene.open_all_objs_by_categories(
            self.all_openable_categories, mode="random", prob=1.0
        )
    
    def force_wakeup_by_categories(
        self, 
        categories: List[str]
    ):
        """
        Force wakeup sleeping objects for all categories
        categories: List[str]
        """
        for c in categories:
            self._force_wakeup_by_category(c)
    
    def _force_wakeup_by_category(
        self, 
        obj_category: str
    ):
        """
        @description: Force wakeup sleeping objects by one category
        ---------
        @param  obj_category: str
        -------
        @Returns: None
        -------
        """
        for obj in self.scene.objects_by_category[obj_category]:
            obj.force_wakeup()