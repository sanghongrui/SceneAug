#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   random_remove.py
@Time    :   2022/07/15 18:34:33
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
import random
import math
import numpy as np
from common.baseline_registry import baseline_registry
from augmentation.methods.random_base import Augmentation_Base


@baseline_registry.register_augmentation_method(name="random_remove")
class Random_Remove(Augmentation_Base):
    '''
    
    '''
    def __init__(
        self, 
        scene,
        config,
    ) -> None:
        super(Random_Remove, self).__init__(scene, config)
        
        self._name = 'random_remove'
        
        self.exclude_categories = self.config.get('exclude_categories')
        self.remove_threshold = self.config.get('remove_threshold', 0.5)
        self.erasing_area_ratio = self.config.get('erasing_area_ratio', [0.02, 0.4])
        self.min_aspect_ratio = self.config.get('min_aspect_ratio', 0.3)
         
    def augmentation(self, env):
        floor = env.task.floor_num
        
        random_rect = self.rect_generator(floor)
        self.random_remove(floor, random_rect)
    
    def random_remove(self, floor, random_rect):
        """
        @description  : random remove objects in area of self.random_rect,
        objects in random_remove_exclude_categories will be excluded.
        ---------
        @param  random_rect: random_rect instance
        -------
        @Returns  : None
        -------
        """
        for c, objs in self.scene.objects_by_category.items():
            if c in self.exclude_categories:
                continue
            
            for obj in objs:    
                aabb_map = self.scene.get_obj_aabb_map(obj)                
                
                if self._to_be_removed(aabb_map, random_rect) is True:   
                    obj.set_position([random.randint(100, 200), random.randint(100, 200), 100.0])
                    self.scene.remove_object_from_trav_map(floor, aabb_map)

    def _to_be_removed(self, obj_aabb_map, random_rect):
        """
        @description  : determine whether an obj should be removed or not 
        according the overlap between obj and self.random_rect
        ---------
        @param  obj_aabb_map: Numpy(rect[[x_min, y_min], [x_max, y_max]]), obj aabb in map space
        @param  random_rect: Numpy(rect[[x_min, y_min], [x_max, y_max]]), random_rect aabb  in map space
        -------
        @Returns  : bool.
        -------
        """
        obj_area_map = (
            (obj_aabb_map[1][0]+1 - obj_aabb_map[0][0]) *
            (obj_aabb_map[1][1]+1 - obj_aabb_map[0][1])
        )
        
        intersection_area = self._calc_intersection_area(obj_aabb_map, random_rect)
        
        if intersection_area / obj_area_map > self.remove_threshold:
            return True
        
        return False

    def _calc_intersection_area(self, rect1, rect2):
        """
        @description  : calculate the intersection area of two rectangles
        ---------
        @param  rect1:  Numpy(rect[[x_min, y_min], [x_max, y_max]])
        @param  rect2:  Numpy(rect[[x_min, y_min], [x_max, y_max]])
        -------
        @Returns  : a scalar of intersection area, zero if no intersection
        -------
        """
        x_min = max(rect1[0][0], rect2[0][0])
        y_min = max(rect1[0][1], rect2[0][1])
        x_max = min(rect1[1][0], rect2[1][0])
        y_max = min(rect1[1][1], rect2[1][1])
        
        if (x_max-x_min) < 0.0 or (y_max - y_min) < 0.0:
            return 0.0
        
        return (x_max - x_min) * (y_max - y_min)

    def rect_generator(self, floor):
        """
        @description  : randomly generate a parameterized rectangle
        ---------
        @param  img_size: image shape h*w*c
        @param  sl: the min erasing area
        @param  sh: the max erasing area
        @param  r1: the min aspect ratio, h/w
        -------
        @Returns  : Numpy(rect[[x_min, y_min], [x_max, y_max]])
        -------
        """
        map_size = self.scene.floor_map[floor].shape
        sl = self.erasing_area_ratio[0]
        sh = self.erasing_area_ratio[1]
        r1 = self.min_aspect_ratio
        
        for attempt in range(100):
            area = map_size[0] * map_size[1]

            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if h < map_size[0] and w < map_size[1]:
                x1 = random.randint(0, map_size[0] - h)
                y1 = random.randint(0, map_size[1] - w)
                break

        return np.array([[x1, y1],[x1+h, y1+w]])