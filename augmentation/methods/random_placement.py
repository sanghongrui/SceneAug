#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   random_placement.py
@Time    :   2022/07/15 15:48:14
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib
import pybullet as p
import numpy as np
import cv2

from typing import List, Dict, Tuple
from abc import abstractmethod
# from igibson import object_states
from igibson.utils.utils import restoreState
from common.baseline_registry import baseline_registry
from augmentation.methods.random_base import Augmentation_Base
from augmentation.envs.augmented_env import AugmentedEnv

class Random_Placement(Augmentation_Base):
    '''
    Base object for random placement, e.g., room_instance_random and scene_random
    '''
    def __init__(
        self, 
        scene,
        config,
    ) -> None:
        super(Random_Placement, self).__init__(scene, config)
        
        self._name = 'random_placement'
        
        # include_categories: List[str]
        # exclude_dict: Dict[str, List[str]]
        self.include_categories = self.config.get('include_categories')
        self.exclude_dict = self.config.get('exclude_dict')
        self.erosion_factor = self.config.get('erosion_factor', 0.6)
        
    def augmentation(self, env):
        self.random_objects_by_category(env)
    
    def random_objects_by_category(self, env:AugmentedEnv):
        """
        @description: Reset the poses of room_random object categories to have no 
        @description: collisions with the scene or the robot
        ---------
        @param  category: object categories list
        -------
        @Returns: None
        -------
        """
        for c in self.include_categories:
            if c not in self.scene.objects_by_category.keys():
                continue

            objs = self.scene.objects_by_category[c]
            self.random_objects(env, objs)

            print("Finished randomizing %s category" % c)
    
    def random_objects(self, env:AugmentedEnv, room_random_objects):
        """
        @description: Reset the poses of room objects in self.include_categories and not in 
        self.excluded_obj without collisions with the scene or the robot. We also exclude objects
        below others and do not edit trav_map when it is above others.
        ---------
        @param  room_random_objects: List[obj], list of object instance
        -------
        @Returns: None
        -------
        """
        max_trials = 100
        floor = env.task.floor_num

        for obj in room_random_objects:
            is_exclude = self._is_in_exclude_dict(obj.name)
            if is_exclude:
                print("WARNING: skip randomlizing %s, it is in exclude dict." %obj.name)
                continue
            
            below_others, is_alone = env.scene.is_object_alone_on_floor(floor, obj)
            if below_others:
                print("WARNING: skip randomlizing %s, it is below others." %obj.name)
                continue
            
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                # get random point
                # _, pos = self.get_random_point_by_room_instance(env, obj)
                _, pos = self.get_random_point_for_obj(env, obj)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])            
                sample_success =  pos is not None
                if not sample_success:
                    break

                reset_success = env.test_valid_position(obj, pos, orn)
                restoreState(state_id)
                if reset_success:
                    break

            if not sample_success:
                print("WARNING: No available space for %s." %obj.name)
            else:
                if not reset_success:
                    print("WARNING: Failed to random %s without collision" %obj.name)

                if reset_success:
                    original_aabb_map = self.scene.get_obj_aabb_map(obj)

                    env.land(obj, pos, orn)

                    if is_alone:
                        self.scene.remove_object_from_trav_map(floor, original_aabb_map)

                    self.scene.add_object_to_trav_map(floor, self.scene.get_obj_aabb_map(obj))

            p.removeState(state_id)    
    
    def _is_in_exclude_dict(self, obj_name):
        """
        @description: whether an obj is in self.exclude_dict
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        exclude_objects = self.exclude_dict.get(self.scene.scene_id)
        if exclude_objects is None:
            return False
        
        if obj_name in exclude_objects:
            return True
        
        return False

    # def _is_object_alone_on_floor(self, floor, obj):
    #     """
    #     @description:An object is considered to be alone on floor, when its VerticalAdjacency 
    #     only contain floor whose body_id is 3
    #     ---------
    #     @param  obj: obj instance
    #     -------
    #     @Returns below_other: object is below others , 
    #     @Returns is_alone: object is not below or above others
    #     -------
    #     """
    #     # OnFloor state can not be used directly
    #     # is_on_floor = obj.states[object_states.OnFloor].get_value()
    #     scene_floor_id = self.scene.objects_by_category["floors"][floor].get_body_ids()
    #     # is_on_floor = obj.states[object_states.Touching].get_value(scene_floor)
    #     vertical_adjacency_list = []

    #     va = obj.states[object_states.VerticalAdjacency].get_value()
    #     vertical_adjacency_list = va.negative_neighbors + va.positive_neighbors

    #     below_others = len(va.positive_neighbors) != 0
    #     is_alone = vertical_adjacency_list == scene_floor_id

    #     return below_others, is_alone
    
    @abstractmethod
    def get_random_point_for_obj(self, env:AugmentedEnv, obj) -> Tuple[int, np.ndarray]:
        """
        @description: method specified random point sampler
        ---------
        @param  obj:  obj instance
        -------
        @Returns floor: floor num
        @Returns pos: position in world space
        -------
        """ 
        raise NotImplementedError()
  

@baseline_registry.register_augmentation_method(name="scene_random")
class Scene_Random(Random_Placement):
    '''
    Randomly place objects in scene.
    '''
    def __init__(
        self, 
        scene,
        config
    ) -> None:
        # super().__init__(scene, config)
        super(Scene_Random, self).__init__(scene, config)
        
        self._name = 'scene_random'
        
    def get_random_point_for_obj(self, env:AugmentedEnv, object_instance):
        """
        Sample a random point in scene for object instance
        :param object_instance: object instance (e.g. chair_19)
        :return: floor (always 0), a randomly sampled point in [x, y, z]
        """
        """
        @description: Sample a random point in scene for object instance
        ---------
        @param object_instance: object instance (e.g. chair_19)
        -------
        @Returns: floor (always 0), a randomly sampled point in world space
        -------
        """
        floor = env.task.floor_num
        
        object_max_map_size = int(
            max(object_instance.avg_obj_dims['size'][0:-1])
            * self.erosion_factor    # erosion scale factor
            / self.scene.trav_map_resolution
        )

        kernel = np.ones((object_max_map_size, object_max_map_size), np.uint8)
        erosion_map = cv2.erode(self.scene.floor_map[floor], kernel)

        erosion_map_max = erosion_map.max()
        # index of traversable space in room instance 
        trav_space = np.where(erosion_map == erosion_map_max)
        if trav_space[0].shape[0] == 0:
            return None, None

        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        
        x, y = self.scene.map_to_world(xy_map)
        z = self.scene.floor_heights[floor]
        
        return floor, np.array([x, y, z])
    

@baseline_registry.register_augmentation_method(name="room_instance_random")        
class Room_Instance_Random(Random_Placement):
    ''''
    Randomly place objects in its originaln room instance.
    '''
    def __init__(
        self, 
        scene,
        config
    ) -> None:
        super(Room_Instance_Random, self).__init__(scene, config)
        
        self._name = 'room_instance_random'
        
    def get_random_point_for_obj(self, env:AugmentedEnv, object_instance):
        """
        @description: Sample a random point for object instance by object's room instance
        ---------
        @param object_instance: object instance (e.g. chair_19)
        -------
        @Returns: floor (always 0), a randomly sampled point in [x, y, z]
        -------
        """
        floor = env.task.floor_num
        
        object_max_map_size = int(
            max(object_instance.avg_obj_dims['size'][0:-1]) # max(x, y)
            * self.erosion_factor    # erosion scale factor, bigger than half of obj size
            / self.scene.trav_map_resolution
        )
        room_instance_name = object_instance.in_rooms[0]
        room_instance_id = self.scene.room_ins_name_to_ins_id[room_instance_name]

        kernel = np.ones((object_max_map_size, object_max_map_size), np.uint8)
        erosion_map = cv2.erode(self.scene.floor_map[floor], kernel)

        erosion_map_halfvalue = erosion_map//2
        erosion_map_halfvalue_max = erosion_map_halfvalue.max()
        seg_map = self.scene.room_ins_map
        trav_seg_map = erosion_map_halfvalue + seg_map

        # index of traversable space in room instance 
        trav_space_room = np.where(trav_seg_map == erosion_map_halfvalue_max+room_instance_id)
        if trav_space_room[0].shape[0] == 0:
            return None, None

        idx = np.random.randint(0, high=trav_space_room[0].shape[0])
        xy_map = np.array([trav_space_room[0][idx], trav_space_room[1][idx]])
        
        x, y = self.scene.map_to_world(xy_map)
        z = self.scene.floor_heights[floor]
        
        return floor, np.array([x, y, z])     