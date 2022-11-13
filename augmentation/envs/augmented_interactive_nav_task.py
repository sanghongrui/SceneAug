#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   interactive_nav_object_random_task.py
@Time    :   2022/05/30 22:20:08
@Author  :   jasonsang
@Version :   1.0
@Contact :   jasonsang@tongji.edu.cn
'''
import os
import numpy as np
import pybullet as p
from igibson.utils.utils import parse_config
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.reward_functions.potential_reward import PotentialReward


class AugmentedInteractiveNavTask(PointNavRandomTask):
    """
    Interactive Navigation with Scene Augmentation Task
    The goal is to navigate to a random goal position, 
    in the presence of randomly placed indoor interactive objects.
    """
    
    def __init__(self, env):
        super(AugmentedInteractiveNavTask, self).__init__(env)
        
        self.reward_functions = [
            PotentialReward(self.config),
            PointGoalReward(self.config),
        ]
                
        self.init_floor_map = None
        self.init_floor_graph = None
        self.task_specified_scene_config = None
        
        self.store_init_scene_floor_map(env)
        
        if self.config.get('load_task_specified_scene') is True:
            self.task_specified_scene_config = parse_config(
                os.path.join(
                    self.config.get('task_specified_scene_config_path'), 
                    env.scene.scene_id + '.yaml'
                )
            )
        
    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        super(AugmentedInteractiveNavTask, self).reset_scene(env)
        
        self.reset_scene_floor_map(env)

        #load init_scene
        env.scene.load_task_specified_scene(env, self.task_specified_scene_config, self.floor_num)

        env.scene.augmentation.run(env)

        # add aabb in env.scene.object_states
        for obj_name in env.scene.object_states:
            obj = env.scene.objects_by_name[obj_name]  
            if obj.category in ["agent"]:
                continue
               
            env.scene.object_states[obj_name]['aabb_map'] = env.scene.get_obj_aabb_map(obj)
    
    def reset_variables(self, env):
        super(AugmentedInteractiveNavTask, self).reset_variables(env)
        
        self.obj_disp_mass = 0.0
        self.ext_force_norm = 0.0
        self.obj_pos = self.get_obj_pos(env)
        self.obj_mass = self.get_obj_mass(env)
        self.obj_body_ids = self.get_obj_body_ids(env)
    
    def get_obj_pos(self, env):
        # Get object position for all scene objs
        obj_pos = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings', 'agent']:
                continue
            body_id = obj.get_body_ids()[0]
            if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
                pos, _ = p.getLinkState(body_id, 0)[0:2]
            else:
                pos, _ = p.getBasePositionAndOrientation(body_id)
            obj_pos.append(pos)
            
        obj_pos = np.array(obj_pos)
        return obj_pos
    
    def get_obj_mass(self, env):
        # Get object mass for all scene objs
        obj_mass = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings', 'agent']:
                continue
            body_id = obj.get_body_ids()[0]
            if p.getBodyInfo(body_id)[0].decode('utf-8') == 'world':
                link_id = 0
            else:
                link_id = -1
            mass = p.getDynamicsInfo(body_id, link_id)[0]
            obj_mass.append(mass)
            
        obj_mass = np.array(obj_mass)
        return obj_mass
    
    def get_obj_body_ids(self, env):
        # Get object body id for all scene objs 
        body_ids = []
        for _, obj in env.scene.objects_by_name.items():
            if obj.category in ['walls', 'floors', 'ceilings', 'agent']:
                continue
            body_ids.append(obj.get_body_ids()[0])

        return body_ids
    
    def store_init_scene_floor_map(self, env):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        self.init_floor_map = [m.copy() for m in env.scene.floor_map]
        self.init_floor_graph = [g.copy() for g in env.scene.floor_graph]
        
    def reset_scene_floor_map(self, env):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        env.scene.floor_map = [m.copy() for m in self.init_floor_map]
        env.scene.floor_graph = [g.copy() for g in self.init_floor_graph]
    
    def has_path(self, env):
        """
        @description: whether agent has path from current position to target position
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """      
        floor = self.floor_num
        source_world = env.robots[0].get_position()[:2]
        target_world = self.target_pos[:2]
        
        has_path = env.scene.has_path(floor, source_world, target_world)
        
        return has_path

    def step(self, env):
        """
        @description: update floor map after simulator step
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        # udpate floor_map first
        if self.config.get('path_reward'):
            self.before_step(env)
        
        super(AugmentedInteractiveNavTask, self).step(env)
        
        # Accumulate the external force that the robot exerts to the env
        ext_force = [col[9] * np.array(col[7]) for col in env.collision_links]
        net_force = np.sum(ext_force, axis=0)  # sum of all forces
        self.ext_force_norm += np.linalg.norm(net_force)

        # Accumulate the object displacement (scaled by mass) by the robot
        collision_objects = set([col[2] for col in env.collision_links])
        new_obj_pos = self.get_obj_pos(env)
        obj_disp_mass = 0.0
        for obj_id in collision_objects:
            # e.g. collide with walls, floors, ceilings
            if obj_id not in self.obj_body_ids:
                continue
            idx = self.obj_body_ids.index(obj_id)
            obj_dist = np.linalg.norm(self.obj_pos[idx] - new_obj_pos[idx])
            obj_disp_mass += obj_dist * self.obj_mass[idx]
        
        self.obj_disp_mass += obj_disp_mass
        self.obj_pos = new_obj_pos
    
    def before_step(self, env):
        """
        @description: 
            after env.step, objects' position may be changed, so we use collision_links to
        index these objects and update scene floor_map and floor_graph.
            #1. get object instance from colision_links
            #2. get new object aabb
            #3. update date floor map with new objects' aabb
            #4. udpate object states
            #5. env.scene.build_trav_graph
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        collision_body_b_ids = list(set(c[2] for c in env.collision_links))
        for c in collision_body_b_ids:
            if c not in env.scene.objects_by_id.keys():
                continue
            
            obj = env.scene.objects_by_id[c]
            if obj.category not in self.config.get("floor_map_update_inlcude_categories"):
                continue
            
            origial_aabb_map = env.scene.object_states[obj.name]['aabb_map']
            new_aabb_map = env.scene.get_obj_aabb_map(obj)
            
            #remove obj from last step trav_map
            env.scene.remove_object_from_trav_map(self.floor_num, origial_aabb_map)
            # add obj to current step trav_map
            env.scene.add_object_to_trav_map(self.floor_num, new_aabb_map)
            #udpate object state   
            env.scene.object_states[obj.name]['aabb_map'] = new_aabb_map
        
        #update floor_graph without modification to trav_map
        env.scene.build_graph_without_editing_trav_map(self.floor_num)
    
    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        @description: Aggreate termination conditions and fill info
        ---------
        @param  :
        -------
        @Returns: done, info
        -------
        """
        done, info = super(AugmentedInteractiveNavTask, self).get_termination(
            env, collision_links, action, info)
        
        if done:
            robot_mass = p.getDynamicsInfo(env.robots[0].get_body_ids()[0], -1)[0]
            self.robot_disp_mass = self.path_length * robot_mass
            info['kinematic_disturbance'] = self.robot_disp_mass / \
                (self.robot_disp_mass + self.obj_disp_mass)
            self.robot_gravity = env.current_step * robot_mass * 9.8
            info['dynamic_disturbance'] = self.robot_gravity / \
                (self.robot_gravity + self.ext_force_norm)
            info['effort_efficiency'] = (info['kinematic_disturbance'] +
                                         info['dynamic_disturbance']) / 2.0
            info['path_efficiency'] = info['spl']
            effort_path_balance = self.config.get('effort_path_balance', 0.5)
            info['int_nav_score'] = effort_path_balance * info['path_efficiency'] + \
                (1.0 - effort_path_balance) * info['effort_efficiency']
        else:
            info['kinematic_disturbance'] = 0.0
            info['dynamic_disturbance'] = 0.0
            info['effort_efficiency'] = 0.0
            info['path_efficiency'] = 0.0
            info['int_nav_score'] = 0.0

        return done, info
                
        
                
        