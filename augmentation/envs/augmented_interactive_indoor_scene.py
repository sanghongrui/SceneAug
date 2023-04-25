#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   augmented_interactive_indoor_scene.py
@Time    :   2022/07/18 15:07:42
@Author  :   jasonsang 
@Version :   1.0
@Contact :   jasonsang.tongji.edu.cn
'''
# here put the import lib

import logging
log = logging.getLogger(__name__)

import os
import numpy as np
import networkx as nx
import cv2
from igibson import object_states
from baseline_registry import baseline_registry
from igibson.utils.utils import l2_distance, parse_config
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

class AugInteractiveIndoorScene(InteractiveIndoorScene):
    '''
    
    '''
    def __init__(
        self,
        scene_id,
        aug_config_file,
        urdf_file=None,
        urdf_path=None,
        pybullet_filename=None,
        trav_map_resolution=0.1,
        trav_map_erosion=2,
        trav_map_type="with_obj",
        build_graph=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
        texture_randomization=False,
        link_collision_tolerance=0.03,
        object_randomization=False,
        object_randomization_idx=None,
        should_open_all_doors=False,
        load_object_categories=None,
        not_load_object_categories=None,
        load_room_types=None,
        load_room_instances=None,
        seg_map_resolution=0.1,
        scene_source="IG",
        merge_fixed_links=True,
        rendering_params=None,
        include_robots=True,
    ):
        """
        :param scene_id: Scene id
        :param urdf_file: name of urdf file to load (without .urdf), default to ig_dataset/scenes/<scene_id>/urdf/<urdf_file>.urdf
        :param urdf_path: full path of URDF file to load (with .urdf)
        :param pybullet_filename: optional specification of which pybullet file to restore after initialization
        :param trav_map_resolution: traversability map resolution
        :param trav_map_erosion: erosion radius of traversability areas, should be robot footprint radius
        :param trav_map_type: type of traversability map, with_obj | no_obj
        :param build_graph: build connectivity graph
        :param num_waypoints: number of way points returned
        :param waypoint_resolution: resolution of adjacent way points
        :param texture_randomization: whether to randomize material/texture
        :param link_collision_tolerance: tolerance of the percentage of links that cannot be fully extended after object randomization
        :param object_randomization: whether to randomize object
        :param object_randomization_idx: index of a pre-computed object randomization model that guarantees good scene quality
        :param should_open_all_doors: whether to open all doors after episode reset (usually required for navigation tasks)
        :param load_object_categories: only load these object categories into the scene (a list of str)
        :param not_load_object_categories: do not load these object categories into the scene (a list of str)
        :param load_room_types: only load objects in these room types into the scene (a list of str)
        :param load_room_instances: only load objects in these room instances into the scene (a list of str)
        :param seg_map_resolution: room segmentation map resolution
        :param scene_source: source of scene data; among IG, CUBICASA, THREEDFRONT
        :param merge_fixed_links: whether to merge fixed links in pybullet
        :param rendering_params: additional rendering params to be passed into object initializers (e.g. texture scale)
        :param include_robots: whether to also include the robot(s) defined in the scene
        """
        super(AugInteractiveIndoorScene, self).__init__(
            scene_id=scene_id,
            urdf_file=urdf_file,
            urdf_path=urdf_path,
            pybullet_filename=pybullet_filename,
            trav_map_resolution=trav_map_resolution,
            trav_map_erosion=trav_map_erosion,
            trav_map_type=trav_map_type,
            build_graph=build_graph,
            num_waypoints=num_waypoints,
            waypoint_resolution=waypoint_resolution,
            texture_randomization=texture_randomization,
            link_collision_tolerance=link_collision_tolerance,
            object_randomization=object_randomization,
            object_randomization_idx=object_randomization_idx,
            should_open_all_doors=should_open_all_doors,
            load_object_categories=load_object_categories,
            not_load_object_categories=not_load_object_categories,
            load_room_types=load_room_types,
            load_room_instances=load_room_instances,
            seg_map_resolution=seg_map_resolution,
            scene_source=scene_source,
            merge_fixed_links=merge_fixed_links,
            rendering_params=rendering_params,
            include_robots=include_robots,    
        )
        
        #room_instance_random, scene_random, remove_random, random_open
        self.augmentation = self._get_augmentation(aug_config_file)

    def _get_augmentation(self, aug_config_file):
        aug_config = parse_config(aug_config_file)
        augmentation_cls = baseline_registry.get_augmentation_method(
            aug_config.get("name")
        )
        
        augmentation_ins = augmentation_cls(self, aug_config)
        
        return augmentation_ins
        
    def has_path(self, floor, source_world, target_world):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """      
        has_target_node = self.has_node(floor, target_world)
        has_source_node  = self.has_node(floor, source_world)
        
        return has_source_node and has_target_node   
    
    def get_shortest_path(self, floor, source_world, target_world, entire_path=False):
        """
        Get the shortest path from one point to another point.
        If any of the given point is not in the graph, add it to the graph and
        create an edge between it to its closest node.

        :param floor: floor number
        :param source_world: 2D source location in world reference frame (metric)
        :param target_world: 2D target location in world reference frame (metric)
        :param entire_path: whether to return the entire path
        """
        assert self.build_graph, "cannot get shortest path without building the graph"
        source_map = tuple(self.world_to_map(source_world))
        target_map = tuple(self.world_to_map(target_world))

        # add copy to avoid modify floor_graph
        g = self.floor_graph[floor].copy()

        if not g.has_node(target_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - target_map, axis=1))])
            g.add_edge(closest_node, target_map, weight=l2_distance(closest_node, target_map))

        if not g.has_node(source_map):
            nodes = np.array(g.nodes)
            closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - source_map, axis=1))])
            g.add_edge(closest_node, source_map, weight=l2_distance(closest_node, source_map))

        path_map = np.array(nx.astar_path(g, source_map, target_map, heuristic=l2_distance))

        path_world = self.map_to_world(path_map)
        geodesic_distance = np.sum(np.linalg.norm(path_world[1:] - path_world[:-1], axis=1))
        path_world = path_world[:: self.waypoint_interval]

        if not entire_path:
            path_world = path_world[: self.num_waypoints]
            num_remaining_waypoints = self.num_waypoints - path_world.shape[0]
            if num_remaining_waypoints > 0:
                remaining_waypoints = np.tile(target_world, (num_remaining_waypoints, 1))
                path_world = np.concatenate((path_world, remaining_waypoints), axis=0)

        return path_world, geodesic_distance
    
    def get_random_point(self, floor=None):
        """
        @description: Sample a random point on the given floor number. If not given, 
        sample a random floor number. The trav_map_erosion is set to zeros before, 
        so erode the map first according the robot size.
        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        if floor is None:
            floor = self.get_random_floor()
        
        trav_map_erosion = cv2.erode(
            self.floor_map[floor], 
            np.ones((
                self.augmentation.config.get("trav_map_erosion", 3), 
                self.augmentation.config.get("trav_map_erosion", 3),
            ))
        )
        
        trav_space = np.where(trav_map_erosion == 255)
        idx = np.random.randint(0, high=trav_space[0].shape[0])
        xy_map = np.array([trav_space[0][idx], trav_space[1][idx]])
        x, y = self.map_to_world(xy_map)
        z = self.floor_heights[floor]
        return floor, np.array([x, y, z])
    
    def build_graph_without_editing_trav_map(self, floor):
        """
        @description: avoid overwriteing the traversability map loaded before,
        this way the 
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        assert self.build_graph, "cannot update the graph without building the graph"
        
        trav_map_erosion = cv2.erode(
            self.floor_map[floor], 
            np.ones((
                self.augmentation.config.get("trav_map_erosion", 3), 
                self.augmentation.config.get("trav_map_erosion", 3),
            ))
        )
        self.build_trav_graph('', floor, trav_map_erosion)
    
    def is_object_alone_on_floor(self, floor, obj):
        """
        @description:An object is considered to be alone on floor, when its VerticalAdjacency 
        only contain floor whose body_id is 3. And is considered to be a support, 
        when positive_contacts has non-fixed objects.
        ---------
        @param  obj: obj instance
        -------
        @Returns is_support: object is a support for others , 
        @Returns is_alone: object is not below or above others
        -------
        """
        # OnFloor state can not be used directly
        # is_on_floor = obj.states[object_states.OnFloor].get_value()
        scene_floor_id = self.objects_by_category["floors"][floor].get_body_ids()
        # is_on_floor = obj.states[object_states.Touching].get_value(scene_floor)
        vertical_adjacency_list = []

        va = obj.states[object_states.VerticalAdjacency].get_value()
        vertical_adjacency_list = va.negative_neighbors + va.positive_neighbors
        is_alone = vertical_adjacency_list == scene_floor_id
        
        is_support = False
        if len(va.positive_neighbors) != 0:
            contacts = obj.states[object_states.ContactBodies].get_value()
            contacts = [c.bodyUniqueIdB for c in contacts]
            positive_contacts = set(va.positive_neighbors) & set(contacts)
            for id in positive_contacts:
                o = self.objects_by_id[id]
                if not hasattr(o, 'fixed_base') or not o.fixed_base:
                    is_support = True
                    break

        return is_support, is_alone
    
    def get_obj_aabb_map(self, obj):
        """
        @description  : get an object aabb in map space
        ---------
        @param  obj: object instance
        -------
        @Returns  : None
        -------
        """
        aabb = obj.states[object_states.AABB].get_value()
        
        x_min = aabb[0][0]
        x_max = aabb[1][0]
        y_min = aabb[0][1]
        y_max = aabb[1][1]
        
        x_min_map, y_min_map = self.world_to_map(np.array([x_min, y_min]))
        x_max_map, y_max_map = self.world_to_map(np.array([x_max, y_max]))
        
        aabb_map = np.array([[x_min_map, y_min_map], [x_max_map, y_max_map]])       
        
        return aabb_map
    
    def add_object_to_trav_map(self, floor, aabb_map):
        trav_map = self.floor_map[floor]
        trav_map_min = trav_map.min()

        x_min_map = aabb_map[0][0]
        x_max_map = aabb_map[1][0]
        y_min_map = aabb_map[0][1]
        y_max_map = aabb_map[1][1]

        trav_map[x_min_map:x_max_map+1, y_min_map:y_max_map+1] = trav_map_min

        return trav_map

    def remove_object_from_trav_map(self, floor, aabb_map):
        trav_map = self.floor_map[floor]
        trav_map_max = trav_map.max()
        
        x_min_map = aabb_map[0][0]
        x_max_map = aabb_map[1][0]
        y_min_map = aabb_map[0][1]
        y_max_map = aabb_map[1][1]

        trav_map[x_min_map:x_max_map+1, y_min_map:y_max_map+1] = trav_map_max

        return trav_map
    
    def load_task_specified_scene(self, env, task_specified_scene_config, floor):
        """
        @description: load task specified scene where some object are mannually placed.
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """        
        if task_specified_scene_config is None:
            return
        
        for name, attrs in task_specified_scene_config.items():
            obj = self.objects_by_name[name]
            if obj.category == 'door':
                obj.force_wakeup()
                for body_id in obj.get_body_ids():
                    self.open_one_obj(body_id, mode=attrs['mode'])
            else:
                origial_aabb_map = self.get_obj_aabb_map(obj)
                if attrs['aabb_remove'] is True:
                    self.remove_object_from_trav_map(floor, origial_aabb_map)
                
                env.land(obj, attrs['pos'], None)
                # obj.set_position(attrs['pos'])
                
                new_aabb_map = self.get_obj_aabb_map(obj)
                self.add_object_to_trav_map(floor, new_aabb_map)
        
        # update floor_graph
        self.build_graph_without_editing_trav_map(floor)