#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   augmented_env.py
@Time    :   2022/05/27 12:03:47
@Author  :   jasonsang
@Version :   1.0
@Contact :   jasonsang@tongji.edu.cn
'''

import pybullet as p

from igibson.object_states import AABB
from igibson.robots import REGISTERED_ROBOTS
from igibson.robots.robot_base import BaseRobot
from igibson.envs.igibson_env import iGibsonEnv
from igibson.object_states.utils import detect_closeness
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.utils.utils import parse_config, l2_distance
from augmentation.envs.augmented_interactive_nav_task import AugmentedInteractiveNavTask
from augmentation.envs.augmented_interactive_indoor_scene import AugInteractiveIndoorScene

class AugmentedEnv(iGibsonEnv):
    """
    iGibson Environment (OpenAI Gym interface).
    """
    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        automatic_reset=False,
        use_pb_gui=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, headless_tensor, gui_interactive, gui_non_interactive, vr
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param rendering_settings: rendering_settings to override the default one
        :param vr_settings: vr_settings to override the default one
        :param device_idx: which GPU to run the simulation and rendering on
        :param automatic_reset: whether to automatic reset after an episode finishes
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        config_data =  parse_config(config_file)

        rendering_settings = MeshRendererSettings(
            enable_shadow=config_data.get("enable_shadow", False),
            enable_pbr=config_data.get("enable_pbr", False),
            msaa=False,
            texture_scale=config_data.get("texture_scale", 1.0),
            blend_highlight=config_data.get("blend_highlight", True),
            optimized=config_data.get("optimized_renderer", True),
            load_textures=config_data.get("load_texture", True),
            hide_robot=config_data.get("hide_robot", True),
        )
        #set to zero, use augmentation config instead. This will keep an uneroded floor map.
        config_data['trav_map_erosion'] = 0
    
        super(AugmentedEnv, self).__init__(
            config_file=config_data,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            rendering_settings=rendering_settings,
            device_idx=device_idx,
            automatic_reset=automatic_reset,
            use_pb_gui=use_pb_gui,
        )

    def load_task_setup(self):
        """
        Load task setup.
        """
        if self.config["task"] == "augmented_interactive_nav_task":
            # load DummyTask first
            self.config.pop("task")    
            super(AugmentedEnv, self).load_task_setup()
            
            # load InteractiveNavWithObjectRandomTask
            self.task = AugmentedInteractiveNavTask(self)
        else:
            # task
            super(AugmentedEnv, self).load_task_setup()

    def load_scene(self):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """    
        if self.config["scene"] != "aug_igibson":
            super(AugmentedEnv, self).load()
            print("Not an aug_igibson, use igibson.")
            return
            
        urdf_file = self.config.get("urdf_file", None)
        if urdf_file is None and not self.config.get("online_sampling", True):
            urdf_file = "{}_task_{}_{}_{}_fixed_furniture".format(
                self.config["scene_id"],
                self.config["task"],
                self.config["task_id"],
                self.config["instance_id"],
            )
        include_robots = self.config.get("include_robots", True)
        scene = AugInteractiveIndoorScene(
            self.config["scene_id"],
            aug_config_file=self.config.get("aug_config_file"),
            urdf_file=urdf_file,
            waypoint_resolution=self.config.get("waypoint_resolution", 0.2),
            num_waypoints=self.config.get("num_waypoints", 10),
            build_graph=self.config.get("build_graph", False),
            trav_map_resolution=self.config.get("trav_map_resolution", 0.1),
            trav_map_erosion=self.config.get("trav_map_erosion", 2),
            trav_map_type=self.config.get("trav_map_type", "with_obj"),
            texture_randomization=self.texture_randomization_freq is not None,
            object_randomization=self.object_randomization_freq is not None,
            object_randomization_idx=self.object_randomization_idx,
            should_open_all_doors=self.config.get("should_open_all_doors", False),
            load_object_categories=self.config.get("load_object_categories", None),
            not_load_object_categories=self.config.get("not_load_object_categories", None),
            load_room_types=self.config.get("load_room_types", None),
            load_room_instances=self.config.get("load_room_instances", None),
            merge_fixed_links=self.config.get("merge_fixed_links", True)
            and not self.config.get("online_sampling", False),
            include_robots=include_robots,
        )
        # TODO: Unify the function import_scene and take out of the if-else clauses.
        first_n = self.config.get("_set_first_n_objects", -1)
        if first_n != -1:
            scene._set_first_n_objects(first_n)
    
        self.simulator.import_scene(scene)

        self.scene = scene
    
    def load_robots(self):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        if self.config["scene"] != "aug_igibson":
            return
        
        # Get robot config
        robot_config = self.config["robot"]

        # If no robot has been imported from the scene
        if len(self.scene.robots) == 0:
            # Get corresponding robot class
            robot_name = robot_config.pop("name")
            assert robot_name in REGISTERED_ROBOTS, \
                "Got invalid robot to instantiate: {}".format(robot_name)
            robot = REGISTERED_ROBOTS[robot_name](**robot_config)

            self.simulator.import_object(robot)

            # The scene might contain cached agent pose
            # By default, we load the agent pose that matches the robot name (e.g. Fetch, BehaviorRobot)
            # The user can also specify "agent_pose" in the config file to use the cached agent pose for any robot
            # For example, the user can load a BehaviorRobot and place it at Fetch's agent pose
            agent_pose_name = self.config.get("agent_pose", robot_name)
            if isinstance(self.scene, AugInteractiveIndoorScene) and agent_pose_name in self.scene.agent_poses:
                pos, orn = self.scene.agent_poses[agent_pose_name]

                if agent_pose_name != robot_name:
                    # Need to change the z-pos - assume we always want to place the robot bottom at z = 0
                    lower, _ = robot.states[AABB].get_value()
                    pos[2] = -lower[2]

                robot.set_position_orientation(pos, orn)

                if any(
                    detect_closeness(
                        bid, 
                        exclude_bodyB=self.scene.objects_by_category["floors"][0].get_body_ids(),
                        distance=0.01
                    ) for bid in robot.get_body_ids()
                ):
                    log.warning("Robot's cached initial pose has collisions.")

        self.robots = self.scene.robots
     
    def load(self):
        """
        @description:
        ---------
        @param  :
        -------
        @Returns: None
        -------
        """
        self.load_scene()
        self.load_robots()
        
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()
    
    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation.

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.reset()
            obj.keep_still()

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if any(len(p.getContactPoints(bodyA=body_id)) > 0 for body_id in obj.get_body_ids()):
                land_success = True
                break

        if is_robot:
            obj.reset()
            obj.keep_still()
    
    def populate_info(self, info):
        super().populate_info(info)

        info['euclidean_distance'] = l2_distance(self.task.initial_pos, self.task.target_pos)
        _, info['geodesic_distance'] = self.scene.get_shortest_path(
            self.task.floor_num, 
            self.task.initial_pos[:2], 
            self.task.target_pos[:2], 
            entire_path=False
        )
        info['floor_map'] = dict()
        info['floor_map']['map_agent_pos'] = \
            self.scene.world_to_map(self.robots[0].get_position()[:2])
        info['floor_map']['agent_ori'] = self.robots[0].get_orientation()
        info['floor_map']['map_agent_inital_pos'] = \
            self.scene.world_to_map(self.task.initial_pos[:2])
        info['floor_map']['map_agent_target_pos'] = \
            self.scene.world_to_map(self.task.target_pos[:2])
        info['floor_map']['map'] = self.scene.floor_map[self.task.floor_num].copy()
        
if __name__ == "__main__":
    import argparse
    import logging
    import time
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()
    args.config = "configs/igibson_config/fetch_interactive_nav.yaml"

    env = AugmentedEnv(
        config_file=args.config, 
        mode=args.mode, 
        action_timestep=1.0 / 10.0, 
        physics_timestep=1.0 / 40.0
    )

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.task_action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
