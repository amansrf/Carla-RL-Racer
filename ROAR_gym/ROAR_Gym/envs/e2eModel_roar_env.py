try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict
from gym.spaces import Discrete, Box
import cv2
import wandb
import skimage.measure
import random

# imports for reading and writing json config files
from ROAR_gym.utility import json_read_write, next_spawn_point

# Load spawn parameters from the ppo_configuration file
from ROAR_gym.configurations.ppo_configuration import spawn_params

mode='newb'
FRAME_STACK = 4
CONFIG = {
    "x_res": 84,
    "y_res": 84
}
WALL_MAGNITUDES = [1,8]

class ROARppoEnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # low=np.array([-2.5, -3.0, 1.0])
        # high=np.array([-0.5, 3.0, 3.0])
        # low=np.array([-4.5, -7.0, 8.0])
        # high=np.array([-2.5, 7.0, 9.0])
        # low=np.array([-3.5, -10.0, 8.0,-7.0])
        # high=np.array([-1.5, 10.0, 10.0,3.0])
        # low=np.array([-7, -10.0])
        # high=np.array([-1.5, 10.0])
        low=np.array([-1, -1.0])
        high=np.array([1, 1.0])
        # low=np.array([-1, -2.0])
        # high=np.array([1, 2.0])
        self.mode=mode
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # self.observation_space = Box(0, 1, shape=(FRAME_STACK,2, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        if mode=='baseline':
            self.observation_space = Box(0, 1, shape=(4, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        else:
            self.observation_space = Box(0, 1, shape=(3, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        self.prev_speed = 0
        self.prev_cross_reward = 0
        self.crash_check = False
        self.ep_rewards = 0
        self.frame_reward = 0
        self.highscore = -1000
        self.highest_chkpt = 0
        self.speeds = []
        self.prev_int_counter = 0
        self.steps=0
        self.largest_steps=0
        self.highspeed=0
        self.complete_loop=False
        self.his_checkpoint=[]
        self.his_score=[]
        self.time_to_waypoint_ratio = 5.0 #0.75
        self.fps = 32
        self.death_line_dis = 2
        ## used to check if stalled
        self.stopped_counter = 0
        self.stopped_max_count =1
        # used to track episode highspeed
        self.speed = 0
        self.current_hs = 0
        # used to check laptime
        if self.carla_runner.world is not None:
            self.last_sim_time = self.carla_runner.world.hud.simulation_time
        else:
            self.last_sim_time = 0
        self.sim_lap_time = 0

        self.deadzone_trigger = True
        self.deadzone_level = 0.001
        self.overlap = False
        self.last_num_collision=0
        self.prev_action= np.array([0, 0, 0])
        self.prev_collisoin=[]

        # Spawn initializations
        # TODO: This is a hacky fix because the reset function seems to be called on init as well.
        if spawn_params["dynamic_spawn"] :
            if spawn_params["dynamic_type"] == "linear forward":
                self.agent_config.spawn_point_id = spawn_params["init_spawn_pt"] - 1
            elif spawn_params["dynamic_type"] == "linear backward":
                self.agent_config.spawn_point_id = spawn_params["init_spawn_pt"] + 1
            elif spawn_params["dynamic_type"] == "uniform random":
                self.agent_config.spawn_point_id = np.random.randint(low=1, high=12)
            else:
                self.agent_config.spawn_point_id = spawn_params["init_spawn_pt"]
        else:
            self.agent_config.spawn_point_id = spawn_params["init_spawn_pt"]

        self.agent.spawn_counter = spawn_params["spawn_int_map"][self.agent_config.spawn_point_id]
        print("#########################\n",self.agent.spawn_counter)


    # def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
    #     obs = []
    #     rewards = []
    #     self.steps += 1

    #     action = action.reshape((-1))
    #     # throttle = (action[0] + 0.5) / 2 + 1
    #     throttle = (action[0] + 2.5) / 2 + 1
    #     # braking = (action[2] - 1.0) / 2
    #     braking = (action[2] - 8.0) / 2

    #     # full_throttle_thre = 0.6
    #     # non_braking_thre = 0.4
    #     # throttle = min(1, throttle_check / full_throttle_thre)
    #     # braking = max(0, (braking_check - non_braking_thre) / (1 - non_braking_thre))

    #     # check = (action[0] + 0.5) / 2 + 1
    #     # if check > 0.5:
    #     #     throttle = 0.7
    #     #     braking = 0
    #     # else:
    #     #     throttle = 0
    #     #     braking = 0.8

    #     # steering = action[1]/3
    #     steering = action[1]/7

    #     if self.deadzone_trigger and abs(steering) < self.deadzone_level:
    #         steering = 0.0


    #     self.agent.kwargs["control"] = VehicleControl(throttle=throttle,
    #                                                     steering=steering,
    #                                                     braking=braking)

    #     ob, reward, is_done, info = super(ROARppoEnvE2E, self).step(action)


    #     obs.append(ob)
    #     rewards.append(reward)

    #     self.render()
    #     self.frame_reward = sum(rewards)
    #     self.ep_rewards += sum(rewards)

    #     self.speed = self.agent.vehicle.get_speed(self.agent.vehicle)
    #     if self.speed > self.current_hs:
    #         self.current_hs = self.speed

    #     if is_done:
    #         self.wandb_logger()
    #         self.crash_check = False
    #         self.update_highscore()
    #     return np.array(obs), self.frame_reward, self._terminal(), self._get_info()
    
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs = []
        rewards = []
        self.steps += 1
        self.speed = self.agent.vehicle.get_speed(self.agent.vehicle)
        if self.speed > self.current_hs:
            self.current_hs = self.speed

        action = action.reshape((-1))
        self.prev_action=np.array(action)
        # decision=action[3]
        # # if decision<-2: #turning mode
        # #     throttle = (action[0] + 1.5) / 2 + 1
        # #     # if action[1]>=5:
        # #     #     action[1]=5.0
        # #     # if action[1]<=-5:
        # #     #     action[1]=-5.0
        # #     steering = action[1]/10
        # #     braking=(action[2]-8)/2
            
        # # else: #speeding mode
        # #     throttle = (action[0] + 1.5) / 2 + 1
        # #     steering = action[1]/80
        # #     if action[2]<=9:
        # #         action[2]=9.0
        # #     braking=(action[2]-9)/4
        
        # throttle = (action[0] + 1.5) / 2 + 1
        # steering = action[1]/10/((decision+7)/10*7+1)
        # # if action[2]<=8+(decision+7)/10:
        # #         action[2]=8+(decision+7)/10
        # # braking=(action[2]-8-(decision+7)/10)/2/((decision+7)/10*2+1)
        # braking=(action[2]-8)/2

        # decision=action[0]+6
        # if decision<0:
        #     throttle=0
        #     braking=abs(decision)
        # else:
        #     throttle=decision/4.5
        #     braking=0
        # steering=action[1]/10

        # decision=action[0]+1
        # if decision<0:
        #     throttle=0
        #     braking=abs(decision)*2
        # else:
        #     throttle=decision*2
        #     braking=0
        # steering=action[1]/2

        decision=action[0]
        if decision<0:
            throttle=0
            braking=abs(decision)
        else:
            if self.speed>40:
                throttle=0 
            else:
                throttle=decision
            braking=0
        steering=action[1]
            
        # throttle = (action[0] + 4.5) / 2
        # braking = (action[2] - 8.0)
        # steering = action[1]/7

        self.agent.kwargs["control"] = VehicleControl(throttle=throttle,
                                                        steering=steering,
                                                        braking=braking)
        #print(self.agent.kwargs)

        ob, reward, is_done, info = super(ROARppoEnvE2E, self).step(action)


        obs.append(ob)
        rewards.append(reward)

        self.render()
        self.frame_reward = sum(rewards)
        self.ep_rewards += sum(rewards)



        if is_done:
            self.wandb_logger()
            # self.crash_check = False
            self.update_highscore()
        return np.array(obs), self.frame_reward, self._terminal(), self._get_info()

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["Current HIGHSCORE"] = self.highscore
        info_dict["Furthest Checkpoint"] = self.highest_chkpt*self.agent.interval
        info_dict["episode reward"] = self.ep_rewards
        info_dict["checkpoints"] = self.agent.int_counter*self.agent.interval
        info_dict["reward"] = self.frame_reward
        info_dict["largest_steps"] = self.largest_steps
        info_dict["current_hs"] = self.current_hs
        info_dict["highest_speed"] = self.highspeed
        info_dict["complete_state"]=self.complete_loop
        info_dict["avg10_checkpoints"]=np.average(self.his_checkpoint)
        info_dict["avg10_score"]=np.average(self.his_score)
        # info_dict["throttle"] = action[0]
        # info_dict["steering"] = action[1]
        # info_dict["braking"] = action[2]
        return info_dict

    def update_highscore(self):
        if self.ep_rewards > self.highscore:
            self.highscore = self.ep_rewards
        if self.agent.int_counter > self.highest_chkpt:
            self.highest_chkpt = self.agent.int_counter
        if self.current_hs > self.highspeed:
            self.highspeed = self.current_hs
        self.current_hs = 0

        if self.carla_runner.world is not None:
            current_time = self.carla_runner.world.hud.simulation_time
            if self.agent.int_counter * self.agent.interval < 1000000:
                self.sim_lap_time = 400
            else:
                self.sim_lap_time = current_time - self.last_sim_time
            self.last_sim_time = current_time
        else:
            self.sim_lap_time = 0
            self.last_sim_time = 0
        return

    def _terminal(self) -> bool:
        if self.stopped_counter >= self.stopped_max_count:
            print("what")
            return True
        if self.agent.off_reward:
            print("off")
            return True
        # if self.carla_runner.get_num_collision() > self.max_collision_allowed:
        #     print("man")
        #     return True
        # if self.carla_runner.world.collision_sensor.history[-1][-1]>=10000:
        #     print('crash')
        #     return True
        if self.crash_check: #elif self.overlap:
            print("pls")
            return True
        # elif self.overlap:
        #     print("overlap--------------------------------------------------------------")
        #     return True
        if self.agent.finish_loop:
            print("halp")
            self.complete_loop=True
            return True
        return False

    def get_reward(self) -> float:
        reward = 0

        
        # if abs(self.agent.vehicle.control.steering) <= 0.1:
        #     reward += 0.1

        if self.crash_check:
            print("no reward")
            return 0

        if self.agent.cross_reward > self.prev_cross_reward:
            if self.agent.off_reward:
                # reward -= 100
                print('off')
                self.crash_check = True
            else:
                reward += (self.agent.cross_reward - self.prev_cross_reward)*self.agent.interval*self.time_to_waypoint_ratio
        # print(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y,self.agent.vehicle.velocity.z)
        # print(self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)].dis(self.agent.vehicle.transform.location.x,self.agent.vehicle.transform.location.z),'----------------------------------')
        # if not (self.agent.bbox_list[max(self.agent.int_counter - self.death_line_dis,1)].has_crossed(self.agent.vehicle.transform))[0]:
        #     reward -= 100
        #     # for i in range(20):
        #     #     print((self.agent.bbox_list[i].has_crossed(self.agent.vehicle.transform))[0])
        #     print('drive back')
        #     self.crash_check = True
        if self.carla_runner.get_num_collision() > self.last_num_collision:
            reward -= self.carla_runner.world.collision_sensor.history[-1][-1]/1000
            self.prev_collisoin=self.carla_runner.world.collision_sensor.history[-1][-1]/1000
            self.last_num_collision=self.carla_runner.get_num_collision()
            print(f'collision number: {self.carla_runner.get_num_collision()}------------------{self.carla_runner.world.collision_sensor.history[-1][-1]}')
            if self.carla_runner.world.collision_sensor.history[-1][-1]>10000:
                reward -= 100
                self.crash_check = True
        else:
            self.prev_collisoin=0
        # print(f'{self.carla_runner.get_num_collision()}')
        # if self.agent.int_counter > 1 and self.agent.vehicle.get_speed(self.agent.vehicle) < 2:
        # print(self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)].get_directional_velocity(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y),'----------------------------------')
        # print(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y)
        if self.steps>100 and self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)].get_directional_velocity(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y) < 0.1:
            self.stopped_counter += 1
            if self.stopped_counter >= self.stopped_max_count:
                reward -= 100
                print('stopped')
                self.crash_check = True
        else:
            self.stopped_counter =0

        # log prev info for next reward computation
        self.prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self.prev_cross_reward = self.agent.cross_reward
        return reward/30

    def _get_obs(self) -> np.ndarray:
        if mode=='baseline':
            index_from=(self.agent.int_counter%len(self.agent.bbox_list))
            if index_from+10<=len(self.agent.bbox_list):
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:index_from+10]
            else:
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:]+self.agent.bbox_list[:index_from+10-len(self.agent.bbox_list)]
            assert(len(next_bbox_list)==10)
            map_list,overlap = self.agent.occupancy_map.get_map_9(transform_list=self.agent.vt_queue,
                                                                  speed_list=self.agent.speed_queue,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                    boundary_size=(CONFIG["x_res"]//2, CONFIG["y_res"]//2),
                                                    bbox_list=self.agent.frame_queue,
                                                                 next_bbox_list=next_bbox_list
                                                    )
            self.overlap=overlap
            # map_list=map_list[:,-1:]
            wall_list=self.agent.occupancy_map.get_wall_series(transform=self.agent.vt_queue[-1],magnitude=WALL_MAGNITUDES,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]))
            # print([x.shape for x in wall_list])

            # wall_list=np.array([skimage.measure.block_reduce(wall_list[i], (WALL_MAGNITUDES[i],WALL_MAGNITUDES[i]), np.max) for i in range(len(wall_list))])
            # print(map_list.shape,wall_list.shape)
            # print(np.max(wall_list),np.min(wall_list))
            map_list=np.vstack((map_list,wall_list))
            image=np.hstack(map_list)
            # Get the image dimensions
            # height, width = image.shape[:2]

            # # Double the size of the image
            # new_height = height * 2
            # new_width = width * 2

            # # Resize the image using cv2.resize()
            # resized_img = cv2.resize(image, (new_width, new_height))

            cv2.imshow("data", image) # uncomment to show occu map
            cv2.waitKey(1)
            #print(mapList.shape,'------------------------------------------------------------------------------------------------------------------------')
            return map_list

        else:
            wall_list=self.agent.occupancy_map.get_wall_series(transform=self.agent.vt_queue[-1],magnitude=WALL_MAGNITUDES,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]))
            # obs=[]
            # yaw=self.agent.vt_queue[-1].rotation.yaw
            # roll=self.agent.vt_queue[-1].rotation.roll
            # pitch=self.agent.vt_queue[-1].rotation.pitch
            # obs.append(roll)
            # obs.append(pitch)
            # obs=np.array(obs)
            rotation=self.agent.vt_queue[-1].rotation.to_array()
            speed=self.agent.vehicle.velocity.to_array()
            acceleration=self.agent.acceleration.to_array()
            angular_v=self.agent.angular_v.to_array()

            target=self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)]
            target_yaw=target.get_yaw()
            target_dis=target.dis(self.agent.vehicle.transform.location.x,self.agent.vehicle.transform.location.z)
            # target_eq=target.eq(self.agent.vehicle.transform.location.x,self.agent.vehicle.transform.location.z)
            reward_line=np.array([target_yaw,target_dis]).flatten()

            obs=np.concatenate([rotation/180,speed/200,acceleration/5,angular_v/np.pi,reward_line/10,self.prev_action])#,np.array(self.prev_collisoin)])
            # print(f"obs shape {obs.shape}")

            target_shape = np.array(wall_list).shape
            # print(f"target shape {target_shape}")
            tmp=np.zeros((target_shape[1], target_shape[2]))
            # print(f"tmp shape {tmp.shape}")
            tmp[0, : len(obs)] = obs
            map_list = np.zeros((target_shape[0] + 1, target_shape[1], target_shape[2]))   
            map_list[:2,:] = np.array(wall_list)
            map_list[2] = tmp
            # print(f"observation shape {map_list.shape}")
            return map_list


    def reset(self) -> Any:
        self.crash_check = False
        if len(self.his_checkpoint)>=10:
            self.his_checkpoint=self.his_checkpoint[-10:]
            self.his_score=self.his_score[-10:]
        if self.agent:
            self.his_checkpoint.append(self.agent.int_counter*self.agent.interval)
            self.his_score.append(self.ep_rewards)
        self.ep_rewards = 0
        self.stopped_counter = 0
        if self.steps>self.largest_steps and not self.complete_loop:
            self.largest_steps=self.steps
        elif self.complete_loop and self.agent.finish_loop and self.steps<self.largest_steps:
            self.largest_steps=self.steps
        self.overlap=False
        # Change Spawn Point before reset
        self.agent_config.spawn_point_id = next_spawn_point(self.agent_config.spawn_point_id)
        print("Spawn Pt ID", self.agent_config.spawn_point_id)
        self.EgoAgentClass.spawn_counter = spawn_params["spawn_int_map"][self.agent_config.spawn_point_id]
        self.agent.spawn_counter = spawn_params["spawn_int_map"][self.agent_config.spawn_point_id]

        super(ROARppoEnvE2E, self).reset()
        self.agent.spawn_counter = spawn_params["spawn_int_map"][self.agent_config.spawn_point_id]
        print(self.agent.spawn_counter,'spawn_counter')
        self.steps=0
        self.agent.kwargs["control"] = VehicleControl(throttle=1.0,
                                                            steering=0.0,
                                                            braking=0.0)
        

        empty_frames = random.randrange(80,180)

        for _ in range(empty_frames):
            # print('step '+str(self.steps))
            super(ROARppoEnvE2E, self).step(None)
            self.steps+=1
            # print(self.agent.bbox_list[self.agent.int_counter%len(self.agent.bbox_list)].get_directional_velocity(self.agent.vehicle.velocity.x,self.agent.vehicle.velocity.y),'----------------------------------')
            self.stopped_counter =0
            self.crash_check = False
        self.crash_step=0
        self.reward_step=0
        self.carla_runner.world.collision_sensor.history=[]
        self.last_num_collision=0
        self.stopped_counter =0
        self.crash_check = False
        return self._get_obs()

    def wandb_logger(self):
        wandb.log({
            "Episode reward": self.ep_rewards,
            "Checkpoint reached": self.agent.int_counter*self.agent.interval,
            "largest_steps" : self.largest_steps,
            "highest_speed" : self.highspeed,
            "Episode_Sim_Time": self.sim_lap_time,
            "episode Highspeed": self.current_hs,
            "avg10_checkpoints":np.average(self.his_checkpoint),
            "avg10_score":np.average(self.his_score),
        })
        return