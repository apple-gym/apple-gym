#!/usr/bin/env python
"""
This script tries to collect demonstrations, by cheating.

It starts at an apple and moves backwards. If it doesn't get trapped, then it start to record, as it moves towards the apple, retracing it's steps.
"""
# coding: utf-8

# In[1]:

import gym
from gym_recording_modified.wrappers import TraceRecordingWrapper
import pybullet as p
import numpy as np
# from diy_gym.utils import flatten, unflatten
from matplotlib import pyplot as plt
from pprint import pprint
from rich import print
from pathlib import Path
import pybullet_planning as pbp
import apple_gym.env
import time
import logging
import hickle
import os
import sys
import argparse

from tqdm.auto import tqdm
plt.ion()


parser = argparse.ArgumentParser(description='collect exported trees and make urdf')
parser.add_argument('-d', '--debug', action="store_true")
parser.add_argument('-r', '--render', action="store_true")
parser.add_argument('-b', '--batch', default=512, type=int)

# %%
if 'ipykernel_launcher' in os.sys.argv[0]:
    args = parser.parse_args("-d".split(' '))
else:
    args = parser.parse_args()


from loguru import logger
from rich import print
from rich.logging import RichHandler
# from rich.progress import track as tqdm
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(rich_tracebacks=True, markup=True)])
logger = logging.getLogger('apple_gym')
logger.setLevel(logging.DEBUG)


# In[8]:

# # Init

directory=Path('data/demonstrations')
directory.mkdir(exist_ok=True)

debug = args.debug
env = gym.make('ApplePick-v0', render=args.render)
env = TraceRecordingWrapper(env, directory=str(directory), batch_size=int(args.batch))
env_diy = env.unwrapped
env_diy.models, env_diy.config


# # Manual Operation, collect demos
# 
# Lets manually grab the berries, this will unit test it and provide offline demonstrations.
# 
# We will use the gym interface to control it.
# 
# - add debug to make sure target position is ok, draw it
# - add instant snap position
# - make sure we can do inv, kin in bullet coords
# - then diy gym

# # Using PBP

# In[9]:



# Now use inverse kinematics to make sure we can reach random dots
class InvKin:
    def __init__(self, arm_uid, ee_link):
        self.end_effector_joint_id = ee_link
        self.arm_uid = arm_uid
        self.rest_position = env_diy.config.get('robot').get('controller').get('rest_position')
        joint_info = [p.getJointInfo(arm_uid, i) for i in range(p.getNumJoints(arm_uid))]
        joints = [info[1].decode('UTF-8') for info in joint_info if info[0] <= self.end_effector_joint_id]
        self.joint_ids = [info[0] for info in joint_info if info[1].decode('UTF-8') in joints and info[3] > -1]
#         self.joint_ids = pbp.get_movable_joints(arm_uid)
        self.torque_limit = [p.getJointInfo(arm_uid, joint_id)[10] for joint_id in self.joint_ids]
        self.joint_position_lower_limit = [p.getJointInfo(arm_uid, joint_id)[8] for joint_id in self.joint_ids]
        self.joint_position_upper_limit = [p.getJointInfo(arm_uid, joint_id)[9] for joint_id in self.joint_ids]
        self.jointRanges = np.subtract(self.joint_position_upper_limit, self.joint_position_lower_limit).tolist()

    def ik(self, target_pose):
        """Calc and execute inv kinematics for arm.
        
        pos_target: world pos of target [x, y, z]
        rot_target: world orientation, either cartesian or quarternian [x, y, z, w]
        """
        pos_target, rot_target = target_pose
        kinematic_conf = p.calculateInverseKinematics(bodyUniqueId=self.arm_uid,
                                                    endEffectorLinkIndex=self.end_effector_joint_id,
                                                    targetPosition=pos_target,
                                                    lowerLimits=self.joint_position_lower_limit,
                                                    upperLimits=self.joint_position_upper_limit,
                                                    jointRanges=self.jointRanges,
                                                    restPoses=self.rest_position,
                                                    targetOrientation=rot_target
                                                    )[:self.end_effector_joint_id - 1]
        

        return kinematic_conf
    
    def move_to_conf(self, kinematic_conf, dist=None):

        if dist is not None:
            pGain = 0.015 / dist
            vGain = 1.0 * dist
        else:
            pGain = 0.015
            vGain = 1.0

        p.setJointMotorControlArray(
            self.arm_uid,
            self.joint_ids,
            p.POSITION_CONTROL,
            targetPositions=kinematic_conf,
            targetVelocities=[0.01] * len(kinematic_conf),
            forces=self.torque_limit,
            # TODO I may want to change this as we get closer
            positionGains=[0.015] * len(kinematic_conf),
            velocityGains=[0.001] * len(kinematic_conf),
        )
        
    def move_to_pose(self, target_pose, dist=None):
        kinematic_conf = self.ik(target_pose)
        self.move_to_conf(kinematic_conf, dist=dist)
        return kinematic_conf
        
    def move_to_target(self, target_uid, dist=None):
        target_pose = pbp.get_link_pose(target_uid, 0)
        return self.move_to_pose(target_pose, dist=dist)
        
    def stop(self):
        kinematic_conf = pbp.get_joint_positions(self.arm_uid, self.joint_ids)
        self.move_to_conf(kinematic_conf)
        return kinematic_conf
        
    def teleport_to_conf(self, kinematic_conf):
        return pbp.set_joint_positions(self.arm_uid, self.joint_ids, kinematic_conf)
        
    def teleport_to_pose(self, target_pose):
        kinematic_conf = self.ik(target_pose)
        self.teleport_to_conf(kinematic_conf)
        return kinematic_conf
        
    def teleport_to_target(self, target_uid):
        target_pose = pbp.get_link_pose(target_uid, 0)
        return self.teleport_to_pose(target_pose)
    
    def distance_to(self, target_uid):
        target_pose = pbp.get_link_pose(target_uid, 0)
        pose_gripper = pbp.get_link_pose(robot, ee_link)
        d0 = pbp.get_distance(pose_gripper[0], target_pose[0])
        return d0
    
    def rand_pose(self):
        pose = np.random.randn(len(arm.joint_ids)) * arm.jointRanges + arm.joint_position_lower_limit
        return pose/2

    @property
    def state(self):
        return pbp.get_link_state(self.arm_uid, self.end_effector_joint_id)




# ## v2 via gym

# In[10]:





robot = env.unwrapped.receptors['robot'].uid
ee_link = pbp.joint_from_name(robot, 'ee_joint')
arm = InvKin(robot, ee_link)

def choose_target(env_diy):
    # Choose random target
    targets = env_diy.addons['tree'].fruitIds
    target_uid = np.random.choice(targets)

    if debug:
        # Mark target
        # target_pose = p.getBasePositionAndOrientation(target_uid)[0]
        target_pose = pbp.get_link_state(target_uid, 0).linkWorldPosition
        pos_gripper = pbp.get_link_state(robot, ee_link).linkWorldPosition
        p.addUserDebugLine(
            lineFromXYZ=pos_gripper, 
            lineToXYZ=target_pose, lineColorRGB=[1,0,0], lineWidth=1, 
            lifeTime=15)
    return target_uid



def name_fruit(lifeTime=30):
    targets = env_diy.addons['tree'].fruitIds
    for i in targets:
        xyz, rpy = pbp.get_pose(i)
        p.addUserDebugText(f"{i}", xyz, lifeTime=lifeTime)
        

# In[ ]:


picked = False
n_picks = 0

infos = []
infos_small = []
t=time.time()
def save_infos():
    hickle.dump(infos_small, f'data/infos/small_{t}.hkl')
    hickle.dump(infos, f'data/infos/{t}.hkl')
    print(f'data/infos/small_{t}.hkl')


with tqdm() as pr:
    def dist(target_uid):
        d = arm.distance_to(target_uid)
        pr.desc=f"picks={n_picks} dist={d:2.4g}"
        return d

    def step(n=1):
        for _ in range(n):
            # pr.update(1)
            if debug:
                time.sleep(0.05)
            p.stepSimulation()
    try:
        tol = env_diy.addons['tree'].picking_tolerance
        for j in range(10000): # try N random targets
            if picked:
                picked = False
                env.reset()
                name_fruit()
                step(10)
            p.removeAllUserDebugItems()

            for i in range(100): # try to get to a target up to N times
                # 0 get initial random pos
                target_uid = choose_target(env_diy)
                env_diy.models['robot'].addons['controller'].reset()

                kinematic_conf_init = pbp.get_joint_positions(robot, arm.joint_ids)

                # 1. teleport to target

                # solve with angle
                target_pose = xyz, rpy = pbp.get_link_pose(target_uid, 0)
                kinematic_conf_target0 = arm.teleport_to_pose(target_pose)
                step(1)

                # 2nd solve without angle (gets us closer)
                target_pose = (xyz, None)
                kinematic_conf_target = arm.teleport_to_pose(target_pose)
                step(1)

                # check its close, and upright
                d0 = dist(target_uid)
                pose_gripper_close = pbp.get_link_pose(robot, ee_link)
                target_pose = pbp.get_link_pose(target_uid, 0)        
                a1 = pbp.get_angle(pose_gripper_close[1], target_pose[1])
                is_close = pbp.is_pose_close(
                    pose_gripper_close, 
                    target_pose,
                    pos_tolerance=.5e-1, 
                    ori_tolerance=.1e-0*np.pi)
                if is_close:
                    break

            if not is_close:
                logger.info(f'1. t={target_uid} couldn\'t get close to target {d0}')
                picked = True
                continue # to next target

            logger.info(f"1. t={target_uid}. d={d0:2.4f} Got close")
            p.removeAllUserDebugItems()

            # 2. move away from target in series of poses
            dist_target = np.random.rand(1, int(np.ceil(2+tol*4)))
            # poses = []
            actions = []
            last_pose = pbp.get_link_pose(robot, ee_link)
            for i in range(4):
                # get a random new position, some distance away
                pos, orn = pose_gripper = pbp.get_link_pose(robot, ee_link)
                target_pos = pos + np.random.randn(len(pos))/2
                target_conf = arm.ik((target_pos, None))

                # Mark target, Blue
                p.addUserDebugText(f'{i}. away', pose_gripper[0])
                p.addUserDebugLine(
                    lineFromXYZ=pose_gripper[0], 
                    lineToXYZ=target_pos, lineColorRGB=[0.2, 0.2, 0.9], lineWidth=1)

                d1 = dist(target_uid)
                logger.info(f'moving i={i} to target_conf={target_conf} d1={d1}')

                arm.move_to_conf(target_conf)
                for i in range(10):
                    step(1)
                    if debug:
                        # Mark target
                        # p.addUserDebugText(f'{i}. away', pose_gripper[0])
                        # Actual path Green
                        p.addUserDebugLine(
                            lineFromXYZ=last_pose[0], 
                            lineToXYZ=pose_gripper[0], lineColorRGB=[0.4, 0.75, 0.4], lineWidth=4)
                    last_pose = pose_gripper

                arm.stop()
                # store actual conf (we could store position instead?)
                action = pbp.get_joint_positions(robot, arm.joint_ids)
                actions.append(action)
            arm.stop()

            actions.append(kinematic_conf_target)

            d1 = dist(target_uid)

            if d1<tol*2:
                logger.info(f"2. t={target_uid}. d={d1:2.4f} Couldn't move away from target")
                continue
            else:
                logger.info(f"2. t={target_uid}. d={d1:2.4f} Got far")

            if debug:
                target_pose = pbp.get_link_pose(target_uid, 0)
                pose_gripper_far = pbp.get_link_pose(robot, ee_link)


            # play back actions
            ds = []
            d = np.inf
            last_pos = pbp.get_link_pose(robot, ee_link)
            for i, action in enumerate(actions):
                if picked:
                    break
                for j in range(20):                    
                    # record dist
                    d3 = dist(target_uid)
                    ds.append(ds)

                    # step
                    observation_new, reward, is_done, info = env.step(action)
                    if debug and len(infos) < 5000:
                        time.sleep(0.05)
                        info_small = {k:np.array(v) for k,v in info.items() if len(np.array(v).shape)<2} 
                        infos_small.append(info_small)
                    pr.update(1)

                    if debug:
                        pos = pbp.get_link_pose(robot, ee_link)
                        # actual replay red
                        p.addUserDebugLine(
                            lineFromXYZ=last_pos[0], 
                            lineToXYZ=pos[0], lineColorRGB=[0.9, .1, 0.5], lineWidth=4)
                        last_pos = pos

                    # picked?
                    if info['env_reward/apple_pick/tree/gripping_fruit_reward']:
                        picked = True
                        logger.warning('picked!')
                        infos.append(info)
                        n_picks += 1
                        save_infos()
                        if picked and debug:
                            plt.title(f"3. t={target_uid}. d={d3:2.4f} picked i={picked}")
                            plt.imshow(last_info['env_obs/robot/arm_camera/rgb'])
                            plt.show()
                    last_info = info
                logger.info(f"3. t={target_uid}. d={d3:2.4f} picked i={picked}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")



# In[ ]:

# Save recordings
r = env.recording.rewards
env.close()

# plt.title('rewards')
# plt.plot(r)
# plt.show()
# print(max(r), len(r))


print('saved recordings with pid', os.getpid())

save_infos()
