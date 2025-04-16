import torch
import numpy as np
import os
import subprocess
import os.path as osp
import pickle
from copy import deepcopy
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf, open_dict
import logging
import time
import wandb
import cv2
from PIL import Image
import json


# from act_utils import load_data, load_data_old # data functions
# from act_utils import (
#     set_seed, np2torch_dict,
#     eval_realrobot, make_policy, generate_vis_samples,
#     visualize_eval_realrobot, normalize, unnormalize
#  ) # helper functions

from droid.user_interface.data_collector import DataCollecter
from droid.controllers.oculus_controller import VRPolicy
from droid.user_interface.gui import RobotGUI
from droid.robot_env import RobotEnv
from droid.data_processing.timestep_processing import TimestepProcesser
from droid.camera_utils.info import camera_type_to_string_dict
from droid.misc.transformations import change_pose_frame



import warnings
warnings.filterwarnings('ignore')

@hydra.main(config_path='configs', config_name='config_realrobot')
def main(cfg):
    logging.info('##### Job Configurations #####')
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f'##### Begin Experiment #####')

    # Set seed
    torch.manual_seed(1)
    np.random.seed(1)

    # === Basic setup ===
    ckpt_dir = cfg.ckpt_dir
    mode = cfg.mode
    action_space = cfg.action_space
    task = cfg.task_name.replace('_', ' ')

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # === Chunk dimension ===
    dim = 6 if 'ori' in cfg.mode else 3

    # === Logging and ckpt ===
    policy_name = cfg.policy.policy_class
    run_id = cfg.run_id
    exp_id = cfg.exp_id
    exp_name = os.path.join(policy_name, f"run{run_id}/{exp_id}")
    logdir = os.path.join(os.getcwd(), "evaluation_logs", exp_name)
    os.makedirs(logdir, exist_ok=True)
    logdirs = [logdir]

    # === Goal Image ===
    model_camera = cfg.model_camera
    if model_camera == 'fixed':
        goal_img_path = '/home/franka/R2D2_3dhat/images/goal_images/fixed_camera_goal_raw.png'
    elif model_camera == 'varied':
        goal_img_path = '/home/franka/R2D2_3dhat/images/goal_images/varied_camera_goal_raw.png'
    elif model_camera == 'hand':
        goal_img_path = '/home/franka/R2D2_3dhat/images/goal_images/hand_camera_goal_raw.png'
    else:
        raise ValueError(f"Unknown model_camera: {model_camera}")

    goal_img = cv2.imread(goal_img_path)
    goal_img = cv2.cvtColor(goal_img, cv2.COLOR_BGR2RGB) / 255.0
    goal_img = cv2.resize(goal_img, (456, 256)).transpose(2, 0, 1)

    # === Camera intrinsics ===
    camera_dict = {}
    for camera_name in ['fixed', 'varied', 'hand']:
        if camera_name == 'fixed':
            params = [526.88745117, 526.88745117, 646.46374512, 353.03808594, 0, 0, 0, 0]  # 25455306
        elif camera_name == 'varied':
            params = [528.82501221, 528.82501221, 636.78747559, 372.25619507, 0, 0, 0, 0]  # 27085680
        elif camera_name == 'hand':
            params = [363.47341919, 363.47341919, 334.21835327, 184.21723938, 0, 0, 0, 0]  # 14436910

        K = np.array([[params[0], 0, params[2]],
                      [0, params[1], params[3]],
                      [0, 0, 1]], dtype=np.float32)
        D = np.array(params[4:], dtype=np.float32)
        camera_dict[camera_name] = (K, D)

    # === Create Pi0 Policy ===
    wrapped_policy = Pi0Wrapper(
        name=policy_name,
        task=task,
        chunk_size=cfg.chunk_size,
        goal_img=goal_img,
        goal_img_path=goal_img_path,
        camera_dict=camera_dict,
        model_camera=model_camera,
    )

    # === Robot Setup ===
    policy_action_space = action_space
    policy_camera_kwargs = {
        "hand_camera": {"depth": True},
        "fixed_camera": {"depth": True},
        "varied_camera": {"depth": True},
    }

    env = RobotEnv(action_space=policy_action_space, camera_kwargs=policy_camera_kwargs)
    controller = VRPolicy()

    data_collector = DataCollecter(
        env=env,
        controller=controller,
        policy=wrapped_policy,
        save_traj_dir=logdirs,
        save_data=True,
    )

    user_interface = RobotGUI(robot=data_collector)



import numpy as np
from openpi_client import websocket_client_policy, image_tools
import cv2


class Pi0Wrapper:
    def __init__(
        self,
        name,
        task,
        chunk_size,
        goal_img,
        goal_img_path,
        camera_dict,
        model_camera,
        remote_host="158.130.52.14",
        remote_port=8000,
    ):
        self.name = name
        self.task = task
        self.chunk_size = chunk_size
        self.goal_img = goal_img
        self.goal_img_path = goal_img_path
        self.camera_dict = camera_dict
        self.remote_host = remote_host
        self.remote_port = remote_port

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=remote_host, port=remote_port
        )

        self.chunk = None
        self.chunk_step = 0
        self.chunk_index = -1

        self.instruction = task  # for Pi0 prompt
        self.external_camera = model_camera  # e.g., 'fixed', 'varied', 'hand'

        print(f"[Pi0Wrapper] Initialized with task: {self.instruction}, camera: {self.external_camera}")

    def reset(self, task=None):
        """Reset chunk state (e.g., on new rollout)"""
        if task:
            self.instruction = task
        self.chunk = None
        self.chunk_step = 0
        self.chunk_index = -1

    def forward(self, obs):
        """Standard policy.forward(obs) interface for real robot"""
        obs_data = self._process_obs(obs)

        # Request a new chunk if needed
        if self.chunk is None or self.chunk_step >= self.chunk_size:
            request_data = {
                "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    obs_data["image"], 224, 224),
                "observation/wrist_image_left": image_tools.resize_with_pad(
                    obs_data["wrist_image"], 224, 224),
                "observation/joint_position": obs_data["joint_position"],
                "observation/gripper_position": obs_data["gripper_position"],
                "prompt": self.instruction,
            }

            response = self.client.infer(request_data)
            self.chunk = response["actions"]
            assert self.chunk.shape == (10, 8)
            self.chunk_step = 0
            self.chunk_index += 1

        # Get current action from chunk
        action = self.chunk[self.chunk_step]
        self.chunk_step += 1

        # Binarize gripper
        gripper = 1.0 if action[-1] > 0.5 else 0.0
        full_action = np.clip(np.concatenate([action[:-1], [gripper]]), -1, 1)

        return full_action

    def _process_obs(self, obs):
        image_obs = obs["image"]
        robot_state = obs["robot_state"]

        # Extract camera images (BGRA to RGB)
        image, wrist = None, None
        for cam_id in image_obs:
            if self._camera_match(cam_id, self.external_camera):
                image = image_obs[cam_id][..., :3][..., ::-1]
            elif "hand" in cam_id:
                wrist = image_obs[cam_id][..., :3][..., ::-1]

        if image is None or wrist is None:
            raise ValueError(f"Missing camera image in obs for {self.external_camera}")

        return {
            "image": image,
            "wrist_image": wrist,
            "joint_position": np.array(robot_state["joint_positions"]),
            "gripper_position": np.array([robot_state["gripper_position"]])
        }

    def _camera_match(self, cam_id, camera_type):
        """Check if camera ID corresponds to a given type (fixed/varied/hand)"""
        if camera_type == 'fixed' and '25455306' in cam_id:
            return True
        if camera_type == 'varied' and '27085680' in cam_id:
            return True
        if camera_type == 'hand' and '14436910' in cam_id:
            return True
        return False
