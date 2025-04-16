#!/usr/bin/env python3
"""
PI0 Robot Evaluation Script

This script runs evaluation trials for the PI0 robot, connecting to a policy server
that provides action predictions based on robot observations.
"""

import datetime
import dataclasses
import faulthandler
import time
import os

import numpy as np
import pandas as pd
import tqdm
import tyro
from droid.robot_env import RobotEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

import pi0_eval_utils as utils

faulthandler.enable()


@dataclasses.dataclass
class Args:
    """Command line arguments for PI0 evaluation."""
    # Hardware parameters
    left_camera_id: str = "25455306"  # e.g., "24259877"
    right_camera_id: str = "27085680"  # fix: "27085680"  move: # "26368109"  
    wrist_camera_id: str = "14436910"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = (
        "left"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )

    # Rollout parameters
    max_timesteps: int = 800
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "158.130.52.14"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )

    # Evaluation parameters
    eval_name: str = "default"  # Name for this evaluation session


def main(args: Args):
    """Main evaluation function."""
    print("Entered main!")
    
    # Validate external camera selection
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the robot environment
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Initialize DataFrame for logging
    df = pd.DataFrame(columns=["success", "duration", "video_filename", "instruction", "comment"])
    
    # Create timestamps and logging directories
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    date = datetime.datetime.now().strftime("%m%d")
    
    # Get main category for this evaluation session
    main_category = input("Enter main category for this evaluation session: ")
    
    # Set up directories and create markdown file
    paths = utils.ensure_directories(date, main_category)
    markdown_file = utils.create_markdown_file(date, main_category)

    # Main evaluation loop
    while True:
        # Get instruction from user
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Create timestamp for this trial
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        
        # Create safe instruction string for filenames
        safe_instruction = instruction.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
        
        # Initialize video storage
        video_frames = []
        wrist_video_frames = []
        
        # Initialize data collection
        joint_positions = []
        action_history = []
        action_chunks_history = []
        chunk_timestamps = []
        current_chunk_index = -1
        
        # Set up visualization
        fig, axs, joint_pos_lines, action_lines, gripper_action_line, gripper_position_line, status_text = (
            utils.create_visualization_figure(instruction)
        )
        
        # Data for visualization
        xdata = []
        joint_ydata = [[] for _ in range(7)]
        action_ydata = [[] for _ in range(8)]
        gripper_action_ydata = []
        gripper_position_ydata = []
        chunk_boundary_lines = []
        
        # Start time for execution metrics
        start_time = time.time()

        # Run the rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        
        for t_step in bar:
            try:
                # Get current observation
                curr_obs = utils.extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=(t_step == 0),
                    main_category=main_category,
                    instruction=instruction,
                    date=date,
                )
                
                # Add instruction text to camera image
                camera_image = utils.add_instruction_text_to_image(
                    curr_obs[f"{args.external_camera}_image"], instruction
                )
                
                # Store frames for video
                video_frames.append(camera_image)
                wrist_video_frames.append(curr_obs["wrist_image"])
                
                # Store joint positions
                joint_positions.append(curr_obs["joint_position"])

                # Request new action chunk if needed
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0
                    current_chunk_index += 1
                    
                    # Prepare observation data for policy server
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(
                            curr_obs["wrist_image"], 224, 224
                        ),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Request action chunk from policy server
                    server_request_start = time.time()
                    with utils.prevent_keyboard_interrupt():
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    server_request_duration = time.time() - server_request_start
                    
                    assert pred_action_chunk.shape == (10, 8)
                    
                    # Record action chunk
                    action_chunks_history.append({
                        "chunk_index": current_chunk_index,
                        "timestep": t_step,
                        "time": time.time() - start_time,
                        "server_request_duration": server_request_duration,
                        "chunk": pred_action_chunk.tolist(),
                    })
                    chunk_timestamps.append(t_step)
                    
                    # Mark chunk boundary in visualization
                    chunk_boundary_lines = utils.mark_chunk_boundary(axs, t_step, chunk_boundary_lines)

                # Select and process current action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1
                action = utils.process_action(action)
                
                # Record action
                action_record = {
                    "timestep": t_step,
                    "time": time.time() - start_time,
                    "chunk_index": current_chunk_index,
                    "action_index_in_chunk": actions_from_chunk_completed - 1,
                    "action": action.tolist(),
                    "joint_position": curr_obs["joint_position"].tolist(),
                    "gripper_position": curr_obs["gripper_position"].tolist()
                }
                action_history.append(action_record)
                
                # Update visualization data
                xdata.append(t_step)
                
                # Joint positions data
                for i in range(7):
                    joint_ydata[i].append(curr_obs["joint_position"][i])
                
                # Action data
                for i in range(7):
                    action_ydata[i].append(action[i])
                action_ydata[7].append(action[7])
                
                # Gripper data
                gripper_action_ydata.append(action[7])
                gripper_position_ydata.append(curr_obs["gripper_position"][0])
                
                # Update visualization periodically
                if t_step % 5 == 0:
                    current_time = time.time() - start_time
                    utils.update_visualization(
                        fig, axs, joint_pos_lines, action_lines, 
                        gripper_action_line, gripper_position_line, status_text,
                        xdata, joint_ydata, action_ydata, gripper_action_ydata, gripper_position_ydata,
                        t_step, current_time, current_chunk_index, actions_from_chunk_completed
                    )

                # Execute action on robot
                env.step(action)
                
            except KeyboardInterrupt:
                break

        # Final execution time
        total_execution_time = time.time() - start_time
        
        # Save visualization snapshot
        vis_snapshot_path = f"{paths['plot_dir']}/eval_{instruction}_visualization_{timestamp}.png"
        utils.save_visualization_snapshot(fig, vis_snapshot_path)
        
        # Close visualization
        plt.close(fig)
        
        # Save action history to JSON
        action_history_file = f"{paths['json_dir']}/eval_{instruction}_action_history_{timestamp}.json"
        
        # Prepare metadata
        action_history_data = {
            "metadata": {
                "instruction": instruction,
                "timestamp": timestamp,
                "date": date,
                "category": main_category,
                "total_timesteps": t_step + 1,
                "total_execution_time": total_execution_time,
                "open_loop_horizon": args.open_loop_horizon,
                "external_camera": args.external_camera,
                "left_camera_id": args.left_camera_id,
                "right_camera_id": args.right_camera_id,
                "wrist_camera_id": args.wrist_camera_id,
                "remote_host": args.remote_host,
                "remote_port": args.remote_port
            },
            "action_chunks": action_chunks_history,
            "chunk_timestamps": [int(t) for t in chunk_timestamps],
            "detailed_actions": action_history
        }
        
        # Save action history
        utils.save_action_history(action_history_data, action_history_file)
        
        # Save combined video
        video_filename = os.path.join(
            paths["video_dir"], 
            f"{args.external_camera}_{safe_instruction}_{timestamp}.mp4"
        )
        utils.save_combined_video(video_frames, wrist_video_frames, video_filename)

        # Get evaluation results from user
        success = utils.get_success_value()
        comment = input("Enter comment about this trial: ")

        # Log trial results
        utils.log_trial_result(
            markdown_file, 
            len(df) + 1, 
            instruction, 
            success, 
            t_step, 
            video_filename, 
            comment
        )

        # Update DataFrame
        df = pd.concat([df, pd.DataFrame([{
            "success": success,
            "duration": t_step,
            "video_filename": video_filename,
            "instruction": instruction,
            "comment": comment
        }])], ignore_index=True)

        # Ask if user wants to continue
        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
            
        # Reset robot for next trial
        env.reset()

    # Save final results to CSV
    csv_filename = markdown_file.replace(".md", ".csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {markdown_file} and {csv_filename}")


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args) 