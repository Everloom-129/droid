"""
PI0 Evaluation Utilities
Tony 04/05/2025
- V 0.1.0
This module contains utility functions and classes for the PI0 robot evaluation script.
It separates core functionality from the main evaluation loop for better organization.
"""

import contextlib
import datetime
import json
import os
import signal
import time
from typing import Dict, List, Tuple, Any, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from openpi_client import image_tools
from moviepy.editor import ImageSequenceClip


# === Data Encoding Utilities ===

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# === Context Managers ===

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


# === Directory and File Management ===

def ensure_directories(date: str, main_category: str) -> Dict[str, str]:
    """
    Create necessary directories for logging and results.
    
    Args:
        date: Date string in MMDD format
        main_category: Main category for this evaluation session
        
    Returns:
        Dictionary of created paths
    """
    # Create main directories
    log_dir = f"results/log/{date}"
    plot_dir = f"{log_dir}/action_plot/{main_category}"
    json_dir = f"{log_dir}/action_json/{main_category}"
    video_dir = f"results/videos/{date}/{main_category}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    
    return {
        "log_dir": log_dir,
        "plot_dir": plot_dir,
        "json_dir": json_dir,
        "video_dir": video_dir
    }


def create_markdown_file(date: str, main_category: str) -> str:
    """
    Create and initialize a markdown file for logging evaluation results.
    
    Args:
        date: Date string in MMDD format
        main_category: Main category for this evaluation session
        
    Returns:
        Path to the created markdown file
    """
    markdown_file = f"results/log/{date}/eval_{main_category}.md"
    
    # Create markdown header
    with open(markdown_file, "a") as f:
        f.write(f"# Pi0-FAST Evaluation: {main_category}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")
    
    return markdown_file


def log_trial_result(markdown_file: str, trial_index: int, instruction: str, 
                    success: float, duration: int, video_path: str, comment: str) -> None:
    """
    Log the result of a trial to the markdown file.
    
    Args:
        markdown_file: Path to the markdown file
        trial_index: Index of the current trial
        instruction: The instruction given to the robot
        success: Success rate (0.0 to 1.0)
        duration: Duration in timesteps
        video_path: Path to the recorded video
        comment: User comment about the trial
    """
    with open(markdown_file, "a") as f:
        f.write(f"### Trial {trial_index}: {instruction}\n")
        f.write(f"- Success: {success * 100}%\n")
        f.write(f"- Duration: {duration} steps\n")
        f.write(f"- Video: [{os.path.basename(video_path)}]({video_path})\n")
        f.write(f"- Comment: {comment}\n\n")


# === Image and Video Processing ===

def extract_observation(args, obs_dict: Dict, *, save_to_disk: bool = False, 
                       main_category: Optional[str] = None, 
                       instruction: Optional[str] = None, 
                       date: Optional[str] = None) -> Dict:
    """
    Extract and process observations from robot sensors.
    
    Args:
        args: Arguments containing camera IDs
        obs_dict: Raw observation dictionary from robot
        save_to_disk: Whether to save images to disk
        main_category: Main category for this evaluation
        instruction: Current instruction
        date: Date string
        
    Returns:
        Processed observation dictionary
    """
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # Note the "left" below refers to the left camera in the stereo pair.
        # The model is only trained on left stereo cams, so we only feed those.
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # Drop the alpha dimension
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # Convert to RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # Save the images to disk so that they can be viewed live while the robot is running
    if save_to_disk and date and main_category and instruction:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")
        combined_image.save(f"results/log/{date}/view_{main_category}_{instruction}.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


def add_instruction_text_to_image(image: np.ndarray, instruction: str) -> np.ndarray:
    """
    Add instruction text to the camera image.
    
    Args:
        image: The image to add text to
        instruction: The instruction text
        
    Returns:
        Image with text added
    """
    camera_image = image.copy()
    
    display_text = instruction[:50] + "..." if len(instruction) > 50 else instruction
    text_position = (10, 30)  
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_color = (255, 255, 255)   
    font_thickness = 2
    
    # Add black outline/shadow for better visibility against any background
    cv2.putText(camera_image, display_text, text_position, font, font_scale, 
               (0, 0, 0), font_thickness + 2)
    
    # Draw the main text in white
    cv2.putText(camera_image, display_text, text_position, font, font_scale, 
               font_color, font_thickness)
    
    return camera_image


def save_combined_video(video_frames: List[np.ndarray], wrist_frames: List[np.ndarray], 
                        save_path: str, fps: int = 10) -> None:
    """
    Create and save a combined video from external and wrist camera frames.
    
    Args:
        video_frames: List of external camera frames
        wrist_frames: List of wrist camera frames
        save_path: Path to save the video
        fps: Frames per second
    """
    video = np.stack(video_frames)
    wrist_video = np.stack(wrist_frames)
    
    # Ensure both videos have the same height for side-by-side display
    target_height = min(video.shape[1], wrist_video.shape[1])
    target_width = min(video.shape[2], wrist_video.shape[2])
    
    # Resize both videos to the same dimensions
    video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in video])
    wrist_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in wrist_video])
    
    # Stack videos horizontally
    combined_video = np.concatenate([video_resized, wrist_video_resized], axis=2)
    
    # Save video
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ImageSequenceClip(list(combined_video), fps=fps).write_videofile(save_path, codec="libx264")


# === Visualization Utilities ===

def create_visualization_figure(instruction: str) -> Tuple[Figure, List, List, List, Any, Any, Any]:
    """
    Create a figure for visualizing robot state during execution.
    
    Args:
        instruction: Current instruction for the title
        
    Returns:
        Tuple containing:
        - Figure object
        - List of axes
        - List of joint position lines
        - List of action lines
        - Gripper action line
        - Gripper position line
        - Status text object
    """
    plt.ion()  # Turn on interactive mode for live updates
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 3 subplots
    fig.suptitle(f"Instruction: {instruction}", fontsize=12)
    
    # First subplot for joint positions
    joint_pos_lines = []
    for i in range(7):  # 7 joint positions
        line, = axs[0].plot([], [], label=f'Joint {i+1}')
        joint_pos_lines.append(line)
    axs[0].set_title('Joint Positions')
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Position')
    axs[0].legend(loc='upper left', fontsize='small')
    
    # Second subplot for velocity actions
    action_lines = []
    for i in range(7):  # 7 joint velocity actions
        line, = axs[1].plot([], [], label=f'Joint {i+1} Velocity')
        action_lines.append(line)
    axs[1].set_title('Joint Velocity Actions')
    axs[1].set_xlabel('Timestep')
    axs[1].set_ylabel('Action Value')
    axs[1].legend(loc='upper left', fontsize='small')
    
    # Third subplot for gripper state
    gripper_action_line, = axs[2].plot([], [], 'r-', linewidth=2, label='Gripper Action')
    gripper_position_line, = axs[2].plot([], [], 'b-', linewidth=2, label='Gripper Position')
    axs[2].set_title('Gripper State')
    axs[2].set_xlabel('Timestep')
    axs[2].set_ylabel('Value [0-1]')
    axs[2].set_ylim(-0.1, 1.1)  # Gripper values are between 0 and 1
    axs[2].legend(loc='upper left', fontsize='small')
    
    # Text area for real-time statistics
    status_text = axs[0].text(0.78, 0.95, '', transform=axs[0].transAxes, 
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show(block=False)
    
    return fig, axs, joint_pos_lines, action_lines, gripper_action_line, gripper_position_line, status_text


def update_visualization(fig, axs, joint_pos_lines, action_lines, gripper_action_line, 
                         gripper_position_line, status_text, xdata, joint_ydata, 
                         action_ydata, gripper_action_ydata, gripper_position_ydata, 
                         t_step, current_time, current_chunk_index, actions_from_chunk_completed):
    """
    Update the visualization with new data.
    
    Args:
        fig: Figure object
        axs: List of axes
        joint_pos_lines: List of joint position plot lines
        action_lines: List of action plot lines
        gripper_action_line: Gripper action plot line
        gripper_position_line: Gripper position plot line
        status_text: Status text object
        xdata: X data for plots (timesteps)
        joint_ydata: Joint position Y data
        action_ydata: Action Y data
        gripper_action_ydata: Gripper action Y data
        gripper_position_ydata: Gripper position Y data
        t_step: Current timestep
        current_time: Current execution time
        current_chunk_index: Current action chunk index
        actions_from_chunk_completed: Number of actions completed in current chunk
    """
    # Update joint position lines
    for i, line in enumerate(joint_pos_lines):
        line.set_data(xdata, joint_ydata[i])
    
    # Update action lines (only joint velocities)
    for i, line in enumerate(action_lines):
        if i < len(action_ydata) - 1:  # Skip gripper in action lines
            line.set_data(xdata, action_ydata[i])
    
    # Update gripper lines
    gripper_action_line.set_data(xdata, gripper_action_ydata)
    gripper_position_line.set_data(xdata, gripper_position_ydata)
    
    # Update status text
    status_text.set_text(f"Timestep: {t_step}\n"
                         f"Time: {current_time:.2f}s\n"
                         f"Chunk: {current_chunk_index}\n"
                         f"Action in chunk: {actions_from_chunk_completed-1}/8\n")
    
    # Rescale axes
    for ax in axs:
        ax.relim()
        ax.autoscale_view()
    
    # Draw and refresh plot
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


def mark_chunk_boundary(axs, t_step, chunk_boundary_lines):
    """
    Add a vertical line to mark a new action chunk boundary.
    
    Args:
        axs: List of plot axes
        t_step: Current timestep
        chunk_boundary_lines: List to store boundary line references
        
    Returns:
        Updated list of chunk boundary lines
    """
    for ax in axs:
        # Add a vertical line at this timestep to show when a new chunk was received
        line = ax.axvline(x=t_step, color='r', linestyle='--', alpha=0.5)
        chunk_boundary_lines.append(line)
    
    return chunk_boundary_lines


def save_visualization_snapshot(fig, save_path):
    """
    Save a snapshot of the current visualization figure.
    
    Args:
        fig: Figure object to save
        save_path: Path to save the figure
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the figure
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization snapshot saved to {save_path}")
    except Exception as e:
        print(f"Failed to save visualization snapshot: {e}")


# === Action Processing ===

def process_action(action):
    """
    Process and normalize the action vector.
    
    Args:
        action: Raw action vector from policy
        
    Returns:
        Processed action vector
    """
    # Binarize gripper action
    if action[-1].item() > 0.5:
        action = np.concatenate([action[:-1], np.ones((1,))])
    else:
        action = np.concatenate([action[:-1], np.zeros((1,))])

    # clip all dimensions of action to [-1, 1]
    action = np.clip(action, -1, 1)
    
    return action


def save_action_history(action_history_data, file_path):
    """
    Save action history data to JSON file.
    
    Args:
        action_history_data: Dictionary containing action history and metadata
        file_path: Path to save the JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(action_history_data, f, indent=2, cls=NumpyJSONEncoder)
    
    print(f"Action history saved to {file_path}")


# === User Input Handling ===

def get_success_value():
    """
    Get success value from user input.
    
    Returns:
        Success value as float between 0 and 1
    """
    success = None
    while not isinstance(success, float):
        success = input(
            "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec: "
        )
        if success == "y" or success == "1":
            success = 1.0
        elif success == "n" or success == "0":
            success = 0.0
        elif success == "-1":
            success = -1
        else:
            try:
                success = float(success) / 100
                if not (0 <= success <= 1):
                    print(f"Success must be a number in [0, 100] but got: {success * 100}")
                    success = None
            except ValueError:
                print("Invalid input. Please enter y, n, or a number between 0-100")
                success = None
    
    return success 