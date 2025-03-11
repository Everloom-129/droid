# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "25455306" # e.g., "24259877"
    right_camera_id: str = "27085680" # "27085680" # "26368109"  
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


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
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


def main(args: Args):
    print("Entered main!")
    # Make sure external camera is specified by user -- we only use one external camera for the policy
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    # Initialize DataFrame and prepare markdown logging
    df = pd.DataFrame(columns=["success", "duration", "video_filename", "instruction", "comment"])
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    date = datetime.datetime.now().strftime("%m%d")
    # Get main category for this evaluation session
    main_category = input("Enter main category for this evaluation session: ")
    os.makedirs(f"results/log/{date}", exist_ok=True)
    markdown_file = f"results/log/{date}/eval_{main_category}.md"


    # Create markdown header
    with open(markdown_file, "a") as f:
        f.write(f"# Pi0-FAST Evaluation: {main_category}\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("## Results\n\n")

    while True:
        instruction = input("Enter instruction: ")

        # Rollout parameters
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # Prepare to save video of rollout
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")

        joint_position_file = f"results/log/{date}/eval_{main_category}_{timestamp}_joints.csv"
        # Create a filename-safe version of the instruction
        safe_instruction = instruction.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]  # limit length
        video = []
        wrist_video = []  # New list for wrist camera frames
        #joint_positions = []
        #action_state = []


        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            try:
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    save_to_disk=t_step == 0,
                )
                # Save both camera views
                video.append(curr_obs[f"{args.external_camera}_image"])
                wrist_video.append(curr_obs["wrist_image"])


                # Send websocket request to policy server if it's time to predict a new chunk
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # We resize images on the robot laptop to minimize the amount of data sent to the policy server
                    # and improve latency.
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # clip all dimensions of action to [-1, 1]
                action = np.clip(action, -1, 1)

                #action_state.append(action)
                #joint_positions.append(curr_obs["joint_position"])

                env.step(action)
            except KeyboardInterrupt:
                break

        # Stack videos side by side
        video = np.stack(video)
        wrist_video = np.stack(wrist_video)
        #action_csv = np.stack(action_state)
        #joint_csv = np.stack(joint_positions)
        #combined_action_csv = np.concatenate([action_csv, joint_csv], axis=1)
        
        # Ensure both videos have the same height for side-by-side display
        target_height = min(video.shape[1], wrist_video.shape[1])
        target_width = min(video.shape[2], wrist_video.shape[2])
        
        # Resize both videos to the same dimensions
        video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in video])
        wrist_video_resized = np.array([image_tools.resize_with_pad(frame, target_height, target_width) for frame in wrist_video])
        
        # Stack videos horizontally
        combined_video = np.concatenate([video_resized, wrist_video_resized], axis=2)

        date = datetime.datetime.now().strftime("%m%d")
        save_dir = f"results/videos/{date}"
        os.makedirs(save_dir, exist_ok=True)
        save_filename = os.path.join(save_dir, f"{args.external_camera }_{safe_instruction}_{timestamp}.mp4")
  
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        # Get success value
        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec: "
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0
            else:
                try:
                    success = float(success) / 100
                    if not (0 <= success <= 1):
                        print(f"Success must be a number in [0, 100] but got: {success * 100}")
                        success = None
                except ValueError:
                    print("Invalid input. Please enter y, n, or a number between 0-100")
                    success = None

        # Get comment about the result
        comment = input("Enter comment about this trial: ")

        # Append to markdown file
        with open(markdown_file, "a") as f:
            f.write(f"### Trial {len(df) + 1}: {instruction}\n")
            f.write(f"- Success: {success * 100}%\n")
            f.write(f"- Duration: {t_step} steps\n")
            f.write(f"- Video: [{os.path.basename(save_filename)}]({save_filename})\n")
            f.write(f"- Comment: {comment}\n\n")


        #joint_df = pd.DataFrame(combined_action_csv)
        #joint_df.to_csv(joint_position_file)

        # Update DataFrame
        df = pd.concat([df, pd.DataFrame([{
            "success": success,
            "duration": t_step,
            "video_filename": save_filename,
            "instruction": instruction,
            "comment": comment
        }])], ignore_index=True)

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    # Save CSV alongside markdown
    csv_filename = markdown_file.replace(".md", ".csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {markdown_file} and {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
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
    # Create one combined image to make live viewing easy
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
