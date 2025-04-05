import cv2
import imageio
import numpy as np
from controllers.spacemouse import SpaceMouseInterface
from openpi_client import image_tools
from robot_env import RobotEnv
from PIL import Image
import dataclasses

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "25455306" # e.g., "24259877"
    right_camera_id: str = "27085680" # fix: "27085680"  move: # "26368109"  
    wrist_camera_id: str = "14436910"  # e.g., "13062452"

    # Policy parameters
    external_camera: str | None = ( 
        "right"  # which external camera should be fed to the policy, choose from ["left", "right"]
    )


POLICY_SIZE = 224
DISPLAY_SIZE = POLICY_SIZE * 3


class InteractiveBot:
    def __init__(self):
        self.env = RobotEnv()
        self.control_freq = 10

    def reset(self):
        self.env.reset()

    def run_teleop(self):
        interface = SpaceMouseInterface(
            pos_sensitivity=10.0,   # 增加位置灵敏度
            rot_sensitivity=10.0,   # 增加旋转灵敏度
            action_scale=0.1      # 增加总体动作量
        )
        
        # TODO check left, right button
        # right button: reset robot, solved
        # left button: toggle gripper, not working
        # TODO : mapping directly into action space?
        # it seems moving mouse will cause gripper to toggle
        interface.start_control()
        print("\nSpaceMouse controls:")
        print("- Move to control position")
        print("- Twist to control orientation")
        print("- Left button: toggle gripper")
        print("- Right button: reset robot")
        print("- Keyboard 'q' or 'ESC': quit, 'g': toggle gripper, 'r': reset\n")

        args = Args()
        frames = []
        
        while True:
            data = interface.get_controller_state()
            
            # Print all control values when any input is detected
            if np.linalg.norm(data["dpos"]) > 0.001 or np.linalg.norm(data["raw_drotation"]) > 0.001:
                print(f"Position: {data['dpos'].round(3)}")
                print(f"Rotation: {data['raw_drotation'].round(3)}")
                print(f"Buttons - Gripper: {data['grasp']}, Hold: {data['hold']}, Reset: {data['lock']}")
            
            if data["lock"]:
                self.reset()
                continue

            dpos = data["dpos"]
            drot = data["raw_drotation"]

            # Fix Z-axis inversion - invert z axis to match intuitive direction
            dpos = np.array([-dpos[1], dpos[0], -dpos[2]])  # Keep z-negated for intuitive up/down
            drot = np.array([-drot[1], drot[0], drot[2]])   

            hold = int(data["hold"])
            gripper_open = int(1 - float(data["grasp"]))  # binary
            
    
            if np.linalg.norm(dpos) or np.linalg.norm(drot) or hold:
                # 使用env.step替代update_robot
                action = np.concatenate([dpos, drot, [gripper_open]])
                # Clip action values to [-1, 1] range
                action = np.clip(action, -1.0, 1.0)

                self.env.step(action)
            
            _vis_robot_views(self.env.get_observation(), args, frames)
            
            key = cv2.waitKey(20)
            
            if key == ord('q') or key == 27:  #  ESC 
                print("Exiting spacemouse teleop...")
                break
            elif key == ord('g'):
                # Toggle gripper with 'g' key for testing
                gripper_open = 1 - gripper_open
                print(f"Gripper {'closed' if gripper_open == 0 else 'open'}")
            elif key == ord('r'):
                self.reset()
                print("Robot reset")

        gif_path = './tmp.gif'
        imageio.mimsave(gif_path, frames, duration=0.06, loop=0)



def _vis_robot_views(obs_dict, args, frames):
    curr_obs = _extract_observation(args, obs_dict, save_to_disk=False)
    
    wrist_image = curr_obs["wrist_image"]
    external_camera = args.external_camera or "right"
    external_image = curr_obs[f"{external_camera}_image"]
    
    # Resize images to policy size first
    wrist_image = image_tools.resize_with_pad(wrist_image, POLICY_SIZE, POLICY_SIZE)
    external_image = image_tools.resize_with_pad(external_image, POLICY_SIZE, POLICY_SIZE)
    
    # Then resize to display size
    wrist_image = cv2.resize(wrist_image, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    external_image = cv2.resize(external_image, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Combine and convert color space
    combined_image = np.concatenate([wrist_image, external_image], axis=1)
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    
    frames.append(combined_image)
    cv2.imshow('Robot Views', combined_image)
    return

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
    print("Starting spacemouse teleop...")
    robot = InteractiveBot()
    robot.reset()
    robot.run_teleop()
