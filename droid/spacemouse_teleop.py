import cv2
import imageio
import numpy as np
from controllers.spacemouse import SpaceMouseInterface
from robot_env import RobotEnv, RobotEnvConfig
import pyrallis

class InteractiveBot:
    def __init__(self, robot_cfg):
        self.env = RobotEnv()
        self.control_freq = 10

    def reset(self):
        proprio = self.env.observe_proprio()
        ee_pos = proprio.eef_pos
        ee_euler = proprio.eef_euler
        self.env.move_to(
            ee_pos,
            ee_euler,
            1.0,
            control_freq=self.control_freq,
            recorder=None,
        )
        # Then reset
        self.env.reset()

    def run_teleop(self):
        # Reduce sensitivity by using smaller values
        interface = SpaceMouseInterface(
            pos_sensitivity=1.0,    # Reduced from 10.0
            rot_sensitivity=1.0,    # Reduced from 18.0
            action_scale=0.05       # Reduce overall scaling 
        )
        
        # Print raw data during teleop
        # interface.debug_mode(True)
        
        interface.start_control()
        print("\nSpaceMouse controls:")
        print("- Move to control position")
        print("- Twist to control orientation")
        print("- Left button: toggle gripper")
        print("- Right button: reset robot")
        print("- Keyboard 'q' or 'ESC': quit, 'g': toggle gripper, 'r': reset\n")

        frames = []
        while True:
            data = interface.get_controller_state()
            
            # Print all control values when any input is detected
            if np.linalg.norm(data["dpos"]) > 0.001 or np.linalg.norm(data["raw_drotation"]) > 0.001:
                print(f"Position: {data['dpos'].round(3)}")
                print(f"Rotation: {data['raw_drotation'].round(3)}")
                print(f"Buttons - Gripper: {data['grasp']}, Hold: {data['hold']}, Reset: {data['lock']}")
            
            dpos = data["dpos"]
            drot = data["raw_drotation"]

            # Fix Z-axis inversion - invert z axis to match intuitive direction
            dpos = np.array([-dpos[1], dpos[0], -dpos[2]])  # Keep z-negated for intuitive up/down
            drot = np.array([-drot[1], drot[0], drot[2]])   

            hold = int(data["hold"])
            gripper_open = int(1 - float(data["grasp"]))  # binary
            
            proprio = self.env.observe_proprio()

            # Unpack the tuple returned by observe()
            obs, _ = self.env.observe()
            vis = obs['wrist_image']
            vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            
            if np.linalg.norm(dpos) or np.linalg.norm(drot) or hold:
                self.env.apply_action(dpos, drot, gripper_open=gripper_open)
            
            frames.append(vis)
            cv2.imshow('vis', vis)
            key = cv2.waitKey(20)
            
            if key == ord('q') or key == 27:  #  ESC 
                print("Exiting spacemouse teleop...")
                break
            elif key == ord('g'):
                # Toggle gripper with 'g' key for testing
                interface.gripper_is_closed = not interface.gripper_is_closed
                print(f"Gripper {'closed' if interface.gripper_is_closed else 'open'}")
            elif key == ord('r'):
                self.reset()
                print("Robot reset")

        gif_path = 'tmp.gif'
        imageio.mimsave(gif_path, frames, duration=0.06, loop=0)

if __name__ == "__main__":
    # pyrallis是一个Python配置解析库:
    # - 可以从YAML文件加载配置到Python数据类
    # - 支持命令行参数覆盖配置
    # - 提供类型检查和验证
    # - 配置继承和嵌套支持
    robot = InteractiveBot()

    robot.reset()
    robot.run_teleop()
