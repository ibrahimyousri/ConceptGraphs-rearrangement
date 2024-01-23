from ai2thor.controller import Controller
from pynput import keyboard
import time
import os
import cv2
import numpy as np

# Initialize the AI2-THOR environment with necessary settings, Enter your desired secene name here:
controller = Controller(scene="FloorPlan207", gridSize=0.25, visibilityDistance=1.5,
                        fieldOfView=90, agentMode='default', renderDepthImage=True,
                        renderObjectImage=False, width=1000, height=1000)
# Camera parameters
fov = 90
width, height = 1000, 1000
cx = (width - 1.) / 2.
cy = (height - 1.) / 2.
f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
print(f, f, cx, cy)

# Directory to save frames and trajectory file
save_dir = 'results'
traj_file = 'traj.txt'
file_counter = 0

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to save frames and trajectory
def save_frames_and_trajectory(event):
    global file_counter

    # Save the RGB frame
    rgb = event.frame
    rgb_filename = os.path.join(save_dir, f'frame{file_counter:06d}.jpg')
    cv2.imwrite(rgb_filename, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

    # Save the depth frame
    depth_frame = event.depth_frame * 1000
    depth_frame_16bit = depth_frame.astype(np.uint16)
    depth_filename = os.path.join(save_dir, f'depth{file_counter:06d}.png')
    cv2.imwrite(depth_filename, depth_frame_16bit)

    # Get agent's position and rotation
    position = event.metadata['agent']['position']
    rotation = event.metadata['agent']['rotation']
    position['y'] += 0.675  # camera y pose


    # Generate transformation matrix
    yaw = np.deg2rad(rotation['y'])
    rotation_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                                [0, 1, 0],
                                [-np.sin(yaw), 0, np.cos(yaw)]])
    T = np.eye(4)
    T[0:3, 0:3] = rotation_matrix
    T[0:3, 3] = [position['x'], position['y'], position['z']]

    # Convert the matrix to scientific notation and flatten
    flattened_matrix = T.flatten()
    formatted_matrix = ' '.join(['{:.16e}'.format(num) for num in flattened_matrix])

    # Write to the trajectory file
    with open(traj_file, 'a') as f_traj:
        f_traj.write(formatted_matrix + '\n')

    file_counter += 1

# Map keys to actions including LookUp and LookDown
key_actions = {
    'w': 'MoveAhead',
    's': 'MoveBack',
    'a': 'MoveLeft',
    'd': 'MoveRight',
    'e': {'action': 'RotateRight', 'degrees': 3},
    'q': {'action': 'RotateLeft', 'degrees': 3},
}

# Function to handle key press
def on_press(key):

    try:
        if key.char in key_actions:
            action = key_actions[key.char]
            # Check if action requires additional parameters like degrees
            if isinstance(action, dict):
                event = controller.step(action=action['action'], degrees=action['degrees'])
            else:
                event = controller.step(action=action)

            if event.metadata['lastActionSuccess']:
                save_frames_and_trajectory(event)
            else:
                print(f"Action {action} failed")
            time.sleep(0.1)
        elif key.char == 'x':
            return False  # Stop the listener to exit the program

    except AttributeError:
        pass

# Start the keyboard listener in non-blocking mode
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Keep the script running
try:
    while listener.is_alive():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass

# Clean up and exit
controller.stop()
print("Exiting the script...")
