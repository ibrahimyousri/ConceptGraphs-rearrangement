from ai2thor.controller import Controller
from pynput import keyboard
import time
import os
import cv2
import numpy as np

# Initialize the AI2-THOR environment with necessary settings
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
    print(event.metadata["objects"][1]["name"])
    print(position)
    print(rotation)
    print(file_counter)

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


# Global variable to track if an object is held
is_object_held = False

# Global variable to track if the 'n' key is pressed
is_n_key_pressed = False

# Map keys to actions including LookUp and LookDown
key_actions = {
    'w': 'MoveAhead',
    's': 'MoveBack',
    'a': 'MoveLeft',
    'd': 'MoveRight',
    'e': {'action': 'RotateRight', 'degrees': 3},
    'q': {'action': 'RotateLeft', 'degrees': 3},
    'u': {'action': 'LookUp', 'degrees': 3},
    'j': {'action': 'LookDown', 'degrees': 3},
    'r': 'PickupObject',  # Key 'r' to pick up an object
    'f': 'ThrowObject',  # Key 'f' to throw an object
    'b': 'ToggleMapView',  # Key 'b' to save frames and trajectory
}


# Function to handle key press
def on_press(key):

    global key_actions, is_object_held, is_n_key_pressed
    try:
        if hasattr(key, 'char') and key.char in key_actions:
            action = key_actions[key.char]

            if action == 'PickupObject':
                # Get the agent's position
                agent_position = controller.last_event.metadata['agent']['position']

                # Get the visible objects from the last event
                visible_objects = controller.last_event.metadata['objects']

                # Filter pickupable objects
                pickupable_objects = [obj for obj in visible_objects if obj['pickupable']]

                # List nearby pickupable objects
                if pickupable_objects:
                    print("\nNearby pickupable objects:")
                    for i, obj in enumerate(pickupable_objects):
                        print(f"{i}: {obj['objectId']}")

                    # Ask the user to choose an object
                    selected_index = input("Enter the number of the object to pick up: ")
                    try:
                        selected_index = int(selected_index)
                        if 0 <= selected_index < len(pickupable_objects):
                            objectId = pickupable_objects[selected_index]['objectId']
                            event = controller.step(action=action, objectId=objectId, forceAction=True)
                            if event.metadata['lastActionSuccess']:
                                is_object_held = True
                        else:
                            print("Invalid selection.")
                    except ValueError:
                        print("Please enter a valid number.")
                else:
                    print("No pickupable object found")

            elif action == 'ThrowObject':
                if is_object_held:
                    # Define the throw force magnitude (adjust as needed)
                    throw_force_magnitude = 20

                    # Perform ThrowObject action with forceAction=True and moveMagnitude
                    event = controller.step(action=action, forceAction=True, moveMagnitude=throw_force_magnitude)
                    if event.metadata['lastActionSuccess']:
                        is_object_held = False
                else:
                    print("No object currently held by the agent")
                    return

            else:
                # For other actions, proceed as before
                if isinstance(action, dict):
                    event = controller.step(action=action['action'], degrees=action['degrees'])
                else:
                    event = controller.step(action=action)

            if 'event' in locals() and not event.metadata['lastActionSuccess']:
                print(f"Action {action} failed")
            if event.metadata['lastActionSuccess'] and is_n_key_pressed:
                save_frames_and_trajectory(event)

            time.sleep(0.1)

        elif hasattr(key, 'char') and key.char == 'x':
            return False  # Stop the listener to exit the program

        # Check if the 'n' key is pressed
        elif hasattr(key, 'char') and key.char == 'n':
            is_n_key_pressed = True
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
