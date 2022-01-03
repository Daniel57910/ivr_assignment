import numpy as np

COMMAND_1 = "rostopic pub -1 /robot/joint1_position_controller/command std_msgs/Float64"
COMMAND_2 = "rostopic pub -1 /robot/joint3_position_controller/command std_msgs/Float64"
COMMAND_3 = "rostopic pub -1 /robot/joint4_position_controller/command std_msgs/Float64"

rot_1 = np.random.uniform(low=-3, high=3, size=(10,))
rot_2 = np.random.uniform(low=-1.5, high=1.5, size=(10,))
rot_3 = np.random.uniform(low=-1.5, high=1.5, size=(10,))

rotations = np.column_stack([rot_1, rot_2, rot_3])
rotations = rotations.round(1)

for r in rotations:
    com_1 = f'{COMMAND_1} "data: {r[0]}"'
    com_2 = f'{COMMAND_2} "data: {r[1]}"'
    com_3 = f'{COMMAND_3} "data: {r[2]}"'
    print(com_1)
    print(com_2)
    print(com_3)
    print("\n")
