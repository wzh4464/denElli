'''
File: /PhantomDanceLoader.py
Created Date: Tuesday November 14th 2023
Author: Zihan
-----
Last Modified: Tuesday, 14th November 2023 11:36:19 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
14-11-2023		Zihan	init, read `example.json` and transform it to numpy array (6408, 99)
'''

import json
import numpy as np

# 假设 'motion_data.json' 是包含上述 JSON 数据的文件
with open('data/example.json') as file:
    data = json.load(file)

# 提取骨骼名称，根位置和旋转
bone_names = data['bone_name']
root_positions = np.array(data['root_positions'])
rotations = np.array(data['rotations'])

# 输出一些信息以检查数据
print("骨骼名称:", bone_names)
print("根位置数组形状:", root_positions.shape)
print("旋转数组形状:", rotations.shape)

for i in range(5):
    print("第 %d 帧的根位置:" % i, root_positions[i])
    print("第 %d 帧的旋转:" % i, rotations[i])
    print('------------------')

# 根位置数组形状: (6408, 3)
# 旋转数组形状: (6408, 24, 4)

# transform rotation to (frame_size, bone_size * 4)
rotations = rotations.reshape(rotations.shape[0], -1)

# 输出一些信息以检查数据
print("旋转数组形状:", rotations.shape)

for i in range(5):
    print("第 %d 帧的旋转:" % i, rotations[i])
    print('------------------')

# concat root position and rotation
data = np.concatenate((root_positions, rotations), axis=1)

# 输出一些信息以检查数据
print("数据形状:", data.shape)
# 数据形状: (6408, 99)
