'''
File: /PhantomDanceLoader.py
Created Date: Tuesday November 14th 2023
Author: Zihan
-----
Last Modified: Wednesday, 15th November 2023 3:59:57 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
15-11-2023		Zihan	pack the code into a class
14-11-2023		Zihan	init, read `example.json` and transform it to numpy array (6408, 99)
'''

import json
import numpy as np
import iisignature


class PhantomDanceLoader(object):
    """Loads motion data from a JSON file and provides access to bone names, root positions, and rotations.

    Args:
        json_path (str): The path to the JSON file containing the motion data.

    Attributes:
        json_path (str): The path to the JSON file containing the motion data.
        data (dict): The loaded JSON data.
        bone_names (list): The names of the bones.
        root_positions (ndarray): The root positions.
        rotations (ndarray): The rotations.
        rotations_flattened (ndarray): The flattened rotations.

    Methods:
        load(): Loads the motion data from the JSON file.

    Raises:
        AssertionError: If a quaternion is not a unit quaternion.

    Examples:
        loader = PhantomDanceLoader('motion_data.json')
        loader.load()
    """

    def __init__(self, json_path):
        self.json_path = json_path
        self.data = None
        self.bone_names = None
        self.root_positions = None
        self.rotations = None
        self.rotations_flattened = None
        self.signatures = None
        self.featureMat = None
        self.signatureOrder = None
        self.signatures = None
        self.load()
        self.genFeatureMat()
        self.genSignature()

    def load(self):
        """Loads motion data from a JSON file and extracts bone names, root positions, and rotations.

        Raises:
            AssertionError: If a quaternion is not a unit quaternion.
        """
        with open(self.json_path) as file:
            self.data = json.load(file)

        # 提取骨骼名称，根位置和旋转
        self.bone_names = self.data['bone_name']
        self.root_positions = np.array(self.data['root_positions'])
        self.rotations = np.array(self.data['rotations'])

        # assert if [a, b, c, d] is a unit quaternion
        for i in range(self.rotations.shape[0]):
            for j in range(self.rotations.shape[1]):
                assert np.abs(np.linalg.norm(self.rotations[i][j]) - 1) < 1e-3

    def genFeatureMat(self):
        """Transforms the rotations into a feature matrix by flattening and concatenating with root positions.

        Returns:
            ndarray: The feature matrix containing root positions and flattened rotations.
        """
        self.rotations_flattened = self.rotations.reshape(
            self.rotations.shape[0], -1)
        self.featureMat = np.concatenate(
            (self.root_positions, self.rotations_flattened), axis=1)

    def genSignature(self, *args, **kwargs):
        """Calculates signatures for each frame of rotations.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        if kwargs.get('signatureOrder') is None:
            signatureOrder = self.signatureOrder
        self.signatures = [iisignature.sig(
            frame, signatureOrder) for frame in self.rotations]
        
class PhatomDanceDatasetLoader(object):
    '''
    Input a path with many json files, return a dataset with numpy array
    Use PhantomDanceLoader to load each json file
    '''
    pass


if __name__ == '__main__':
    # 假设 'motion_data.json' 是包含上述 JSON 数据的文件
    with open('data/example.json') as file:
        data = json.load(file)

    # 提取骨骼名称，根位置和旋转
    bone_names = data['bone_name']
    root_positions = np.array(data['root_positions'])
    rotations = np.array(data['rotations'])

    # assert if [a, b, c, d] is a unit quaternion
    for i in range(rotations.shape[0]):
        for j in range(rotations.shape[1]):
            assert np.abs(np.linalg.norm(rotations[i][j]) - 1) < 1e-3

    rotations_flattened = rotations.reshape(rotations.shape[0], -1)

    # concat root position and rotation
    data = np.concatenate((root_positions, rotations_flattened), axis=1)

    # 输出一些信息以检查数据
    print("数据形状:", data.shape)
    # 数据形状: (6408, 99)

    # 计算签名的阶数，例如 2
    order = 2

    signatures = [iisignature.sig(frame, order) for frame in rotations]

    print("签名数组形状:", np.array(signatures).shape)
