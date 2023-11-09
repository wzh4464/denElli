'''
File: /row_see_consecutive.py
Created Date: Wednesday November 1st 2023
Author: Zihan
-----
Last Modified: Thursday, 9th November 2023 10:53:28 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import glob
import re
from matplotlib import pyplot as plt
import numpy as np
# import opencv
import cv2


class Result:
    def __init__(self, i, j, numbers):
        self.i = i
        self.j = j
        self.numbers = numbers
        self.timeLength = 103  # TODO: change this

    def __str__(self):
        stringList = []
        stringList.append('i: ' + str(self.i))
        stringList.append('j: ' + str(self.j))
        for i in self.numbers:
            stringList.append('patch:' + str(patchNumbers(i)))
            stringList.append('numbers: ' + str(i))
            # stringList.append('------')
        string4 = '\n'.join(stringList)

        return string4

    def __repr__(self):
        return str(self)

    def subFigure(self):
        """Return the subfigure of the result.

        Returns:
            list: A list of numbers.
        """
        canvas = np.zeros((self.timeLength, len(self.numbers)))
        for i in range(len(self.numbers)):
            for j in self.numbers[i]:
                canvas[j][i] = 1

        return canvas.T


def plotTimeDomain(numbers, x, y):
    fig = plt.figure(figsize=(20, 10))
    for i in x:
        for j in y:
            # print(numbers[i][j])
            if numbers[i][j] is not None:
                ax = fig.add_subplot(len(x), len(
                    y), (i - min(x))*len(y) + (j - min(y)) + 1)
                ax.imshow(numbers[i][j].subFigure(), aspect='4')
                ax.set_title(f'({i}, {j})')

    plt.show()


def extract_numbers(directory):
    # 找到指定目录中的所有.txt文件
    files = glob.glob(f'{directory}/*.txt')

    # 2-dimensional list (9*9)
    all_numbers = [[None for _ in range(9)] for _ in range(9)]

    # 遍历每个文件
    for file in files:
        # filename r1r2_0_0.txt, then i = 0 and j = 0
        name = file.split('/')[-1]
        i = int(name[5])
        j = int(name[7])
        # print(i, j)
        with open(file, 'r') as f:
            content = f.read()
            # 使用正则表达式查找所有的数字
            numbers = re.findall(r'row members \[([0-9, ]+)\]', content)
            # 将字符串转换为整数列表
            numbers = [list(map(int, i.split(','))) for i in numbers]
            # print(len(numbers))

            # 将子列表转换为元组，然后转换为集合以删除重复项
            unique_numbers_set = set(tuple(sublist) for sublist in numbers)

            # 将元组转换回列表
            unique_numbers = [list(sublist) for sublist in unique_numbers_set]

            # print(len(unique_numbers))
            # print('-----------------')

            r = Result(i, j, unique_numbers)
            # print(r)
            all_numbers[i][j] = r

    return all_numbers


def patchNumbers(numberList):
    """Given a number list, calculate how many consecutive patches it has.

    Args:
        numberList (list): A list of numbers.

    Returns:
        int: The number of consecutive patches.
    """
    # check if the list is int list
    if not all(isinstance(x, int) for x in numberList):
        raise TypeError('The list should only contain integers.')

    # Order the list in ascending order
    numberList.sort()

    # Calculate the number of consecutive patches
    count = 1
    for i in range(len(numberList)-1):
        if numberList[i+1] - numberList[i] > 1:
            count += 1

    return count


# 用法示例:
directory = 'result/timebased_2'
numbers = extract_numbers(directory)

x = range(2, 7)
y = range(5, 8)
# for i in x:
#     for j in y:
#         print(numbers[i][j])
#         print('-----------------')

# plotTimeDomain(numbers, x, y)

# import video from data/nturgb/video/video_0.avi

cap = cv2.VideoCapture('data/nturgb/video/video_0.avi')

# extract the frames from the video
# frames = [frame1, frame2, ...]
frames = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
    else:
        break

# print(len(frames))
print('length of frames: ', len(frames))

# use numbers[4][5] to mask the frames
groups = numbers[3][5].numbers
print(groups)

# build a map from i to group index
i2g = {}
for i in range(len(groups)):
    for j in groups[i]:
        i2g[j] = i

# random a color list (length = len(groups))
colorList = np.random.randint(0, 255, (len(groups), 3))

# for i in range(len(groups)):
#     for j in groups[i]:
#         # make frames j in cap transparently masked by colorList[i]
#         frames[j] = cv2.addWeighted(
#             frames[j], 0.5, np.zeros(frames[j].shape, frames[j].dtype), 0.5, 0)
new_frames = []
cap.release()
cap = cv2.VideoCapture('data/nturgb/video/video_0.avi')
frame_number = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_number += 1
        mask = np.zeros_like(frame)
        # color of mask is colorList[i2g[frame_number]]
        try:
            mask[:] = colorList[i2g[frame_number]]
        except:
            pass
        # make frames j in cap transparently masked by colorList[i]
        frame = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)
        new_frames.append(frame)
        cv2.imshow('frame', frame)
        # wait 5ms
        # cv2.waitKey(50)
            
    else:
        break

# save the new frames to a video
height, width, layers = new_frames[0].shape
size = (width, height)
out = cv2.VideoWriter('result/nturgb/video/video_0_masked.avi', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
for i in range(len(new_frames)):
    out.write(new_frames[i])
out.release()
cap.release()
cv2.destroyAllWindows()
