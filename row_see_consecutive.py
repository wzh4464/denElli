'''
File: /row_see_consecutive.py
Created Date: Wednesday November 1st 2023
Author: Zihan
-----
Last Modified: Thursday, 2nd November 2023 10:49:14 am
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


class Result:
    def __init__(self, i, j, numbers):
        self.i = i
        self.j = j
        self.numbers = numbers
        self.timeLength = 103  # TODO: change this

    def __str__(self):
        string1 = f'================'
        string2 = f'({self.i}, {self.j})'
        string3 = f'----------------'
        stringList = [string1, string2, string3]
        for i in self.numbers:
            stringList.append(str(patchNumbers(i)))
            stringList.append(str(i))
            stringList.append('------')
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
            print(len(numbers))

            # 将子列表转换为元组，然后转换为集合以删除重复项
            unique_numbers_set = set(tuple(sublist) for sublist in numbers)

            # 将元组转换回列表
            unique_numbers = [list(sublist) for sublist in unique_numbers_set]

            print(len(unique_numbers))
            print('-----------------')

            r = Result(i, j, unique_numbers)
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
# print(numbers)

x = range(2, 7)
y = range(5, 8)

fig = plt.figure(figsize=(20, 10))
for i in x:
    for j in y:
        # print(numbers[i][j])
        if numbers[i][j] is not None:
            ax = fig.add_subplot(len(x), len(y), (i - min(x))*len(y) + (j - min(y)) + 1)
            ax.imshow(numbers[i][j].subFigure(), aspect='4')
            ax.set_title(f'({i}, {j})')

plt.show()
