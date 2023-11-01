'''
File: /dealTimeBasedResult.py
Created Date: Monday October 30th 2023
Author: Zihan
-----
Last Modified: Wednesday, 1st November 2023 11:04:25 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

# result/nturgb/video_0.npy

import sys
import pstats
import numpy as np
import matplotlib.pyplot as plt

# 3 decimal places
np.set_printoptions(precision=3)
# use up the width of the terminal
np.set_printoptions(linewidth=np.inf)
# disable scientific notation
np.set_printoptions(suppress=True)

# Load the data and sort it
p = pstats.Stats('result/profile.txt')
p.sort_stats('cumulative')

# Dump the stats to a binary file
p.dump_stats('result/profile_binary.txt')

# Load the binary stats and redirect the output to a file
p = pstats.Stats('result/profile_binary.txt',
                 stream=open('result/profile_readable.txt', 'w'))
p.sort_stats('cumulative')
p.print_stats()
