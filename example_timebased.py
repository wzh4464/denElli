'''
File: /example_timebased.py
Created Date: Monday October 30th 2023
Author: Zihan
-----
Last Modified: Saturday, 4th November 2023 9:41:49 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
4-11-2023		Zihan	600 files
30-10-2023		Zihan	draw bicluster image
30-10-2023		Zihan	tpdm
'''

import big_matrix_cocluster.coclusterSVD as ccSVD
import big_matrix_cocluster.bicluster as bc
import numpy as np
import matplotlib.pyplot as plt
# Pool
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
import time
import os
import sys
from sendEmail import Email

DEBUG = False


def timebasedVideoBicluster(n, i1=4, i2=8, j1=5, j2=8, datasetPath='data/nturgb/', prefix='video_', savePath='result/timebased_3/r1r2_'):
    B = np.load(datasetPath + prefix + str(n) + '.npy')
    B = B.reshape(B.shape[0], B.shape[1] * B.shape[2])
    coclusterer = ccSVD.coclusterer(B, B.shape[0], B.shape[1], debug=False)
    for i in range(i1, i2):
        for j in range(j1, j2):
            coclusterer.cocluster(10e-2, i, j, True)
            coclusterer.printBiclusterList(
                save=True, path=savePath + str(n) + 'at' + str(i) + '_' + str(j) + '.txt')


def cocluster_timebase(i):
    A = np.load('data/nturgb/video_' + str(i) + '.npy')
    # A (103, 25, 150)
    # reshape to (103, 25*150)
    A = A.reshape(A.shape[0], A.shape[1]*A.shape[2])

    if DEBUG:
        print('frame: ' + str(i))
        print(A.shape)
        print('--------------------------')

    coclusterI = ccSVD.coclusterer(
        A, A.shape[0], A.shape[1], debug=True)
    coclusterI.cocluster(10e-1, 3, 5, True)
    return coclusterI


def main():
    # np print style
    # 3 decimal places
    np.set_printoptions(precision=3)
    # use up the width of the terminal
    np.set_printoptions(linewidth=np.inf)
    # disable scientific notation
    np.set_printoptions(suppress=True)

    # read data/nturgb/video_*.py
    PARALLEL = True

    datasetPath = 'data/nturgb/'
    # datasize is numbers of files with name begin with 'video_'
    datasetSize = len([f for f in os.listdir(datasetPath)
                       if os.path.isfile(os.path.join(datasetPath, f)) and f.startswith('video_')])
    print('datasetSize: ' + str(datasetSize))
    # but now use 40 for test
    # datasetSize = 40

    # def wrapper(args):
    #     return coclusterer.cocluster(*args)

    if not PARALLEL:
        timebasedVideoBicluster(0)

    else:
        poolArgList = [(i) for i in range(datasetSize)]
        with mp.Pool(mp.cpu_count()-4) as p:
            results = list(
                tqdm(p.imap(timebasedVideoBicluster, poolArgList), total=len(poolArgList)))

        p.join()
        print('Done!')


if __name__ == "__main__":
    finisherEmail = Email()
    try:
        main()
    except Exception as e:
        finisherEmail.setContent(str(e))
        finisherEmail.setSubject('Error in example_timebased.py')
        finisherEmail.send()
        raise e
    else:
        finisherEmail.setContent('Finished')
        finisherEmail.setSubject('Finished example_timebased.py')
        finisherEmail.send()
