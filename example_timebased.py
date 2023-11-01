'''
File: /example_timebased.py
Created Date: Monday October 30th 2023
Author: Zihan
-----
Last Modified: Wednesday, 1st November 2023 6:13:29 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
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

if __name__ == "__main__":
    # np print style
    # 3 decimal places
    np.set_printoptions(precision=3)
    # use up the width of the terminal
    np.set_printoptions(linewidth=np.inf)
    # disable scientific notation
    np.set_printoptions(suppress=True)

    # read data/nturgb/video_0.py
    datasize = 45
    coclusterer = []
    PARALLEL = True
    DEBUG = True
    # for i in range(datasize):
    #     A = np.load('data/nturgb/video_' + str(i) + '.npy')
    #     # A (103, 25, 150)
    #     # reshape to (103, 25*150)
    #     A = A.reshape(A.shape[0], A.shape[1]*A.shape[2])
    #     coclusterer.append(ccSVD.coclusterer(
    #         A, A.shape[0], A.shape[1]))
    #     result = coclusterer[-1].cocluster(10e-1, 3, 5, True)

    # parallelized
    # result is coclusterer list
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

    B = np.load('data/nturgb/video_0.npy')
    B = B.reshape(B.shape[0], B.shape[1]*B.shape[2])
    subI = np.ones(B.shape[0], dtype=bool)
    subJ = np.ones(B.shape[1], dtype=bool)
    r1, r2 = ccSVD.estimateRank(B, subI, subJ)
    coclusterer = ccSVD.coclusterer(B, B.shape[0], B.shape[1], debug=False)

    def wrapper(args):
        return coclusterer.cocluster(*args)

    if PARALLEL:
        poolArgList = [(10e-2, i, j, True)
                       for i in range(r1, r2) for j in range(r1, r2)]
        with mp.Pool(mp.cpu_count()-4) as p:
            results = list(
                tqdm(p.imap(wrapper, poolArgList), total=len(poolArgList)))

        path = 'result/timebased/r1r2_'

        p.join()
        for i in range(len(results)):
            results[i].printBiclusterList(
                save=True, path=path + str(i // (r2-r1)) + '_' + str(i % (r2-r1)) + '.txt')

    else:
        coclusterer.cocluster(10e-1, 3, 5, True)
        coclusterer.printBiclusterList()
