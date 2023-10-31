'''
File: /example_timebased.py
Created Date: Monday October 30th 2023
Author: Zihan
-----
Last Modified: Tuesday, 31st October 2023 10:42:26 am
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

if __name__ == "__main__":
    # np print style
    # 3 decimal places
    np.set_printoptions(precision=3)
    # use up the width of the terminal
    np.set_printoptions(linewidth=np.inf)

    # read data/nturgb/video_0.py
    datasize = 45
    coclusterer = []
    PARALLEL = False
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
        coclusterI = ccSVD.coclusterer(
            A, A.shape[0], A.shape[1], debug=True)
        coclusterI.cocluster(10e-1, 3, 5, True)
        return coclusterI

    def cocluster(i):
        A = np.load('data/nturgb/video_0.npy')
        # A (103, 25, 150)

        A = A[0, :, :]
        coclusterI = ccSVD.coclusterer(
            A, A.shape[0], A.shape[1], debug=True)
        coclusterI.cocluster(10e-1, 3, 5, True)
        coclusterI.saveNewMat('result/nturgb/video_' + str(i) + '.npy')
        return coclusterI

    if PARALLEL:
        with Pool(mp.cpu_count()-4) as p:
            # p.map(cocluster, range(datasize))
            # use tqdm to show progress
            results = list(
                tqdm(p.imap(cocluster_timebase, range(datasize)), total=datasize))
        fig, axs = plt.subplots(len(results), 2)
        for i in range(len(results)):
            axs[i, 0].imshow(results[i].matrix)
            axs[i, 1].imshow(results[i].newMat)
        # plt.show()
        # save to result/
        plt.savefig('result/nturgb.png')
    else:
        # show original matrix first
        # show 10*10 first of them
        # use numpy to print
        A = np.load('data/nturgb/video_0.npy')
        # print 10*10 of A
        print(A[0, 0:25, 0:25])
        ccl = cocluster_timebase(0)

        ccl.printBiclusterList()
        ccl.imageShowBicluster(save=True, filename='result/nturgb_bicluster.png')
