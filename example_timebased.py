import big_matrix_cocluster.coclusterSVD as ccSVD
import big_matrix_cocluster.bicluster as bc
import numpy as np
import matplotlib.pyplot as plt
# Pool
from multiprocessing import Pool
import multiprocessing as mp

if __name__ == "__main__":
    # read data/nturgb/video_0.py
    datasize = 600
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
    def cocluster(i):
        A = np.load('data/nturgb/video_' + str(i) + '.npy')
        # A (103, 25, 150)
        # reshape to (103, 25*150)
        A = A.reshape(A.shape[0], A.shape[1]*A.shape[2])
        coclusterer.append(ccSVD.coclusterer(
            A, A.shape[0], A.shape[1]))
        coclusterer[-1].cocluster(10e-1, 3, 5, True)
        return True

    if PARALLEL:
        with Pool(mp.cpu_count()-5) as p:
            p.map(cocluster, range(datasize))
    else:
        # just do it for video_0
        cocluster(0)

    fig, axs = plt.subplots(len(coclusterer), 2)
    for i in range(len(coclusterer)):
        axs[i, 0].imshow(coclusterer[i].matrix)
        axs[i, 1].imshow(coclusterer[i].newMat)
    # plt.show()
    # save to result/
    plt.savefig('result/nturgb.png')
