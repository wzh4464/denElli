import big_matrix_cocluster.coclusterSVD as ccSVD
import big_matrix_cocluster.bicluster as bc
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # read data/nturgb/video_0.py
    datasize = 600
    # A = np.load('data/nturgb/video_50.npy')
    
    # print(A.shape): (103, 25, 150)

    # coclusterer = ccSVD.coclusterer(A[50, :, :].reshape(25, 150), 25, 150)
    # result = coclusterer.cocluster(10e-1, 10, 5, True)

    # for frame range(0, 100, 10) do coclustering
    coclusterer = []
    for i in range(0, 70, 10):
        A = np.load('data/nturgb/video_' + str(i) + '.npy')
        coclusterer.append(ccSVD.coclusterer(
            A[i, :, :].reshape(A.shape[1], A.shape[2]), A.shape[1], A.shape[2]))
        result = coclusterer[-1].cocluster(10e-1, 3, 5, True)
        print('frame: ' + str(i))
        # print(result)
        print('--------------------------')

    # draw subplots, left: coclusterer[i].matrix, right: coclusterer[i].newMat
    fig, axs = plt.subplots(len(coclusterer), 2)
    for i in range(len(coclusterer)):
        axs[i, 0].imshow(coclusterer[i].matrix)
        axs[i, 1].imshow(coclusterer[i].newMat)
    plt.show()
