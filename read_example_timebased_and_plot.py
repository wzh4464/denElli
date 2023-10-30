import matplotlib.pyplot as plt
import numpy as np

# read result/newMat_0.npy and result/newMat_1.npy
A = np.load('result/newMat_0.npy')
B = np.load('result/newMat_1.npy')

# copy every rows in A and repeat ten times
# [   0,
#     0,
#     ...
#     0,
#     1,
#     1,
#     ...,
#     1,
#     ...,
#     n,
#     n,
#     ...,
#     n
# ]

for i in range(A.shape[0]):
    for j in range(10):
        if j == 0 and i == 0:
            C = A[i]
        else:
            C = np.vstack((C, A[i]))

# print shape A and shape C
print('shape of A: ' + str(A.shape))
print('shape of C: ' + str(C.shape))

# same for B
for i in range(B.shape[0]):
    for j in range(10):
        if j == 0 and i == 0:
            D = B[i]
        else:
            D = np.vstack((D, B[i]))
            
# plot C and D
fig, axs = plt.subplots(1,2)
axs[0].imshow(C)
axs[1].imshow(D)
fig.set_dpi(600)
plt.savefig('result/nturgb_2.png')
plt.show()
