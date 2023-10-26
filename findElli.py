#!/home/zihan/miniconda3/envs/cocluster/bin/python
'''
File: /findElli.py
Created Date: Tuesday October 10th 2023
Author: Zihan
-----
Last Modified: Thursday, 12th October 2023 11:17:42 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import ELSDc.elsdc as el
import big_matrix_cocluster.submatrix as sm
import big_matrix_cocluster.coclusterSVD as ccSVD
import big_matrix_cocluster.bicluster as bc

import os

DEBUG = False
# DEBUG = True
count = 0

class Ellipse:
    '''
    A class for holding an ellipse
    '''

    def __init__(self, ring: el.Ring, img: np.ndarray):
        '''
        label: int
        center: np.ndarray
        axis: np.ndarray
        angle: float
        start_angle: float
        end_angle: float
        points: np.ndarray
        '''
        self.label = ring.label
        self.center = np.stack((ring.cx, ring.cy))
        self.long_axis = ring.ax
        self.short_axis = ring.bx
        self.angle = ring.theta
        self.start_angle = ring.ang_start
        self.end_angle = ring.ang_end
        self.points = getPoints(img, ring)
        self.c, self.f1, self.f2 = self.calFoci()
        self.inner_points = self.getInnerPoints(img)

    def calFoci(self) -> [float, np.ndarray]:
        '''
        Calculate the foci of the ellipse
        '''
        # calculate the foci
        c = np.sqrt(self.long_axis ** 2 - self.short_axis ** 2)
        f1 = self.center + c * \
            np.array([np.cos(self.angle), np.sin(self.angle)])
        f2 = self.center - c * \
            np.array([np.cos(self.angle), np.sin(self.angle)])

        return [c, f1, f2]

    def getInnerPoints(self, img) -> np.ndarray:
        xrangemin = np.floor(self.center[0]).astype(
            int) - np.ceil(self.long_axis).astype(int)
        xrangemax = np.floor(self.center[0]).astype(
            int) + np.ceil(self.long_axis).astype(int)
        yrangemin = np.floor(self.center[1]).astype(
            int) - np.ceil(self.long_axis).astype(int)
        yrangemax = np.floor(self.center[1]).astype(
            int) + np.ceil(self.long_axis).astype(int)

        # consider the img_size
        xrangemin = max(xrangemin, 0)
        xrangemax = min(xrangemax, img.shape[1])
        yrangemin = max(yrangemin, 0)
        yrangemax = min(yrangemax, img.shape[0])
        
        # get the points inside the ellipse
        pool = Pool(mp.cpu_count())
        results = pool.starmap(self.isInside, [(x, y) for x in range(xrangemin, xrangemax) for y in range(yrangemin, yrangemax)])
    
        # fill results to a matrix
        results = np.array(results).reshape(xrangemax - xrangemin, yrangemax - yrangemin)
        
        points = []
        
        # get the points inside the ellipse
        for x in range(xrangemin, xrangemax):
            for y in range(yrangemin, yrangemax):
                if results[x - xrangemin, y - yrangemin]:
                    points.append([x, y])
    
        return np.array(points)

    def isInside(self, x, y) -> bool:
        '''
        Check if the point (x, y) is inside the ellipse
        '''
        # if |(x, y) - f1| + |(x, y) - f2| <= 2a for both ellipses
        one = np.linalg.norm(np.array(
            [x, y]) - self.f1) + np.linalg.norm(np.array([x, y]) - self.f2) <= 2 * self.long_axis
        return one


def getPoints(img: np.ndarray, ellipse: el.Ring) -> np.ndarray:
    '''
    Get the points inside the ellipse
    '''
    # get the center and the axis of the ellipse
    label = ellipse.label

    # find values equal to the label
    pointsx, pointsy = np.where(img == label)

    return np.vstack((pointsx, pointsy)).T

def compatibility(e1: Ellipse, e2: Ellipse, img_size) -> float:
    '''
    Calculate the compatibility between two ellipses
    '''

    # simple choice is do the iou
    # area e1 and e2
    interArea = 0
    unionArea = 0

    xrangemin = np.min(np.array([np.floor(e1.center[0]).astype(int) - np.ceil(e1.long_axis).astype(
        int), np.floor(e2.center[0]).astype(int) - np.ceil(e2.long_axis).astype(int)]))
    xrangemax = np.max(np.array([np.floor(e1.center[0]).astype(int) + np.ceil(e1.long_axis).astype(
        int), np.floor(e2.center[0]).astype(int) + np.ceil(e2.long_axis).astype(int)]))
    yrangemin = np.min(np.array([np.floor(e1.center[1]).astype(int) - np.ceil(e1.long_axis).astype(
        int), np.floor(e2.center[1]).astype(int) - np.ceil(e2.long_axis).astype(int)]))
    yrangemax = np.max(np.array([np.floor(e1.center[1]).astype(int) + np.ceil(e1.long_axis).astype(
        int), np.floor(e2.center[1]).astype(int) + np.ceil(e2.long_axis).astype(int)]))

    # consider the img_size
    xrangemin = max(xrangemin, 0)
    xrangemax = min(xrangemax, img_size[1])
    yrangemin = max(yrangemin, 0)
    yrangemax = min(yrangemax, img_size[0])

    # debug: plot the intersection area and union area
    if DEBUG:
        blank = np.zeros([img_size[0], img_size[1], 3], dtype=np.uint8)

    for x in range(xrangemin, xrangemax):
        for y in range(yrangemin, yrangemax):
            # if |(x, y) - f1| + |(x, y) - f2| <= 2a for both ellipses
            one = np.linalg.norm(np.array(
                [x, y]) - e1.f1) + np.linalg.norm(np.array([x, y]) - e1.f2) <= 2 * e1.long_axis
            two = np.linalg.norm(np.array(
                [x, y]) - e2.f1) + np.linalg.norm(np.array([x, y]) - e2.f2) <= 2 * e2.long_axis
            if one or two:
                unionArea += 1
                if DEBUG:
                    if not (one and two):
                        blank[y, x, :] = [0, 255, 0]
            if one and two:
                interArea += 1
                if DEBUG:
                    blank[y, x, :] = [0, 0, 255]

    if DEBUG:
        cv2.imwrite('debug.png', blank)
        print("interArea = {}, unionArea = {}".format(interArea, unionArea))

    # print two labels as {a, b}
    # print("label = {{{}, {}}}".format(e1.label, e2.label))
    return interArea / unionArea


def extract_skeleton_from_points(points, img_size=(500, 500)):
    # 创建一个空白的二值图像
    img = np.zeros(img_size, np.uint8)

    # 在图像上标记点
    for x, y in points:
        img[x, y] = 255

    # 提取骨架
    skeleton_img = cv2.ximgproc.thinning(
        img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 获取骨架上的点的坐标
    out_points = np.column_stack(np.where(skeleton_img > 0))

    return out_points.tolist()


def calComMat(e1 : Ellipse, e2 : Ellipse) -> float:
    '''
    Calculate the compatibility matrix
    '''
    global count
    points_1 = e1.inner_points
    points_2 = e2.inner_points

    set_1 = set(tuple(x) for x in points_1)
    set_2 = set(tuple(x) for x in points_2)
    # ex: points_1 = [[1, 2], [3, 4], [5, 6]]
    #     points_2 = [[2, 2], [3, 4]]
    #     intersection = [[3, 4]]
    
    intersection_set = set_1 & set_2
    intersection = [list(x) for x in intersection_set]
    
    count += 1
    result = len(intersection) / (len(points_1) + len(points_2) - len(intersection))
    if result > 1:
        error = "result = {}, len(intersection) = {}, len(points_1) = {}, len(points_2) = {}".format(result, len(intersection), len(points_1), len(points_2))
        raise ValueError(error)
    # print("label = {{{}, {}}}, result = {}".format(e1.label, e2.label, result))
    # monitorCount()
    
    return result

def monitorCount():
    '''
    Monitor the count
    '''
    global count
    # while True:
    print("progress: {}".format(count))
        # time.sleep(5)


# if main
if __name__ == '__main__':
    datafolder = 'data/'
    targetfolder = 'result/'
    img = cv2.imread(datafolder + 'image/006.jpg')

    ellipses, polygons, out_img = el.detect_primitives(img)
    # points = getPoints(out_img, ellipses[0])
    # points = extract_skeleton_from_points(getPoints(out_img, ellipses[0]), img.shape[:2])

    if DEBUG:
        ellipse = Ellipse(ellipses[0], out_img)
        print("a ={}, b = {}, c = {}, f1 = {}, f2 = {}".format(
            ellipse.long_axis, ellipse.short_axis, ellipse.c, ellipse.f1, ellipse.f2))
        print(calComMat(ellipse, ellipse))

    # for each two ellipses, calculate the compatibility
    # for i in range(len(ellipses)):
    #     for j in range(i + 1, len(ellipses)):
    #         ComMat[i, j] = compatibility(Ellipse(ellipses[i], out_img), Ellipse(ellipses[j], out_img), img.shape[:2])

    # print(len(ellipses) ** 2)
    # fork a process, every 5 seconds output the timestamp and count/total
    # p = mp.Process(target=monitorCount)
    # p.start()
    

    # ElliResult = pool.starmap(Ellipse, [(ellipses[i], out_img) for i in range(len(ellipses))])
    
    # # save ElliResult to Ellipses
    # while len(ElliResult) > 0:
    #     Ellipses.append(ElliResult.pop(0))
        
    # print("len(Ellipses) = {}".format(len(Ellipses)))
        
    
    # if tasks.npy not exists
    if not os.path.exists(targetfolder + 'tasks.npy'):
        ComMat = np.zeros([len(ellipses), len(ellipses)])
        # parrelize the compatibility calculation
        cpu_count = mp.cpu_count()
        pool = Pool(cpu_count)
        
        # initialize the Ellipses
        Ellipses = []
        for i in range(len(ellipses)):
            Ellipses.append(Ellipse(ellipses[i], out_img))
            print("{}/{} Ellipses initialized".format(i + 1, len(ellipses)))
        
        # results = pool.starmap(calComMat, [Ellipses[i] for i in range(len(Ellipses)) for j in range(i + 1, len(Ellipses))])
        k = len(Ellipses)
        tasks = []
        # task_count = 0
        for i in range(k):
            for j in range(i + 1, k):
                tasks.append((Ellipses[i], Ellipses[j]))
            
        # save tasks to file, tasks are not pickable
        pass
    else:
        pass
            
    # if ComMat.npy exists
    if os.path.exists(targetfolder + 'ComMat.npy'):
        ComMat = np.load(targetfolder + 'ComMat.npy')
        # get the count
    else:
        results = pool.starmap(calComMat, tasks)
        # calComMat(tasks[0][0], tasks[0][1])

        # save the results to the ComMat
        for i in range(len(ellipses)):
            for j in range(i + 1, len(ellipses)):
                ComMat[i, j] = results.pop(0)
                ComMat[j, i] = ComMat[i, j]

        # save the ComMat
        # p.terminate()
        print("len(CalMat) = {}".format(len(ComMat)))
        np.save(targetfolder + 'ComMat.npy', ComMat)
    
    # imshow the ComMat and save it
    plt.imshow(ComMat)
    plt.savefig(targetfolder + 'ComMat.png')
