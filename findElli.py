#!/home/zihan/miniconda3/envs/cocluster/bin/python
'''
File: /findElli.py
Created Date: Tuesday October 10th 2023
Author: Zihan
-----
Last Modified: Tuesday, 10th October 2023 9:48:43 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import big_matrix_cocluster.bicluster as bc
import big_matrix_cocluster.coclusterSVD as ccSVD
import big_matrix_cocluster.submatrix as sm
import ELSDc.elsdc as el


import numpy as np
import cv2

import matplotlib.pyplot as plt

class Ellipse:
    '''
    A class for holding an ellipse
    '''
    def __init__(self, ring : el.Ring, img : np.ndarray):
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

def getPoints(img : np.ndarray, ellipse : el.Ring) -> np.ndarray:
    '''
    Get the points inside the ellipse
    '''
    # get the center and the axis of the ellipse
    label = ellipse.label
    
    # find values equal to the label
    pointsx, pointsy = np.where(img == label)
    
    return np.vstack((pointsx, pointsy)).T

def compatibility(e1 : Ellipse, e2 : Ellipse) -> float:
    '''
    Calculate the compatibility between two ellipses
    '''
    
    

# if main
if __name__ == '__main__':
    datafolder = 'data/'
    targetfolder = 'result/'
    img = cv2.imread(datafolder + 'image/001.jpg')
    
    ellipses, polygons, out_img = el.detect_primitives(img)
    points = getPoints(out_img, ellipses[0])
    print(points)
