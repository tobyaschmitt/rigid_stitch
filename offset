###################################
#
# Gets the offset of two images with some overlap that are to be rigidly stitched together.
#
# Change file_1 and file_2 to be the images you want to stitch together.
#
###################################

file_1 = '0_1.png'

file_2 = '0_2.png'


###################################




import PIL
from PIL import Image
import os, shutil
import time
import tkinter
import statistics

import matplotlib.pyplot as plt

import scipy

import imutils
from imutils import paths
import numpy as np
import argparse
import cv2

import stitch2d

cv2.ocl.setUseOpenCL(False)

feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'knn'

def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using a specific method
    """
    
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create(nfeatures=2000)
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)



def createMatcher(method,crossCheck):
    '''
    Create and return a Matcher Object
    '''

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)

    max_matches = 60
    current_matches = 0
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio and len(matches) < 60:
            matches.append(m)
    print("Matches (knn):", len(matches))
    return matches


def get_matches(img_1, img_2, feature_extractor='orb', feature_matching='knn'):
    '''
    img_1 and img_2 are cv2.imread() images
    '''

    img_1_gray= cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
    img_2_gray= cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)


    kpsA, featuresA = detectAndDescribe(img_1_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(img_2_gray, method=feature_extractor)



    



    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)

    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])


    ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

    return (ptsA, ptsB)

def offset_xy(img_1, img_2):
    ptsA, ptsB = get_matches(img_1, img_2)
    #print("matches:")
    #print(ptsA)
    #print(ptsB)
    differences_x = []
    differences_y = []
    for point_no in range(len(ptsA)):
      differences_x.append(int(ptsB[point_no][0] - ptsA[point_no][0]))
      differences_y.append(int(ptsB[point_no][1] - ptsA[point_no][1]))

    print(differences_x)
    print(differences_y)

    return (statistics.mode(differences_x), statistics.mode(differences_y))





print(offset_xy(cv2.imread('0_1.png'), cv2.imread('0_2.png')))
