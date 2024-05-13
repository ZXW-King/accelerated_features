import random
import time

import numpy as np
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

from modules.xfeat import XFeat


def draw_point(ref_points, dst_points, img1, img2):
    H,W,C = img2.shape
    img = np.hstack((img1, img2))
    counts = min(len(ref_points),len(dst_points))
    ptL = ref_points.astype(int)
    ptR = np.copy(dst_points).astype(int)
    ptR[:,0] += W
    print("xfeat points:",counts)
    for i in range(counts):
        cv2.circle(img, tuple(ptL[i]), 1, (0, 255, 0), -1, lineType=16)
        cv2.circle(img, tuple(ptR[i]), 1, (0, 255, 0), -1, lineType=16)
    return img

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 255), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]
    print("MNN:",len(matches))
    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=0)

    return img_matches


def draw_match_bf(img1,img2,matches,pts1,pts2):
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in pts1]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in pts2]
    matches = sorted(matches, key=lambda x: x.distance)
    match_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return match_result


if __name__ == '__main__':
    xfeat = XFeat()
    # Load some example images
    im1 = cv2.imread('../assets/ref.png')
    im2 = cv2.imread('../assets/tgt.png')
    # Use out-of-the-box function for extraction + MNN matching
    mkpts_0, mkpts_1 = xfeat.match_xfeat(im1, im2, top_k=1000)
    img = draw_point(mkpts_0,mkpts_1,im1,im2)
    cv2.imshow("img1", img)
    canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
    cv2.imshow("aaa",canvas)

    matches, pts1, pts2 = xfeat.match_xfeat_BFMatcher(im1,im2,top_k=1000)
    img_bf = draw_point(pts1, pts2, im1, im2)
    cv2.imshow("img2", img_bf)
    stereo_img = draw_match_bf(im1,im2,matches,pts1,pts2)
    cv2.imshow("Xfeat", stereo_img)
    cv2.waitKey()
    cv2.destroyAllWindows()