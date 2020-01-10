# Program to detect vanishing points in single images

import base64
import urllib

import cv2
from Robust_may_13_tuning import HoughDetect_May_13
from Kalman_Filter import Matrix
import numpy as np


def tri_crop(image, b_left, top, b_right):
    pts = np.array([b_left, top, b_right])
    cropped = image[0:151, 0:151].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    bg = np.ones_like(cropped, np.uint8) * 255  # add white background
    cv2.bitwise_not(bg, bg, mask=mask)

    return cropped


def image_VP(b64img):
    decoded = urllib.parse.unquote(b64img)
    imgdata = base64.b64decode(decoded)
    filename = 'IN/uploaded_image.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    img = cv2.imread('IN/uploaded_image.jpg')
    print("Image size: {0}".format(img.shape[:2]))
    half_w = img.shape[1] / 2
    half_h = img.shape[0] / 2
    # Image Resolution
    res = 1

    # Initial state and covariance matrix
    # x = Matrix([[960.], [330.]])  # 960, 330 are close to half of the original image height and width
    x = Matrix([[half_w * res], [half_h * res]])
    P = Matrix([[10000., 0.], [0., 10000.]])
    VPoint_coord = []

    count_hough = 0
    t1 = []
    while count_hough < 1:
        img_result, VPoint_coord, crop_s, crop_f = HoughDetect_May_13(img, x, P, resolution=res)
        count_hough += 1

    # print('Processing time = {0:.2f} sec'.format(t1) + '\n')
    file_result = 'OUT/' + 'uploaded_image_center' + '.jpg'
    cropped_file_result = 'OUT/' + 'uploaded_image_cropped' + '.jpg'
    # tri_file_result = 'OUT/' + os.path.splitext(os.path.basename(img_file))[0] + '_triangle' + '.jpg'
    y1 = crop_s[1]
    height, width, channels = img.shape
    #print(VPoint_coord)
    crop_img = img[y1:height, 0:width]
    cv2.imwrite(cropped_file_result, crop_img)
    cv2.imwrite(file_result, img_result)

    return VPoint_coord[0], VPoint_coord[1]



# file_path ="."
# print(image_VP(file_path))
