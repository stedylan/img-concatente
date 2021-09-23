import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

# template matching method
def stitch(img_prev, img_curr, img):
    ratio=15

    if img_prev is None or img_curr is None:
        raise FileNotFoundError
    if img_prev.shape != img_curr.shape:
        raise ValueError

    # to ignore sidebar, shifting 20px both left and right
    img1 = img_prev[:, 20:img_prev.shape[1] - 20, :]
    img2 = img_curr[:, 20:img_curr.shape[1] - 20, :]

    if img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2

    h, w = gray1.shape

    thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    sub = thresh1 - thresh2
    sub = cv2.medianBlur(sub, 3)
    sub = sub // 255

    height = h
    min_height = h*0.97
    # if the white pixel < thresh, we thought they are the same area(bottom status bar) 
    thresh = w // 10
    for i in range(h - 1, 0, -1):
        if np.sum(sub[i]) > thresh and height < min_height:
            break
        height = height - 1
    # block is the pre-defined template
    block = sub.shape[0] // ratio
    templ = gray1[height - block:height, ]
    # don't pocess similar picture
    if templ.shape[0] < block:
        return img

    res = cv2.matchTemplate(gray2, templ, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if min_val < 0.05:
        return draw(img_prev, img_curr, img, min_loc[1] / h, h - height + block)
    else:
        return img
    
    
def draw(img1, img2, img=None, top=0, bottom_height=0):
    if img is None:
        img = img1

    h, w, c = img2.shape

    top = int(top * h)

    roi2 = img2[top:h, ]
    roi = img[:-bottom_height, ]

    image = np.concatenate((roi, roi2), axis=0)

    return image

def from_video(path):
    (name, ext) = os.path.splitext(path)
    cap = cv2.VideoCapture(path)
    fs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img = None

    if cap.isOpened():
        ret, prev = cap.read()
        for i in tqdm(range(1, fs)):
            ret, curr = cap.read()
            if ret==True:
                if i%20 == 0:
                    img = stitch(prev, curr, img)
                    prev = curr

    cv2.imwrite('{}_concat.jpg'.format(name), img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='record', help='record or images')
    parser.add_argument('--data', default='test.mp4', help='record or images')

    opt = parser.parse_args()
    if opt.mode == 'record':
        from_video(opt.data)
    
