import cv2
import glob
import numpy as np
import sys
import os

FPS = 15
IMG_SIZE = (640, 480)

folder = sys.argv[1]
COLORIZED_ONES = np.ones((480, 640, 3)) * (1, 0, 0)

masks = list(sorted(glob.glob(os.path.join(folder, '*_mask.png')),
                    key=lambda x: int(x.split('/')[-1].split('_')[0])))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter('out.mp4', fourcc, FPS, IMG_SIZE)

for mask in masks:
    img = cv2.imread(mask.replace('_mask.png', '.png'))
    mask = cv2.imread(mask) * COLORIZED_ONES

    out = cv2.addWeighted(img, 1, mask.astype(np.uint8), 0.7, 0)

    video.write(out)

video.release()
