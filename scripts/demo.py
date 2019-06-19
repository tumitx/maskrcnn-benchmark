import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import time
import os

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "configs/shufa_cfgs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=300,
    confidence_threshold=0.7,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")


# from http://cocodataset.org/#explore?id=345434
for imname in os.listdir('/data/16_AIshufa/imgs_out/1/'):
    if imname[-4:] != '.png':
        continue
    image = cv2.imread(os.path.join('/data/16_AIshufa/imgs_out/1/', imname))
    starttime = time.time()
    # compute predictions
    predictions = coco_demo.run_on_opencv_image(image)
    endtime = time.time()
    print('{}\ttime: {}'.format(imname, (endtime - starttime)))
    cv2.imwrite(os.path.join('shufa_out', imname),predictions)

# # set up demo for keypoints
# config_file = "configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml"
# cfg.merge_from_file(config_file)
# cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.merge_from_list(["MODEL.MASK_ON", False])

# coco_demo = COCODemo(
#     cfg,
#     min_image_size=800,
#     confidence_threshold=0.7,
# )

# # run demo
# image = load("http://farm9.staticflickr.com/8419/8710147224_ff637cc4fc_z.jpg")
# starttime = time.time()
# predictions = coco_demo.run_on_opencv_image(image)
# endtime = time.time()
# print(endtime - starttime)
# cv2.imwrite('kp_prediction.jpg',predictions)