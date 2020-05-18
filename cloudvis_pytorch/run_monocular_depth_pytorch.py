#!/usr/bin/python3

import sys
import cv2
from cloudvis import CloudVis

network_path = "/home/gsamvedi/code/unsupervised_depth_estimation_converted/pytorch_version/src"
sys.path.append(network_path)

from run_pytorch_model import RunPytorchModel
# from  run_pytorch_model import RunPytorchModel

PORT = 6003


def callback(request, response, data):
    img = request.getImage('image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response.addImage('result', gray)


if __name__ == '__main__':
    cloudvis = CloudVis(PORT)
    cloudvis.run(callback)
