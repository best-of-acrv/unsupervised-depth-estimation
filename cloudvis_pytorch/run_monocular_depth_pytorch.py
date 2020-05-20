#!/usr/bin/python3

from cloudvis import CloudVis
from run_network import RunModel

PORT = 6003


def disparityToBW(disparity_img):
    disparity_img_b = disparity_img.numpy()
    disparity_img_bw = disparity_img_b / 30.0
    disparity_img_bw = disparity_img_bw * 255

    return disparity_img_bw


def callback(request, response, data):
    img = request.getImage('image')
    print('Image received')

    disparity_img = unsupervised_network.run(img)
    disparity_img_bw = disparityToBW(disparity_img)

    response.addImage('result', disparity_img_bw)


if __name__ == '__main__':
    cloudvis = CloudVis(PORT)
    unsupervised_network = RunModel()
    cloudvis.run(callback)
