#!/usr/bin/python3

from cloudvis import CloudVis
from run_network import RunModel


class RunMonocularDepthPytorch:
    def __init__(self):
        self.port_number = 6003
        self.unsupervised_network = RunModel()

    def disparity_to_BW(self, disparity_img):
        disparity_img_b = disparity_img.numpy()
        disparity_img_bw = disparity_img_b / 30.0
        disparity_img_bw = disparity_img_bw * 255

        return disparity_img_bw

    def callback(self, request, response, data):
        img = request.getImage('image')
        print('Image received')

        disparity_img = self.unsupervised_network.run(img)
        disparity_img_bw = self.disparity_to_BW(disparity_img)

        response.addImage('result', disparity_img_bw)

    def main(self):
        cloudvis = CloudVis(self.port_number)
        cloudvis.run(self.callback)


if __name__ == '__main__':
    monocam = RunMonocularDepthPytorch()
    monocam.main()
