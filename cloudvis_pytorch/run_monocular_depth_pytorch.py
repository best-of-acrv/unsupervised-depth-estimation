import cv2
from cloudvis import CloudVis
from RunPytorchModel import *

PORT = 6003


def callback(request, response, data):
    img = request.getImage('image')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    response.addImage('result', gray)


if __name__ == '__main__':
    cloudvis = CloudVis(PORT)
    cloudvis.run(callback)
