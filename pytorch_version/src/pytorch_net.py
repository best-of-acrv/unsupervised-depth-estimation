#!/usr/bin/python3

caffenet_path = "/home/garima/code/bestOfACRV/pytorch_caffe"
import sys
sys.path.append(caffenet_path)
from caffenet import *


class PytorchNet(CaffeNet):
    def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN'):
        super(PytorchNet, self).__init__(protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN')