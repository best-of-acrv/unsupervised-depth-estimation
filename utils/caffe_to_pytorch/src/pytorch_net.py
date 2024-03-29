#!/usr/bin/python3

import sys
sys.path.append('../convertor_tool/pytorch_caffe')

from caffenet import *

# convertor_tool.pytorch_caffe.caffenet import *

class PytorchNet(CaffeNet):
    def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN'):
        super(PytorchNet, self).__init__(protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN')
