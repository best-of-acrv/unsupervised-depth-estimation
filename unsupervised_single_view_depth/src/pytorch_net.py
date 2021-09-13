#!/usr/bin/python3

import sys
import os
base_dir =  os.path.abspath(os.path.join(os.path.join(os.path.abspath(__file__), os.pardir), "../../utils/caffe_to_pytorch/convertor_tool/pytorch_caffe"))
print(base_dir)
# sys.path.append('../../utils/caffe_to_pytorch/convertor_tool/pytorch_caffe')
sys.path.append(base_dir)

from caffenet import *


class PytorchNet(CaffeNet):
    def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN'):
        super(PytorchNet, self).__init__(protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN')
