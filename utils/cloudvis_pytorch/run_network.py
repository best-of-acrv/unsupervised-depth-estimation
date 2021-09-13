#!/usr/bin/python3

import os
import torch
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import numpy
from pytorch_net import *

## Run Model for CloudVis
# As class is initiated, it loads the pytorch model defined in the model file
# When a new image is given, call the .run(img) function
# It will call the forward function of the pre-loaded network and return the result.
# For debugging, you can call the show_pytorch_output - it will run the last given image and it's result


class RunModel:
    def __init__(self):
        self.base_directory = os.path.abspath(os.path.join(os.path.join(os.getcwd(), os.pardir), os.pardir))
        self.pytorch_model_file = self.base_directory + "/unsupervised_single_view_depth" +"/network/deploy_resnet50by2_pool_pytorch.pth"
        self.prediction_disp = None
        self.processed_image = None
        self.original_image = None
        self.plt = plt.figure()
        self.pytorch_model = self.load_pytorch_model()
        self.pytorch_model_output = None
        self.pytorch_net_model = None

    def run(self, img):
        self.get_processed_image_to_use(img)
        self.run_pytorch_model()
        # self.show_pytorch_output()
        return self.prediction_disp

    def get_processed_image_to_use(self, read_image):
        self.print_coloured("Preprocessing images", colour='green')
        self.original_image = read_image
        # Resize to be 160 by 608 pixels.
        image = imresize(self.original_image, (160, 608))
        image_BGR = image[..., ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
        image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
        image_BGR.astype(float)
        # mean_data = numpy.mean(image_BGR)
        mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                               (image_BGR.shape[0], image_BGR.shape[1], 1))
        image_BGR = image_BGR - mean_data
        image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
        image_BGR = numpy.expand_dims(image_BGR, axis=0)

        self.processed_image = image_BGR

    def load_pytorch_model(self):
        self.pytorch_model = torch.load(self.pytorch_model_file)
        self.print_coloured("PyTorch model loaded.", colour='green')
        return self.pytorch_model

    def run_pytorch_model(self):
        if self.pytorch_model:
            self.pytorch_model.eval()
            image = torch.from_numpy(self.processed_image).float()
            self.pytorch_model_output = self.pytorch_model.forward(image)
            self.pytorch_net_model = self.pytorch_model.models

            output = self.pytorch_model_output['h_flow'].detach()
            output_transposed = numpy.transpose(output, (2, 3, 0, 1))
            self.prediction_disp = numpy.squeeze(output_transposed)
        else:
            raise ValueError("Are you sure the PyTorch model is loaded? Model (self.pytorch_model) not found.")

    def show_pytorch_output(self):
        # if not self.prediction_disp:
        self.print_images(subplot_number=211, image_to_use=self.original_image, title="Input image")
        self.print_images(subplot_number=212, image_to_use=self.prediction_disp, title="Output image: Pytorch", show_plot=True)
        # else:
        #     raise ValueError("Are you sure you ran the inference? self.pytorch_model_output not found.")

    def print_images(self, subplot_number, image_to_use, title, show_plot=False):
        plt.subplot(subplot_number)
        plt.imshow(image_to_use, cmap='plasma')
        plt.title(title)
        if show_plot:
            plt.show()

    def print_coloured(self, text, colour='red'):
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        END = '\033[0m'

        if colour == 'red':
            print("{} {} {} {}".format(RED, BOLD, text, END))
        if colour == 'blue':
            print("{} {} {} {}".format(BLUE, BOLD, text, END))
        if colour == 'purple':
            print("{} {} {} {}".format(PURPLE, BOLD, text, END))
        if colour == 'cyan':
            print("{} {} {} {}".format(CYAN, BOLD, text, END))
        if colour == 'green':
            print("{} {} {} {}".format(GREEN, BOLD, text, END))
        if colour == 'yellow':
            print("{} {} {} {}".format(YELLOW, BOLD, text, END))

