#!/usr/bin/python3

import os
import torch
import matplotlib.pyplot as plt
import numpy
import tarfile
from torchvision import models
import cv2
from pytorch_net import *

class UnsupervisedSingleViewDepth:
    def __init__(self, load_snapshot=None, gpu_id=0, use_gpu_if_available=True):
        self.base_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.pytorch_model = None
        self.plt = plt.figure()

        if load_snapshot is None:
            self.pytorch_model_file = (self.base_directory + "/network/deploy_resnet50by2_pool_pytorch_modelAndWeights.pth")
            self.check_networkfile_unzipped()
        else:
            self.pytorch_model_file = load_snapshot
        self.load_pytorch_model()
        if use_gpu_if_available is True:
            self.device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        self.print_coloured("Setting Device : {}".format(self.device), colour='green')

    def check_networkfile_unzipped(self):
        tar_ext = ".tar.xz"
        network_directory = os.path.dirname(self.pytorch_model_file)
        file_list = os.listdir(network_directory)

        # Check if unzip ".pth" file in the directory
        if not os.path.basename(self.pytorch_model_file) in file_list:
            # If not, check if zip version exists
            self.print_coloured("PyTorch network file {} NOT found in folder {}. Checking for tarball.".format(self.pytorch_model_file, network_directory), colour='cyan')
            if os.path.basename(self.pytorch_model_file).split(".")[0] + tar_ext in file_list:
                self.print_coloured("PyTorch tarball {} found. Extracting...".format(os.path.basename(self.pytorch_model_file).split(".")[0] + tar_ext), colour='green')

                network_tar = tarfile.open(self.pytorch_model_file.split(".")[0] + tar_ext)
                network_tar.extractall(os.path.dirname(self.pytorch_model_file))
                self.print_coloured("Done.", colour='green')

                if os.path.basename(self.pytorch_model_file) in os.listdir(network_directory):
                    self.print_coloured("PyTorch network extracted.", colour='green')
                else:
                    raise ValueError("Something went wrong. PyTorch file not found. Please check.")
            else:
                raise ValueError("No PyTorch model file found. Please confirm the correct network file is being used.")

    def load_pytorch_model(self):
        try:
            self.pytorch_model = torch.load(self.pytorch_model_file)
            self.print_coloured("PyTorch model loaded : {}.".format(self.pytorch_model_file), colour='green')
        except Exception as e:
            print("ERROR! Couldn't load model file. {}".format(e))

    def get_processed_image(self, image):
        # Resize to be 160 by 608 pixels.
        image = cv2.resize(image, (608, 160)) # changing column and row from (160,608) to (608,160) for it to work with cv2
        image_BGR = image[...,
                    ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
        image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
        image_BGR.astype(float)
        mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                               (image_BGR.shape[0], image_BGR.shape[1], 1))
        image_BGR = image_BGR - mean_data
        image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
        image_BGR = numpy.expand_dims(image_BGR, axis=0)
        return image_BGR

    def predict(self, image=None, image_path=None, output_file=None, plot_prediction=False):
        if image is None and image_path is None:
            raise ValueError("Predict must have an input cv2 image or a path to a image.")
        if image is not None and image_path is not None:
            raise ValueError("Predict must have only one input - either cv2 image or a path to a image.")
        if image is None and image_path is not None:
            image = cv2.imread(image_path)

        if self.pytorch_model:
            self.pytorch_model.to(self.device)
            self.pytorch_model.eval()

            processed_image = self.get_processed_image(image)
            processed_image = torch.from_numpy(processed_image).float()
            processed_image = processed_image.to(self.device)

            output= self.pytorch_model.forward(processed_image)
            output_image = output['h_flow'].cpu().detach()
            output_transposed = numpy.transpose(output_image, (2, 3, 0, 1))
            prediction = numpy.squeeze(output_transposed)

            # TODO : REMOVE THIS WARNING (MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
            #   plt.subplot(subplot_number)
            if plot_prediction:
                self.print_images(subplot_number=211, image_to_use=image, title="Input image")
            self.print_images(subplot_number=212, image_to_use=prediction, title="Estimated Depth", show_plot=plot_prediction)

            if output_file:
                cv2.imwrite(output_file, prediction.numpy())

            return prediction

        else:
            raise ValueError("Are you sure the PyTorch model is loaded? Model (self.pytorch_model) not found.")

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
