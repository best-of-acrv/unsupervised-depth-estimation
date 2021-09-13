#!/usr/bin/python3

import os
import torch
import matplotlib.pyplot as plt
# from scipy.misc import imread, imresize

import numpy
from pytorch_net import *
import tarfile
from torchvision import models
import cv2

class RunSingleViewDepthExample:
    def __init__(self):
        self.base_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.pytorch_model_file = self.base_directory + "/network/deploy_resnet50by2_pool_pytorch_modelAndWeights.pth"
        self.image_folder = self.base_directory + "/sample_images"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_disp = {}
        self.read_images = []
        self.processed_images = []
        self.plt = plt.figure()
        self.pytorch_model = None
        self.pytorch_model_output = {}
        self.pytorch_net_model = {}

    def run(self):
        self.check_networkfile_unzipped()
        self.get_processed_image_to_use()
        self.load_pytorch_model()
        self.run_pytorch_model()
        self.show_pytorch_output()

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

    def get_processed_image_to_use(self):
        self.print_coloured("Loading all images ending in .png or .jpg from the given folder {}".format(self.image_folder), colour='green')
        # There are 4 images in the directory called 1,2,3,4.png. Read all files ending with ".png" or ".jpg" in the folder.
        for file in os.listdir(self.image_folder):
            if file.endswith('png') or file.endswith('jpg'):
                read_file = "{}/{}".format(self.image_folder, file)
                self.print_coloured("Reading image file {}".format(read_file), colour='green')
                # Read each image at a time - shape: (188, 620, 3)
                image = cv2.imread(read_file)
                # Resize to be 160 by 608 pixels.
                # image = imresize(image, (160, 608))
                image = cv2.resize(image, (608,160)) # changing column and row to work with cv2
                image_BGR = image[..., ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
                image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
                image_BGR.astype(float)
                # mean_data = numpy.mean(image_BGR)
                mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                                       (image_BGR.shape[0], image_BGR.shape[1], 1))
                image_BGR = image_BGR - mean_data
                image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
                image_BGR = numpy.expand_dims(image_BGR, axis=0)

                self.read_images.append(image)
                self.processed_images.append(image_BGR)

    def load_pytorch_model(self):
        self.pytorch_model = torch.load(self.pytorch_model_file)
        self.print_coloured("PyTorch model loaded.", colour='green')
        if torch.cuda.is_available():
            curr_dev = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(curr_dev)
        else:
            curr_dev = "CPU"
            dev_name = ""
        self.print_coloured("PyTorch using {} {}.".format(curr_dev, dev_name), colour='green')

    def get_processed_image(self, image):
        # Resize to be 160 by 608 pixels.
        # image = imresize(image, (160, 608)) # old scipy
        image = cv2.resize(image, (608, 160)) # changing column and row to work with cv2

        image_BGR = image[...,
                    ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
        image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
        image_BGR.astype(float)
        # mean_data = numpy.mean(image_BGR)
        mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                               (image_BGR.shape[0], image_BGR.shape[1], 1))
        image_BGR = image_BGR - mean_data
        image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
        image_BGR = numpy.expand_dims(image_BGR, axis=0)
        return image_BGR

    def predict(self, image, output_file=None):
        self.pytorch_model = torch.load(self.pytorch_model_file)
        # self.pytorch_model.to(self.device)
        self.pytorch_model.eval()
        processed_image = self.get_processed_image(image)
        processed_image = torch.from_numpy(processed_image).float()
        output= self.pytorch_model.forward(processed_image)
        output_image = output['h_flow'].detach()
        output_transposed = numpy.transpose(output_image, (2, 3, 0, 1))
        prediction = numpy.squeeze(output_transposed)
        # TODO : FINISH SAVING TO FILE FEATURE
        if output_file:
            cv2.imwrite(output_file, prediction)
        return prediction

    def run_pytorch_model(self):
        if self.pytorch_model:
            self.pytorch_model.eval()
            for index, img in enumerate(self.processed_images):
                image = torch.from_numpy(img).float()
                # online_resnet50 = models.resnet50(pretrained=True)
                # print(online_resnet50)
                self.pytorch_model_output.update({index: self.pytorch_model.forward(image)})
                self.pytorch_net_model.update({index: self.pytorch_model.models})
        else:
            raise ValueError("Are you sure the PyTorch model is loaded? Model (self.pytorch_model) not found.")

    def show_pytorch_output(self):
        if self.pytorch_model_output:
            for keys, network_output in self.pytorch_model_output.items():
                output = network_output['h_flow'].detach()
                output_transposed = numpy.transpose(output, (2, 3, 0, 1))
                self.prediction_disp[keys] = numpy.squeeze(output_transposed)
                self.print_images(subplot_number=211, image_to_use=self.read_images[keys], title="Input image")
                self.print_images(subplot_number=212, image_to_use=self.prediction_disp[keys], title="Output image: Pytorch", show_plot=True)

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


if __name__ == '__main__':
    model = RunSingleViewDepthExample()
    model.run()
