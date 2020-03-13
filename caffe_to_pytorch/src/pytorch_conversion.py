#!/usr/bin/python3

import torch

caffenet_path = "/home/garima/code/bestOfACRV/pytorch_caffe"
import sys
sys.path.append(caffenet_path)
from caffenet import *

CAFFE_ROOT = '/usr/bin'
import os

os.chdir(CAFFE_ROOT)
import numpy
import numpy.matlib
import caffe
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import csv

from pytorch_net import *


class TestPytorchConversion():
    def __init__(self):
        self.base_directory = "/home/garima/code/bestOfACRV/Unsupervised_Depth_Estimation"
        self.write_base_directory = "/home/garima/code/bestOfACRV/other"
        self.caffe_net_model = self.base_directory + "/scripts/deploy_resnet50by2_pool.prototxt"
        self.caffe_net_weights = self.base_directory + "/model/train_iter_40000.caffemodel"
        self.phase = 'test'
        self.image_folder = self.base_directory + "/images"
        self.prediction_disp = {}
        self.prediction_disp_caffe = {}
        self.read_images = []
        self.processed_images = []
        self.plt = plt.figure()
        self.pytorch_caffe_net = None
        self.caffe_net = None
        self.output_pytorch = None
        self.output_caffe = None
        self.net_model = None
        self.saved_pytorch_model = None
        self.saved_pytorch_model_output = None
        self.saved_pytorch_net_model = None
        self.debug_mode = False                          # Just added printing
        self.save_model = False

    def run(self):
        self.get_processed_image_to_use()
        self.load_caffe_network()
        self.check_caffe_output()
        self.load_caffe_model_into_pytorch()
        if self.save_model:
            self.save_pytorch_model(save_full=True)
        self.test_saved_model()
        self.run_saved_pytorch_model()
        self.check_pytorch_output()
        self.check_saved_pytorch_model()
        plt.show()
        self.check_output()

    def get_processed_image_to_use(self):
        self.print_coloured("Load single image for now", colour='green')
        # There are 4 images in the directory 
        # for i in range(1, 5):
        for i in range(1, 2):
            # Images are called 1,2,3,4.png. Write bad code to read these files using absolute path
            read_file = "{}/{}.png".format(self.image_folder, i)
            self.print_coloured("Reading image file {}".format(read_file), colour='green')
            # Read each image at a time - shape: (188, 620, 3)
            image = imread(read_file)
            # Resize to be 160 by 608 pixels.
            image = imresize(image, (160, 608))
            image_BGR = image[...,
                        ::-1]  # In this case, the ellipsis ... is equivalent to :,: while ::-1 inverts the order of the last dimension (channels). The data has 3 dimensions: width, height, and color. ::-1 effectively reverses the order of the colors. The width and height are not affected. img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
            image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
            image_BGR.astype(float)
            # mean_data = numpy.mean(image_BGR)

            mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                                   (image_BGR.shape[0], image_BGR.shape[1], 1))
            image_BGR = image_BGR - mean_data

            self.print_images(subplot_number=411, image_to_use=image, title="Input image")
            # self.print_images(subplot_number=412, image_to_use=image_BGR, title="Processed image")

            image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
            image_BGR = numpy.expand_dims(image_BGR, axis=0)  # np.transpose(im[:, :, :, np.newaxis], (3, 2, 0, 1)) # ()

            self.read_images.append(image)
            self.processed_images.append(image_BGR)
            if self.debug_mode:
                self.print_coloured("Image lists {} {}".format(len(self.read_images), len(self.processed_images)), colour='blue')

    def load_caffe_network(self, print_layers=False):
        self.print_coloured('###########################################################################################', colour='blue')
        self.caffe_net = caffe.Net(self.caffe_net_model, self.caffe_net_weights, caffe.TEST)

        if self.caffe_net:
            if print_layers:
                for layer_name, blob in self.caffe_net.blobs.items():
                    self.print_coloured("{} : {} ".format(layer_name, blob.data.shape), colour='blue')
            for i in self.processed_images:
                if self.debug_mode:
                    self.print_coloured(i.shape, colour='blue')
                self.caffe_net.blobs['imL'].data[...] = i
                self.output_caffe = self.caffe_net.forward()
        else:
            raise ValueError("self.caffe_net not found")

    def check_caffe_output(self):
        if self.caffe_net:
            self.write_to_file(filename='caffe_output.csv', data=self.caffe_net.blobs)
        else:
            raise ValueError("self.caffe_net not found")

        if self.output_caffe:
            print_data = self.output_caffe['h_flow'].data
            # disp = print_data[0][0] # (H, W)
            output_transposed = numpy.transpose(print_data, (2, 3, 1, 0))
            self.prediction_disp_caffe[1] = numpy.squeeze(output_transposed)
            self.print_images(subplot_number=412, image_to_use=self.prediction_disp_caffe[1], title="Output image Caffe")
        else:
            raise ValueError("self.output_caffe not found")

    def load_caffe_model_into_pytorch(self):
        # Read caffe net into pytorch using the plugin - Using dummy class PyTorch to ensure serialization is not dependent on having the tool.
        # self.pytorch_caffe_net = CaffeNet(self.caffe_net_model)
        self.pytorch_caffe_net = PytorchNet(self.caffe_net_model)
        self.pytorch_caffe_net.load_weights(self.caffe_net_weights)
        self.pytorch_caffe_net.eval()
        if self.debug_mode:
            # self.write_to_file(filename='pytorch_net.csv', data=self.pytorch_caffe_net)
            self.print_coloured(self.pytorch_caffe_net)
            sys.exit("see whats here - before there is too much printing..")

        if self.pytorch_caffe_net:
            for i in self.processed_images:
                image = torch.from_numpy(i).float()
                self.output_pytorch = self.pytorch_caffe_net.forward(image)
                self.net_model = self.pytorch_caffe_net.models

                if self.debug_mode:
                    self.print_coloured(image.type(), colour='blue')
                    self.print_coloured(self.output_pytorch['h_flow'].shape, colour='blue')
                    self.write_to_file(filename='pytorch_tensors.csv', data=self.output_pytorch)
        else:
            raise ValueError("PyTorch model not found")

    def check_pytorch_output(self):
        if self.output_pytorch:
            output = self.output_pytorch['h_flow'].detach()
            if self.debug_mode:
                # print(self.output)
                self.print_coloured("Caffe: {}".format(self.output_caffe['h_flow']), colour='blue')
                self.print_coloured("Pytorch: {}".format(output), colour='blue')

            # disp = output[0][0] # (H, W)

            output_transposed = numpy.transpose(output, (2, 3, 0, 1))
            self.prediction_disp[1] = numpy.squeeze(output_transposed)

            self.print_images(subplot_number=413, image_to_use=self.prediction_disp[1], title="Output image Pytorch")

        # # # Processing  disparity to depth : image.shape (160, 608, 3), prediction_disp[1].shape (1, 1, 160, 608)
        # scale = image.shape[1]/prediction_disp[i].shape[1]
        # flowHr = scale*imresize(prediction_disp[i], (370,1224))
        # #  estimate depth
        # predicted_depths = 389.6304/flowHr

        # # load eigens mask and apply it for evaluation
        # # L = logical(A) converts A into an array of logical values. Any nonzero element of A is converted to logical 1 (true) and zeros are converted to logical 0 (false). Complex values and NaNs cannot be converted to logical values and result in a conversion error.
        # # >>> x = np.array([1,3,-1, 5, 7, -1])
        # # mask = logical(imread(mask_file));
        # #  mask = (x > 0)
        # # >>> out.astype(np.bool)
        # # array([ True,  True, False,  True,  True, False])
        # # >>> out.astype(np.int)
        # # array([1, 1, 0, 1, 1, 0])

        # mask_file = imread(base_directory + "/images/mask_eigen.png")
        # mask = (mask_file != 0).astype(numpy.int)
        # # p = predicted_depths[i]
        # # p(~mask(:)) = np.nan
        # # predicted_depths[i] = p

        # plt.subplot(413)
        # plt.imshow(self.prediction_disp[1])

        # plt.imshow(self.prediction_disp[1], cmap='plasma')
        # plt.title("Output/Disparity Images")
        # plt.subplot(414)
        # plt.imshow(disp, cmap='plasma')
        # plt.title("Predicted Depth Images")
        # print("Image {} processed \n -----------------------------------------------------------".format(i))

        else:
            raise ValueError("self.output_pytorch not found.")

    def save_pytorch_model(self, save_full=True):
        if save_full:
            # filename = self.write_base_directory + "/depth_net_pytorch_full.pt"
            filename = self.write_base_directory + "/deploy_resnet50by2_pool_pytorch_modelAndWeights.pth"
            torch.save(self.pytorch_caffe_net, filename) # Right
        else:
            filename = self.write_base_directory + "/deploy_resnet50by2_pool_pytorch_weights.pth"
            torch.save(self.pytorch_caffe_net.state_dict(), filename)

    def test_saved_model(self):
        # pytorch_file = "/home/garima/code/bestOfACRV/other/deploy_resnet50by2_pool_fullModelandWeights.pth"
        pytorch_file = "/home/garima/code/bestOfACRV/other/deploy_resnet50by2_pool_pytorch_modelAndWeights.pth"
        self.saved_pytorch_model = torch.load(pytorch_file)
        self.print_coloured("PyTorch model loaded.", colour='green')

    def run_saved_pytorch_model(self):
        if self.saved_pytorch_model:
            for i in self.processed_images:
                image = torch.from_numpy(i).float()
                self.saved_pytorch_model.eval()
                self.saved_pytorch_model_output = self.saved_pytorch_model.forward(image)
                self.saved_pytorch_net_model = self.saved_pytorch_model.models
            self.write_to_file(filename='saved_pytorch_tensors.csv', data=self.saved_pytorch_model_output)
        else:
            raise ValueError("Are you sure the saved PyTorch model is loaded? Model not found.")

    def check_saved_pytorch_model(self):
        if self.saved_pytorch_model_output:
            output = self.saved_pytorch_model_output['h_flow'].detach()
            if self.debug_mode:
                self.print_coloured("Saved pytorch: {}".format(output), colour='blue')
            output_transposed = numpy.transpose(output, (2, 3, 0, 1))
            self.prediction_disp[1] = numpy.squeeze(output_transposed)
            self.print_images(subplot_number=414, image_to_use=self.prediction_disp[1], title="Output image Saved Pytorch")

    def check_output(self):
        if self.output_caffe and self.output_pytorch and self.saved_pytorch_model_output:
            # Check the difference in final output
            # output shape is ([1, 1, 160, 608])
            output_pytorch = self.output_pytorch['h_flow'].detach()
            output_caffe = self.output_caffe['h_flow']
            output_saved_pytorch = self.saved_pytorch_model_output['h_flow'].detach()

            # output shape should now be (160, 608) and both should now be numpy arrays
            output_pytorch_processed = numpy.squeeze(numpy.transpose(output_pytorch, (2, 3, 0, 1))).numpy()
            output_caffe_processed = numpy.squeeze(numpy.transpose(output_caffe, (2, 3, 0, 1)))
            output_saved_pytorch_processed = numpy.squeeze(numpy.transpose(output_saved_pytorch, (2, 3, 0, 1))).numpy()

            # diff = output_caffe_processed - output_pytorch_processed
            diff = numpy.subtract(output_caffe_processed, output_pytorch_processed)
            max_val = numpy.amax(diff)
            max_val_pytorches = numpy.amax(numpy.subtract(output_pytorch_processed, output_saved_pytorch_processed))

            if self.debug_mode:
                self.print_coloured("Max difference between pytorch and caffe is {} and tensors are {}".format(max_val, diff), colour='purple')
                self.print_coloured("Max difference between pytorch loaded and saved pytorch model is {} and tensors are {}".format(max_val_pytorches,
                numpy.subtract(output_pytorch_processed, output_saved_pytorch_processed)), colour='purple')
                self.write_tensor_to_file(filename='compare_output.csv',
                                          label='Max difference between Caffe and PyTorch output is {} and between the two pytorches is: '.format(
                                          max_val, max_val_pytorches), data=diff)

            numpy.testing.assert_almost_equal(output_caffe_processed, output_pytorch_processed, decimal=4)          # Equal at 4 decimal but not 5.
            numpy.testing.assert_almost_equal(output_saved_pytorch_processed, output_pytorch_processed, decimal=4)  # Equal at 4 decimal but not 5.

    def write_to_file(self, filename, data):
        file_to_write = self.write_base_directory + "/" + filename
        with open(file_to_write, 'w') as csvfile:
            fieldname = data.keys()
            writer = csv.writer(csvfile)
            for layer_name, data in data.items():
                writer.writerow([layer_name, data.data])
        csvfile.close()

    def write_tensor_to_file(self, filename, label, data):
        file_to_write = self.write_base_directory + "/" + filename
        numpy.savetxt(file_to_write, X=data, header=label, delimiter=',')

    def print_images(self, subplot_number, image_to_use, title, show_plot=False):
        plt.subplot(subplot_number)
        plt.imshow(image_to_use)
        plt.title(title)
        plt.subplot(subplot_number)
        plt.imshow(image_to_use)
        plt.title(title)

        if show_plot:
            plt.show()

    def print_coloured(self, text, colour='red'):
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
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


# class PytorchNet(CaffeNet):
#     def __init__(self, protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN'):
#         super(PytorchNet, self).__init__(protofile, width=None, height=None, channels=None, omit_data_layer=False, phase='TRAIN')


if __name__ == "__main__":
    pytorch_conversion = TestPytorchConversion()
    pytorch_conversion.run()
