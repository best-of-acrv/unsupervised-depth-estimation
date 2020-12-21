#!/usr/bin/python3

import torch
import os
import sys
import numpy
import numpy.matlib
import caffe
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import csv
from pytorch_net import *
import copy


class TestPytorchConversion:
    def __init__(self):
        self.base_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.write_base_directory = self.base_directory + "/outputs"
        self.parent_directory = os.path.abspath(os.path.join(self.base_directory, os.pardir))
        self.caffe_net_model = self.parent_directory + "/pycaffe_version/network/deploy_resnet50by2_pool.prototxt"
        self.caffe_net_weights = self.parent_directory + "/pycaffe_version/network/train_iter_40000.caffemodel"
        self.save_pytorch_filename = self.write_base_directory + "/deploy_resnet50by2_pool_pytorch"
        self.pytorch_presaved_model = self.parent_directory + "/pytorch_version/network/deploy_resnet50by2_pool_pytorch_modelAndWeights.pth"
        self.phase = 'test'
        self.image_folder = self.base_directory + "/sample_images"
        self.prediction_disp = {}
        self.prediction_disp_caffe = {}
        self.prediction_disp_saved_pytorch = {}
        self.read_images = []
        self.processed_images = []
        self.plt = plt.figure()
        self.pytorch_caffe_net = None
        self.caffe_net = None
        self.output_pytorch = {}
        self.output_caffe = {}
        self.net_model = {}
        self.saved_pytorch_model = None
        self.saved_pytorch_model_output = {}
        self.saved_pytorch_net_model = {}
        self.debug_mode = False                          # Just additional printing
        self.save_model = True

    def run(self):
        self.get_processed_image_to_use()
        self.load_caffe_network()
        self.check_caffe_output()
        self.load_caffe_model_into_pytorch()
        if self.save_model:
            self.save_pytorch_model(save_full=True)    # Default: save architecture & weights, instead of just weights.
        self.test_saved_model()
        self.run_saved_pytorch_model()
        self.check_pytorch_output()
        self.check_saved_pytorch_model()
        self.check_output()
        self.show_output_images()

    def get_processed_image_to_use(self):
        self.print_coloured(
            "Loading all images ending in .png or .jpg from the given folder {}".format(self.image_folder),
            colour='green')
        # There are 4 images in the directory called 1,2,3,4.png. Read all files ending with ".png" or ".jpg" in the
        # folder.
        for file in os.listdir(self.image_folder):
            if file.endswith('png') or file.endswith('jpg'):
                read_file = "{}/{}".format(self.image_folder, file)
                self.print_coloured("Reading image file {}".format(read_file), colour='green')
                # Read each image at a time - shape: (188, 620, 3)
                image = imread(read_file)
                # Resize to be 160 by 608 pixels.
                image = imresize(image, (160, 608))
                image_BGR = image[..., ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:,
                # :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
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
                if self.debug_mode:
                    self.print_coloured("Image lists {} {}".format(len(self.read_images), len(self.processed_images)), colour='blue')

    def load_caffe_network(self, print_layers=False):
        self.print_coloured('###########################################################################################', colour='blue')
        self.caffe_net = caffe.Net(self.caffe_net_model, self.caffe_net_weights, caffe.TEST)

        if self.caffe_net:
            self.print_coloured("Caffe model loaded.", colour='green')
            if print_layers:
                for layer_name, blob in self.caffe_net.blobs.items():
                    self.print_coloured("{} : {} ".format(layer_name, blob.data.shape), colour='blue')

            for index, img in enumerate(self.processed_images):
                if self.debug_mode:
                    self.print_coloured(img.shape, colour='blue')
                self.caffe_net.blobs['imL'].data[...] = img
                output = copy.deepcopy(self.caffe_net.forward())
                self.output_caffe.update({index: output})
        else:
            raise ValueError("self.caffe_net not found.")

    def check_caffe_output(self):
        if self.debug_mode:
            if self.caffe_net:
                self.write_to_file(filename='caffe_output.csv', data=self.caffe_net.blobs)
            else:
                raise ValueError("self.caffe_net not found")

        if self.output_caffe:
            for keys, network_outputs in self.output_caffe.items():
                print_data = network_outputs['h_flow'].data
                output_transposed = numpy.transpose(print_data, (2, 3, 1, 0))
                self.prediction_disp_caffe[keys] = numpy.squeeze(output_transposed)
        else:
            raise ValueError("self.output_caffe not found")

    def load_caffe_model_into_pytorch(self):
        # Read caffe net into pytorch using the plugin - Using dummy class PyTorch to ensure serialization is not
        # dependent on having the tool.
        self.print_coloured('###########################################################################################', colour='blue')
        self.print_coloured("Initiating conversion of Caffe model to PyTorch model.", colour='green')
        self.pytorch_caffe_net = PytorchNet(self.caffe_net_model)
        self.pytorch_caffe_net.load_weights(self.caffe_net_weights)
        self.pytorch_caffe_net.eval()
        if self.debug_mode:
            self.print_coloured(self.pytorch_caffe_net)
            sys.exit("see whats here - before there is too much printing..")

        if self.pytorch_caffe_net:
            self.print_coloured("PyTorch model converted.", colour='green')
            for index, img in enumerate(self.processed_images):
                image = torch.from_numpy(img).float()
                output = self.pytorch_caffe_net.forward(image)
                self.output_pytorch.update({index: output})
                self.net_model.update({index: self.pytorch_caffe_net.models})

                if self.debug_mode:
                    self.print_coloured(image.type(), colour='blue')
                    self.print_coloured(output['h_flow'].shape, colour='blue')
                    self.write_to_file(filename='pytorch_tensors.csv', data=output)
        else:
            raise ValueError("PyTorch model not found")

    def check_pytorch_output(self):
        if self.output_pytorch:
            for keys, network_output in self.output_pytorch.items():
                output = network_output['h_flow'].detach()
                if self.debug_mode:
                    self.print_coloured("Caffe: {}".format(self.output_caffe[keys]['h_flow']), colour='blue')
                    self.print_coloured("Pytorch: {}".format(output), colour='blue')

                output_transposed = numpy.transpose(output, (2, 3, 0, 1))
                self.prediction_disp[keys] = numpy.squeeze(output_transposed)
        else:
            raise ValueError("self.output_pytorch not found.")

    def save_pytorch_model(self, save_full=True):
        if save_full:
            self.save_pytorch_filename = self.save_pytorch_filename + ".pth"
            torch.save(self.pytorch_caffe_net, self.save_pytorch_filename)
        else:
            self.save_pytorch_filename = self.save_pytorch_filename + "_justWeights.pth"
            torch.save(self.pytorch_caffe_net.state_dict(), self.save_pytorch_filename)

    def test_saved_model(self):
        self.print_coloured('###########################################################################################', colour='blue')
        if self.save_model:
            pytorch_file = self.save_pytorch_filename
        else:
            pytorch_file = self.pytorch_presaved_model
        self.saved_pytorch_model = torch.load(pytorch_file)
        self.print_coloured("PyTorch model loaded.", colour='green')

    def run_saved_pytorch_model(self):
        if self.saved_pytorch_model:
            for index, img in enumerate(self.processed_images):
                image = torch.from_numpy(img).float()
                self.saved_pytorch_model.eval()
                output = self.saved_pytorch_model.forward(image)
                self.saved_pytorch_model_output.update({index: output})
                self.saved_pytorch_net_model.update({index: self.saved_pytorch_model.models})
            self.write_to_file(filename='saved_pytorch_tensors.csv', data=output)
        else:
            raise ValueError("Are you sure the saved PyTorch model is loaded? Model not found.")

    def check_saved_pytorch_model(self):
        if self.saved_pytorch_model_output:
            for keys, network_output in self.saved_pytorch_model_output.items():
                output = network_output['h_flow'].detach()
                if self.debug_mode:
                    self.print_coloured("Saved pytorch: {}".format(output), colour='blue')
                output_transposed = numpy.transpose(output, (2, 3, 0, 1))
                self.prediction_disp_saved_pytorch[keys] = numpy.squeeze(output_transposed)

    def check_output(self):
        if self.output_caffe and self.output_pytorch and self.saved_pytorch_model_output:
            # Check the difference in final output
            # output shape is ([1, 1, 160, 608])
            for keys, vals in self.output_pytorch.items():
                output_pytorch = self.output_pytorch[keys]['h_flow'].detach()
                output_caffe = self.output_caffe[keys]['h_flow']
                output_saved_pytorch = self.saved_pytorch_model_output[keys]['h_flow'].detach()

                # output shape should now be (160, 608) and both should now be numpy arrays
                output_pytorch_processed = numpy.squeeze(numpy.transpose(output_pytorch, (2, 3, 0, 1))).numpy()
                output_caffe_processed = numpy.squeeze(numpy.transpose(output_caffe, (2, 3, 0, 1)))
                output_saved_pytorch_processed = numpy.squeeze(numpy.transpose(output_saved_pytorch, (2, 3, 0, 1))).numpy()

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

    def show_output_images(self):
        if self.output_caffe and self.output_pytorch and self.saved_pytorch_model_output:
            for keys, vals in self.output_pytorch.items():
                self.print_images(subplot_number=411, image_to_use=self.read_images[keys], title="Input image")
                self.print_images(subplot_number=412, image_to_use=self.prediction_disp_caffe[keys], title="Output image Caffe")
                self.print_images(subplot_number=413, image_to_use=self.prediction_disp[keys], title="Output image Pytorch")
                self.print_images(subplot_number=414, image_to_use=self.prediction_disp_saved_pytorch[keys], title="Output image Saved Pytorch")
                plt.show()

    def write_to_file(self, filename, data):
        file_to_write = self.write_base_directory + "/" + filename
        with open(file_to_write, 'w') as csvfile:
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


if __name__ == "__main__":
    pytorch_conversion = TestPytorchConversion()
    pytorch_conversion.run()
