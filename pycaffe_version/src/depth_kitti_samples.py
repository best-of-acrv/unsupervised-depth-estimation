#!/usr/bin/python3

import os
import numpy.matlib
import caffe
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt

# Converting Ravi's MatCaffe code to PyCaffe

USE_GPU = False

if USE_GPU:
    caffe.set_device(0)  # Or the index of the GPU you want to use
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

print("Initialized caffe")

base_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
net_model = base_directory + "/network/deploy_resnet50by2_pool.prototxt"
net_weights = base_directory + "/network/train_iter_40000.caffemodel"
image_folder = base_directory + "/sample_images"
phase = 'test'
prediction_disp = {}

if not net_model:
    print("Couldn't find the model. Ensure you have downloaded the model before running this script.")

net = caffe.Net(net_model, net_weights, caffe.TEST)

# There are 4 images in the directory - read all of them for demo.
for i in range(1, 5):
    # Images are called 1,2,3,4.png.
    read_file = "{}/{}.png".format(image_folder, i)
    print("Reading image file {}".format(read_file))
    # Read each image at a time - shape: (188, 620, 3)
    image = io.imread(read_file)
    # Resize to be h=160 by w=608 pixels.
    image = numpy.array(Image.fromarray(image).resize(size=(608, 160)))

    # In this case, the ellipsis ... is equivalent to :,: while ::-1 inverts the order of the last
    # dimension (channels). The data has 3 dimensions: width, height, and color. ::-1 effectively reverses the order
    # of the colors. The width and height are not affected. img = img[:, :, : :-1] is equivalent to img = img[:, :,
    # [2,1,0]].
    image_BGR = image[..., ::-1]
    image_BGR = numpy.transpose(image_BGR, (1, 0, 2))
    image_BGR.astype(float)
    # mean_data = numpy.mean(image_BGR)
    mean_data = numpy.tile(numpy.reshape([104, 117, 123], (1, 1, 3), order='F'),
                           (image_BGR.shape[0], image_BGR.shape[1], 1))
    image_BGR = image_BGR - mean_data

    plt.figure()
    plt.subplot(211)
    plt.imshow(image)
    plt.title("Input image")

    image_BGR = numpy.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)
    net.blobs['imL'].data[...] = numpy.expand_dims(image_BGR, axis=0)

    scores = net.forward()

    output = net.blobs['h_flow'].data
    disp = output[0][0]  # (H, W)
    output_transposed = numpy.transpose(output, (2, 3, 1, 0))
    prediction_disp[i] = numpy.squeeze(output_transposed)

    plt.subplot(212)
    plt.imshow(prediction_disp[i], cmap='plasma')
    plt.title("Output/Disparity Images")
    plt.show()
    print("Image {} processed \n -----------------------------------------------------------".format(i))
