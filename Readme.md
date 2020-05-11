# Monocam Unsupervised Depth Estimation 

This repository wraps Ravi Garg's Unsupervised CNN for Single View Depth Estimation work. The original work was developed, trained and deployed in Caffe 1 and can be found here: https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation. 

We have converted the Caffe network and weights to a PyTorch compatible network (architecture and weights). 

This repository provides 1) a pycaffe implementation of the original network (to allow for non Matlab inference), 2) a pytorch implementation to run inferences (along with a the converted architecture and weights from the original implementation) and 3) the tool used for this caffe to pytorch conversion of the network architecture and weights.  

## Repository Structure
This repository is split into three sections, one for each implementation.

### PyCaffe Version

PyCaffe version consists of pycaffe implementation of the original matcaffe sample inference script along with the trained caffe network. It also contains some sample images. Finally, it has a conda environment definition file that can be used to quickly create the virtual environment required to use it.

### PyTorch Version

PyTorch version includes the converted Caffe to PyTorch network (architecture and the weights) along with a sample script to run the inference and some sample images. It also has a seperate conda environment file that can be used to create an virtual environment to run the pytorch version.

### Caffe to PyTorch Version

This is essentially the code we used to convert and validate Ravi's network from Caffe to PyTorch along with the sample images. We used a modified version of an external tool called [pytorch-caffe](https://github.com/marvis/pytorch-caffe) by marvis. The modified tool is also part of this sub-folder. 

*Note: We are assuming you are using a Linux:Ubuntu system.*
