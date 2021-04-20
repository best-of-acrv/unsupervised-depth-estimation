<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# Unsupervised CNN for Single View Depth Estimation 

[![Best of ACRV Repository](https://img.shields.io/badge/collection-best--of--acrv-%23a31b2a)](https://roboticvision.org/best-of-acrv)

Unsupervised CNN for Single View Depth Estimation, is a nueral networks that can predict depth from a single RGB image. It achieves this by training the network analogous to an autoencoder with a pair of images, source and target, with a small known camera motion between the two. 

TODO: MAKE THE SENTENCE BELOW EASIER TO DIGEST BY NON EXPERTS.

This can be achieved because the convolution encoder is trained to predict the depth map of the source image. To do this, inverse warp of the target image is generated using the predicted depth and known inter-view displacement, to reconstruct the source image; the photometric error in the reconstruction is the reconstruction loss for the encoder. 

[The paper](https://arxiv.org/pdf/1603.04992v2.pdf) further details the unsupervised deep learning framework developed to predict scene depth from a single image, that does not require a pre-training stage or annotated depth ground-truth.

TODO : MAKE IMAGE BETTER!

<p align="center">
<img alt="Single View Sample output on KITTI dataset" src="https://github.com/best-of-acrv/unsupervised-depth-estimation/raw/repo_restructure/docs/sample_output_1.png"/>
</p>

This repository contains an PyTorch (and PyCaffe) open-source implementation of Unsupervised CNN for Single View Depth Estimation with official weights converted from caffe. This package currently provides inference implementation that can be deployed. We are working on providing training and evaluation implementation in PyTorch as well. Dependencies of both the PyTorch and PyCaffe packages can be easily installed using `conda`, and if you prefer a more manual approach, via `pip`. The PyTorch version of the package can also be installed from `pip`.

Our code is free to use, and licensed under BSD-3. We simply ask that you [cite our and Ravi's work](#citing-our-work) if you use Single View Depth Estimation in your own research.

## Related resources

This repository updates Ravi Garg's open-source unsupervised CNN for Single View Depth Estimation work by providing the the network's implementation in PyTorch and PyCaffe. 

- Original paper : [Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://arxiv.org/pdf/1603.04992v2.pdf)
- The original Caffe 1 implementaion: [Realtime Unsupervised Depth Estimation from an Image](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation). 

## Repository Structure
This repository is split into three sections:

1) a pytorch implementation to run inferences (along with a the converted architecture and weights from the original implementation) and 
2) a pycaffe implementation of the original caffe network (to allow for non Matlab inference),
3) the tool used for this caffe to pytorch conversion of the network architecture and weights.  

### PyTorch Version

PyTorch version includes the converted Caffe to PyTorch network (architecture and the weights) along with a sample script to run the inference and some sample images. It also has a seperate conda environment file that can be used to create an virtual environment to run the pytorch version.

### PyCaffe Version

PyCaffe version consists of pycaffe implementation of the original matcaffe sample inference script along with the trained caffe network. It also contains some sample images. Finally, it has a conda environment definition file that can be used to quickly create the virtual environment required to use it.

### Caffe to PyTorch Version

This is essentially the code we used to convert and validate Ravi's network from Caffe to PyTorch along with the sample images. We used a modified version of an external tool called [pytorch-caffe](https://github.com/marvis/pytorch-caffe) by marvis. The modified tool is also part of this sub-folder. 

*Note: We are assuming you are using a Linux:Ubuntu system.*

## Installing Single View Depth Estimator
We offer three methods to install our packages:
1. [Through our Conda Package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs package and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies


### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and are inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). From there, simply run:

```
u@pc:~$ conda install acrv-single-view
```

You can see a list of our Conda dependencies in the [`./pytorch-env.yml`](./pytorch_version/pytorch-env.yml) file.

### Pip

Before installing via `pip`, you must have the following system dependencies installed:

- CUDA
- TODO the rest of this list

Then Single View Depth Estimator, and all its Python dependencies can be installed via:

```
u@pc:~$ pip install acrv-single-view
```

### From source

Installing from source is very similar to the `pip` method above due to Single  View Depth Estimator only containing Python code. Simply clone the repository, enter the directory, and install via `pip`:

```
u@pc:~$ pip install -e .
```

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

## Using PyTorch version of Single View Depth Estimator

Once installed, single view depth estimator can be used directly from the command line using Python.

TODO: add details for quickstart scripts that run directly from the command line

Once installed, our pytorch package can be used like any other Python package. It consists of a `SingleViewDepthEstimator` class with currently one main functions for inference and deployment. We are working on adding training and evaluation. Below are some examples to help get you started. 

TODO: THE LINES BELOW!!
If you chose to build from source, the inference script `run_depth_estimator.py` can run from the command line to get sample output.

## Single View Depth Estimator PyTorch API

The code snipet shows how to use single view depth estimator directly in your own projects.

```python
from singleview_depth_estimator import SingleViewDepthEstimator

# Initialise a full RefineNet network with no pre-trained model
sv = SingleViewDepthEstimator()

# Load a previous snapshot from a 152 layer network
sv = SingleViewDepthEstimator(load_snapshot='/path/to/snapshot')

# Get a predicted segmentation as a NumPy image, given an input NumPy image
segmentation_image = sv.predict(image=my_image)

# Save a segmentation image to file, given an image from another image file
sv.predict(image_file='/my/prediction.jpg',
          output_file='/my/segmentation/image.jpg')
```

## Citing our work

If using Single View Depth Estimation in your work, please cite [our original ECCV paper](https://arxiv.org/pdf/1603.04992v2.pdf):

```bibtex
@inproceedings{garg2016unsupervised,
  title={Unsupervised CNN for single view depth estimation: Geometry to the rescue},
  author={Garg, Ravi and Kumar, BG Vijay and Carneiro, Gustavo and Reid, Ian},
  booktitle={European Conference on Computer Vision},
  pages={740--756},
  year={2016},
  organization={Springer}
}
```

TODO: ADD ADDITIONAL CITING?