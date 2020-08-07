# Package - PyCaffe Version.

The original network for Unsupervised CNN for Single View Depth Estimation was implemented in Caffe 1 and can be found here: https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation. This original version provides a Matlab (matcaffe) based script for inference. 

In this package, we provide a PyCaffe based script to run inferences for the original caffe networks along with an Anaconda virtual environment needed to use this work.

## Usage  
To use the PyCaffe script you will first need to create a `conda` virtual environment which has all the relevant dependencies. 

### Virtual Environment
#### Anaconda setup
If you do not have `conda` or [Anaconda](https://www.anaconda.com/distribution/#linux) installed on your system, first get Anaconda setup. We are going to use the command-line installer. 

*Note: We are assuming you are using a Linux:Ubuntu system.*

```bash
# The version of Anaconda may be different depending on when you are installing`

$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ sh Miniconda3-latest-Linux-x86_64.sh

# and follow the prompts. Make sure to select yes for running conda init, otherwise the defaults are generally good.`

```

*You may have to open a new terminal or re-source your ~/.bashrc to get access to the conda command.*

Next, in case you do not have `pip` installed for Python 3.x:

```bash
$ sudo apt install python3-pip
```

#### `conda` environment setup
Now that we have Anaconda/conda setup on our machine, we can clone and create a new environment suitable to run this package.

If you haven't already cloned this repository, please do so now. 
```bash
$ git clone https://github.com/RoboticVisionOrg/unsupervised_depth_estimation.git
$ cd unsupervised_depth_estimation
```

<!-- From the directory `unsupervised_depth_estimation` navigate to the subdirectory `pycaffe_version (which is where this Readme is located) and then navigate to the subdirectory `virtual_environment`. Find the file called `caffe-env.yml`. This files contains the required packages and their dependencies for the `conda` environment.  -->

Create the new virtual environment:

```bash
$ pwd
~/unsupervised_depth_estimation
$ conda env create -f pycaffe_version/virtual_environment/caffe-env.yml
```   

Ensure that the conda environment is created:

```bash
$ conda info --envs
```

If the environment isn't already running (you don't see `caffe-env` before `$` ), start it with:
```bash
$ conda activate caffe-env
```

Now your terminal should indicate that you are in the virtual environment.
```bash
(caffe-env) $
```

### Script
Now that we have an active conda environment, we can run the script. 

Let's run the script as it is. This script will do a list of things:
 1. load the `deploy_resnet50by2_pool.prototxt` and `train_iter_40000.caffemodel` Caffe model from the `network` folder (*you can change the files by modifying the script parameters `net_model` and `net_weights` at the start of the script*),
 2. It will read all images from the `sample_images` subdirectory have '.png' or 'jpg' extension, 
 3. Run inference for each image through and display the output figure of all the images. 
 
 To run the script, change directory to `src` folder: 
 
```bash
# From the root directory of the repository
(caffe-env) $ cd pycaffe_version/src/
(caffe-env) $ python3 depth_kitti_samples.py
``` 
