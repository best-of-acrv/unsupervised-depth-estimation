# Package - Pytorch Version.

The network for Unsupervised CNN for Single View Depth Estimation was implemented in Caffe 1. We have converted the original Caffe's deployment network `deploy_resnet50by2_pool` to a Pytorch version `deploy_renet50by2_pool_pytorch.pth` which has both the architecture and weights. 

This package holds the inference scripts and pytorch network along with the virtual environment file that can be used to run it.

## Usage  
To use this tool you will need to create a `conda` virtual environment which has all the relevant dependencies. 

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
$ git clone git@bitbucket.org:acrv/unsupervised_depth_estimation_converted.git
$ cd unsupervised_depth_estimation_converted
```

<!-- From the directory `unsupervised_depth_estimation_converted` navigate to the subdirectory `pytorch_version` (which is where this Readme is located) and then navigate to the subdirectory `virtual_environment`. Find the file called `pytorch-env.yml`. This files contains the required packages and their dependencies for the `conda` environment.  -->

Create the new virtual environment:

```bash
$ pwd
~/unsupervised_depth_estimation_converted
$ conda env create -f pytorch_version/virtual_environment/pytorch-env.yml
```   

Ensure that the conda environment is created:

```bash
$ conda info --envs
```

If the environment isn't already running (you don't see `pytorch-env` before `$` ), start it with:
```bash
$ conda activate pytorch-env
```

Now your terminal should indicate that you are in the virtual environment.
```bash
(pytorch-env) $
```
### Network
If you have just pulled this repository, the pytorch network will be compressed. Uncompress it on your local copy before the next steps. 

First navigate to where the network file is:
```bash
$ pwd
~/unsupervised_depth_estimation_converted
$ cd pytorch_version/network
```

You should see the network tar file there. Let's uncompress it using `tar -xf <network-file>.tar.xz` 

```bash
$ tar -xf deploy_resnet50by2_pool_pytorch_modelAndWeights.tar.xz
$ cd ../../
```


### Script
Now that we have an active conda environment, we can run the script. 

Let's run the script as it is. This script will do a list of things:

 1. Load the pre-saved PyTorch network `deploy_renet50by2_pool_pytorch.pth` from the `network` folder (*different network can be chosen by changing the `self.pytorch_model_file` value at the start of the script*),
 2. it will read all images from the `sample_images` subdirectory have '.png' or 'jpg' extension, 
 3. run inference for each image through the PyTorch model,
 4. and Display the output figure of all the images. 
 
 To run the script, change directory to `src` folder: 
 
```bash
# From the root directory of the repository
(pytorch-caffe) $ cd pytorch_version/src
(pytorch-caffe) $ python3 run_pytorch_model.py
``` 

### Behind the scenes
You might have noticed the file called `pytorch_net.py`. This class inherits from the parent class of the conversion tool and will pass all the conversion commands to the conversion tool. Currently this class exists to allow you to save and transport the PyTorch network without having to also wrap the tool. That is, if you use the saved network architecture generated using the tool,(until I find a way around this), you will need the `pytorch_net.py` class along with the saved network to load it into a PyTorch. 


