# Package - Convert Caffe Network to Pytorch Version.
The network for Unsupervised CNN for Single View Depth Estimation was developed, trained and deployed in Caffe 1. We have converted the original Caffe's deployment network `deploy_resnet50by2_pool` to a Pytorch version `deploy_renet50by2_pool_pytorch.pth` which has both the architecture and weights. 

This package holds the scripts and external tool that can be used to convert other Caffe networks for the Unsupervised CNN for Single View Depth Estimation work; along with running, comparing and checking the inference of the various forms of the network.      

## Tool used for conversion
We used an external tool called [pytorch-caffe](https://github.com/marvis/pytorch-caffe) by marvis. We had to minimally modify the tool to work for our network, and the link to modified tool can be found under the subdirectory `convertor_tool`.

## Usage  
To use this tool you will need to create a `conda` virtual environment which has all the relevant dependencies. 

### Virtual Environment
#### Anaconda setup
If you do not have `conda` or [Anaconda](https://www.anaconda.com/distribution/#linux) installed on your system, first get Anaconda setup. We are going to use the command-line installer. 

*Note: We are assuming you are using a Linux:Ubuntu system.*

 ```
# The version of Anaconda may be different depending on when you are installing`

$ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# and follow the prompts. The defaults are generally good.`

```

*You may have to open a new terminal or re-source your ~/.bashrc to get access to the conda command.*

Next, in case you do not have `pip` installed for Python 3.x:

```
$ sudo apt install python3-pip
```
#### `conda` environment setup
Now that we have Anaconda/conda setup on our machine, we can clone and create a new environment suitable to run this package.

If you haven't already cloned this repository, please do so now. 
```buildoutcfg
$ git clone git@bitbucket.org:acrv/unsupervised_depth_estimation_converted.git
```

From the directory `unsupervised_depth_estimation_converted` navigate to the subdirectory `caffe_to_pytorch` (which is where this Readme is located) and then navigate to the subdirectory `virtual_environment`. Find the file called `pytorch-caffe-env.yml`. This files contains the required packages and their dependencies for the `conda` environment. 

Create the new virtual environment:

```
$ conda env create -f pytorch-caffe-env.yml
```   

Ensure that the conda environment is created:

```
$ conda info --envs
```

If the environment isn't already running (you don't see `pytorch-caffe` before `$` ), start it with:
```
$ conda activate pytorch-caffe
```

Now your terminal should indicate that you are in the virtual environment.
```
(pytorch-caffe) $
```

### Script
Now that we have an active conda environment, we can run the script. 

Let's run the script as it is. This script will do a list of things:
 1. It will read all images from the `sample_images` subdirectory have '.png' or 'jpg' extension, 
 2. read the `deploy_resnet50by2_pool` Caffe Network, 
 3. call the conversion tool and convert this network to a Pytorch version (store it in memory) 
 4. load the pre-saved PyTorch network `deploy_renet50by2_pool_pytorch.pth`,
 5. Run inference for each image through the original Caffe Model, the in-memory PyTorch model and the loaded PyTorch model.
 6. Finally, it will run some assertion checks to ensure the output is within the acceptable range (same up to 4 decimal places)
 7. and Display the output figure of all the images. 
 
 To run the script, change directory to `src` folder: 
 
 ```
(pytorch-caffe) $ pwd
/unsupervised_depth_estimation_converted/caffe_to_pytorch

(pytorch-caffe) $ cd src

(pytorch-caffe) $ python3 pytorch_conversion.py
``` 

You might have noted, that the script currently does not save the model. To save the model, open the python script, and toggle the `self.save_model` from `False` to `True`.

While you are there, you would have noticed the `self.debug_mode` flag. Toggling it to `True` will enable additional printing of data flow through the various stages of the script with information we found useful during debugging.   

### Behind the scenes
You might have noticed the file called `pytorch_net.py`. This class inherits from the parent class of the conversion tool and will pass all the conversion commands to the conversion tool. Currently this class exists to allow you to save and transport the PyTorch network without having to also wrap the tool. That is, if you save the network architecture using this script (until I find a way around this), you will need the `pytorch_net.py` class along with the saved network to load it into a PyTorch. For code inspiration on how to do that, check out the `pytorch_version` subdirectory - specifically, `src/run_pytorch_model.py`.


