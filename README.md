# Moment Propagation 
This repository contains the source code (as described in the publications below) to transform a given neural network (NN), trained with standard dropout beforehand, into an NN that is able to approximate MC dropout in a single forward-pass by propagating the first two moments (Expectation E, Variance V) through the network

<img src="https://user-images.githubusercontent.com/49025372/111510086-b68f2c80-874d-11eb-8144-e3f8d21a8eb8.png" width="1000">

# Single-Shot MC Dropout Approximation
Source Code for the ICML 2020 Paper on Uncertainty &amp; Robustness in Deep Learning 

If you use this code in academic context, please cite the following publication:
<pre>
@article{Brach2020SingleSM,
  title={Single Shot MC Dropout Approximation},
  author={Kai Brach and B. Sick and Oliver D{\"u}rr},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.03293}
}
</pre>
# Single-Shot Bayesian Approximation for Neural Networks
coming soon...

# Important Prerequisites
I worked with `Conda` for creating the virtual environment and used `PIP` as package manager. Here are some useful links if you not familar with them:
* [Overview: Conda vs PIP vs Venv](https://docs.conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands)
* [PIP reference guide](https://pip.pypa.io/en/stable/reference/pip/)
* [Conda reference guide](https://docs.conda.io/projects/conda/en/latest/commands.html#conda-general-commands)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. [environment.yml](https://github.com/kaibrach/DNN-MP/blob/master/environment.yml) contains all modules you need


# GPU Support for TensorFlow 2.3.1
1. Download and Install [Cuda Tolkit 10.1 ](https://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe)
2. Download [Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip) and copy the files in the zip file
to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1`

**Note:** 
I did not update to **TensorFlow 2.4.x** because I ran into trouble with my GPU because TensorFlow >=2.4.0 needs CUDA 11. 
Nevertheless, feel free to update to TensorFlow 2.4.x, the source code should work with this version also.
[Here](https://www.tensorflow.org/install/gpu) you hand find some information about TensorFlow and GPU support


# Work with Conda Environment

1. Activate an existing conda environment: 

        conda activate <envname>

2. Export an existing conda environment (no build numbers): 

        conda env export --no-builds -f <envname>.yml
  
3. Create an environment from config file: 
        
        conda env create -f <envname>.yml
        
4. List conda environments

        conda env list
        
5. Remove conda environments
        
        conda env remove -n <envname>


# Work with Jupyter Notebooks
1. Activate an existing conda environment: 

        conda activate <envname>
2. Adding Jupyter kernels 

   If you want to use different virtual environments for jupyter kernels you have to execute the following in every conda environment.
   Later you can see that there are different kernels in your Jupyter notebook you can choose.

        pip install ipykernel (if not already done)
        python -m ipykernel install --user --name=<kernelname>
    
3. Start Jupyter Notebook Server

        jupyter notebook
        
4. Kernel Errors

   If you found an Kernel error (Clicking on that red button gives more details about the error)
   
   `ImportError: DLL load failed while importing win32api: The specified module could not be found.`
   
   You must run the comand prompt with administrator privilegs and execute the following
   
       python [environment path]\Scripts\pywin32_postinstall.py -install
   
5. List Jupyter kernels

       jupyter kernelspec list

6. Remove Jupyter kernels

       jupyter kernelspec remove <kernelname>

7. Jupyter extensions

  There are a lot of configurable nbextensions that could be very useful
  * Jupytext (creates py file from notebook, good for debugging)
  * Scratchpad (Ctrl-B)
  * Snippets
  * Table of Contents
  * ....

  The extensions can be enabled via `nbextensions_configurator` (Nbextensions tab in the main window when starting jupyter) or
  via commandline for example:
        
        jupyter nbextension enable --py widgetsnbextension
