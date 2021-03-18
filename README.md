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
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. [environment.yml](https://github.com/kaibrach/DNN-MP/blob/master/environment.yml) contains all modules you need

# GPU Support for Tensorflow 2.3.1
1. Download and Install [Cuda Tolkit 10.1 ](https://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe)
2. Download [Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip) and copy the files in the zip file
to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1`

# Work with Conda Environment

1. Activate an existing conda environment: 

        conda activate <envname>

2. Export an existing conda environment (no build numbers): 

        conda env export --no-builds -f <envname>.yml
  
3. Create an environment from config file: 
        
        conda env create -f <envname>.yml
# Work with Jupyter Notebooks
1. Activate an existing conda environment: 

        conda activate <envname>
2. Using virtual environment for jupyter kernels

        pip install ipykernel (if not already done)
        python -m ipykernel install --user --name=tf_moment_propagation
    
3. Start Jupyter Notebook Server

        jupyter notebook
4. Kernel Errors

   If you found an Kernel error (Clicking on that red button gives more details about the error)
   `ImportError: DLL load failed while importing win32api: The specified module could not be found.`
   
       python [environment path]\Scripts\pywin32_postinstall.py -install
   
   

