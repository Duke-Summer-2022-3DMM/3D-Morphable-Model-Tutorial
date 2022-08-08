# 3DMM-Tutorial
3D Morphable Model Tutorial


## Installation
### Requirements 
For the moment this code has been tested only on linux, and python 3.9.

### Installing miniconda (optional)
You can skip this step and use the tool of your choice to create the python virtual environment with python 3.9.*

We recommend miniconda for a light and easy installation. If you already have it installed in your system, you can skip this step, otherwise follow the instructions 
https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

### Init clone the repo and init venv
```
$ git clone THISREPO
$ cd 3DMM-TUTORIAL
$ conda create --prefix ./env/main 
$ conda activate env/main
```

### Install required python packages
``` 
$ make install
```