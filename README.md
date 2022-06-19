# Regularized Polynomial Regression
Polynomial regression learns a function <img width="284" alt="Screen Shot 2022-06-19 at 8 10 11 AM" src="https://user-images.githubusercontent.com/31524675/174487839-bfee31c7-fc2e-419f-a74b-e2a634566992.png">, where d represents the polynomial's highest degree. This can be reqritten in the form of a linear model with d features.

<img width="378" alt="Screen Shot 2022-06-19 at 8 12 21 AM" src="https://user-images.githubusercontent.com/31524675/174487954-e7699b76-c938-458e-bd5f-dcb2573b4673.png">

We explore the effect of increasing regularization on the model.

# Results
<img width="665" alt="Screen Shot 2022-06-19 at 8 23 00 AM" src="https://user-images.githubusercontent.com/31524675/174488433-3c57fc9a-29c1-4f1b-bdb9-d47a64f24234.png">

# Setup

**You only need to do setup once**, then for future homeworks you can run `conda activate cse446`.

## Miniconda installation
Before you start working with this repo, you should install Anaconda.

### Download links

[Anaconda (default)](https://www.anaconda.com/products/individual#Downloads)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) (if running low on disk space)

You can find more detailed instructions for installation [at this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages).

## Environment Setup
First make sure you have at least ~5GB of free drive.
From this directory run:
```
conda env create -f environment.yaml
conda activate cse446
pip install -e .
```

First command might take long time. Especially if your connection is slow.

**Then whenever you come back to work on homeworks** just run: `conda activate cse446`

![A quite long gif visualizing how to setup enviornment](./README_media/setup-env.gif)

