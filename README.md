# Optum Data Science Challenge

Approx time taken: 4-5 hours

My solution to this task is located in 3 jupyter notebooks and should be viewed in the following order:

1) `eda_initial.ipynb`

2) `eda_visualisation.ipynb`

3) `modelling.ipynb`

I abstracted most of the code out and made a python package called `optum_challenge` if you want to run the notebook interactively

# How to set up environment to run notebooks

1) Python version used was 3.7.5
2) Set up a python virtual environment. I used virtualenv but I don't think it matters. 
2) Install the required packages inside the virtual environment using `pip install -r requirements.txt`
3) Install the self made python package I made using `python setup.py install` or `python setup.py develop`, either
should work
4) Open Jupyter and the notebook should be runnable. From terminal run `jupyter-lab` and it should open
5) I've tested the results of this on a fresh virtual environment and everything works