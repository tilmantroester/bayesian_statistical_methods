# Bayesian Statistical Methods and Data Analysis

## About this course

The course aims to focus mostly on showing how to use statistical tools, rather than giving a detailed introduction to theory behind probability and statistics. There are many great books and courses on this topic, which I encourage you to look at before the lectures. The resources listed here all are available online, either directly or through the ETH library.

- Weighing the odds, a course in probability and statistics, Williams, 2001 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117170967205503). A good introduction to probability theory and statistics with a high level of mathematical rigour.
- Bayesian Data Analysis, Gelman, 2013 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117222397805503). The title says it all.
- Information Theory, Inference, and Learning Algorithms, MacKay, 2003 [Link](http://www.inference.org.uk/itprnn/book.pdf). Heavy on the information theory but also covers inference methods nicely. The exercises come with solutions.
- Practical Statistics for Astronomers, Wall, 2012 [ETH library](https://eth.swisscovery.slsp.ch/permalink/41SLSP_ETH/lshl64/alma99117170816205503). Short, with a focus on practical applications. Many of the examples and exercises are from astrophysics but are generally applicable, especially for the physical sciences, which often come a bit short in general statistics textbooks. Solutions and data sets are [available online](https://www.astro.ubc.ca/people/jvw/ASTROSTATS/pracstats_web_ed2.html).

## This repository

- `lessons`: The lesson notebooks, which are used to create the slides
- `slides`: Slides for the lessons
- `pdf`: PDF versions of the lecture notebooks
- `exercise_solutions`: Solutions to selected exercises
- `course_tools`: Tools for creating this course, such as a script to generate the slides from the notebooks and a python package with helper functions.

## Using JupyterLab

If you are seeing this in the course JupyterLab, you are probably in the directory with the course materials:

<img src="assets/jupyterlab_landing.png" height="200">

To make your live easier later on with version control, go up one level in the directory structure and make your own directory for the code you are going to write in this course:

<img src="assets/jupyterlab_toplevel_annotated.png" height="400">



## On your own device

All the materials can be access outside of JupyterLab as well. The repository with the course materials can be found [on github](https://github.com/tilmantroester/bayesian_statistical_methods/tree/fs2023).

To set up your own computing environment I strongly recommend anaconda. Either using [miniconda](https://docs.conda.io/en/latest/miniconda.html) or the much faster [mamba](https://mamba.readthedocs.io/en/latest/installation.html#install-script) implementation.