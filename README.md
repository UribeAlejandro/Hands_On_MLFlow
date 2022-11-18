# Hands-on MLFlow <!-- omit in toc -->

## Table of Contents <!-- omit in toc -->

- [Introduction](#introduction)
- [Setting up the environment](#setting-up-the-environment)
  - [Environment variables](#environment-variables)
  - [MLFlow Server](#mlflow-server)
- [About the dataset](#about-the-dataset)
- [Training Loop](#training-loop)
  - [Classical Machine Learning](#classical-machine-learning)
  - [Deep Learning](#deep-learning)
- [Acknowledgments](#acknowledgments)
- [Disclaimer](#disclaimer)

## Introduction

The MNIST dataset is useful for those who want to try learning techniques and pattern recognition methods on real-world data. The classification task is tackled using classical Machine Learning and Deep Learning approaches. On top of the training loop, there is an experiment tracker that will allow the data practitioner to decide which approach is better.

## Setting up the environment

It is recommended to use `virtualenvwrapper`. Find [here](https://virtualenvwrapper.readthedocs.io/en/latest/) the instructions to install and use it.

### Environment variables

Set the `MLFlow` environment variables as follows

```bash
export MLFLOW_TRACKING_URI=sqlite:///experiment/mlflow/db/mydb.sqlite
export ARTIFACT_ROOT=./experiment/mlflow/mlruns/
```

You can also find them in [mlflow.cfg](mlflow.cfg).

### MLFlow Server

To enable experiment tracking and model registry start `MLFlow` server as follows:

```bash
mlflow server --default-artifact-root $ARTIFACT_ROOT --backend-store-uri $MLFLOW_TRACKING_URI
```

## About the dataset

The *MNIST* dataset contains 70,000 grayscale small images (28x28) of labeled handwritten digits, from *0 - 9*. This problem is often called the "Hello World" of *Machine Learning* because anyone who learns *Machine Learning* tackles this problem at any time.

Further information about the dataset can be found on the following web pages:

- [MNIST - Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- [MNIST 784 - OpenML](https://www.openml.org/d/554)
- [MNIST - Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/mnist)

Some examples of the digits are shown below.
![MNIST](./assets/MNIST.png)

## Training Loop

Three basic components:

- ETL
- Model
  - Training
  - Evaluation
- Deployment

Depending on the model evaluation or data practitioner criteria the ETL or the Model may suffer changes.

### Classical Machine Learning

A simplified approach using binary classification, a `5-detector`:

- Classical Machine Learning
  - Scikit-Learn: Stochastic Gradient Descent
  - Scikit-Learn: Random Forest

### Deep Learning

A multiclass classification with the proper architecture:

- Convolutional Neural Network
  - Using scaled images
  - using `n_pca_components` to keep 95% of the explained variance

## Acknowledgments

This hands-on experience with *Computer Vision* common projects was inspired by **Tensorflow in Practice by Laurence Moroney** - *Coursera* and the concepts explained in **Hands-On Machine Learning with Scikit-Learn, Keras &amp; Tensorflow by Aureélien Géron** - *O'Reily*.

- [Tensorflow in Practice - Coursera Specialization](https://www.coursera.org/specializations/tensorflow-in-practice)
- [Hands\-On Machine Learning with Scikit\-Learn, Keras &amp; Tensorflow \- O'Reily Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

## Disclaimer

Sections of code were taken from both sources stated in [Acknowledgments](#acknowledgments) and all the datasets used in this notebook are open-sourced.