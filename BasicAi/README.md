# BasicAi

This project is an attempt to learn and understand the structure and implementation of [FastAi][fai] whose projects I highly admire.

I take NO CREDIT for the code in this repository, all credit goes to the wonderful people at [FastAi][fai].

## The Goal

The goal of this project is to distill down the FastAi library and have a lightweight infrastructure that can be used to help structure ML and DL projects in a succinct way.

Each part of the pipeline should ideally simply provide guidance for which one can fill it up with one's own requirements

Currently, the project is set up for supervised learning task. The hope is to adapt it as I learn more about the other tasks.

Feature submission is also more than welcomed!

## Structure of Project

This project is split into 4 main parts:

* The data builder
* The training loop
* The callbacks attached to the training loop
* The optimizer that updates the weights each iteration of the training loop

## The Data builder

The data builder consist of everything from loading data to using it from training

Specifically:

* Retrieve Data
* Pre-process the data if needed
* Split the Data into train, valid, and test
* Label the Data
* Apply any data augmentation lazily
* Have the data available as tensor when indexing
* Put everything into a dataset
* Put the dataset into a loader that retrieves data base on batch size

## The training Loop

The current training loop:



[fai]:https://www.fast.ai/