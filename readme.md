# 3sigma

## Introduction: a side hobby

This project started as an exploration into neural network as well as a means to understand and implement many of the concepts covered in the [fastai course v3](https://github.com/fastai/course-v3/tree/master/nbs/dl2).

Currently, much work has been made into understanding the loading and cleaning process of datasets. In particular, csv files as 1d convolution images.

More attempts will be made in the coming weeks to dive into 2d images and more.

## List of To-Dos

- Fix up Learner
  - Callbacks
    - Recorder to track stats and progress
      - Add metric measurement
        - Loss, any other metrics
      - time elapse for each epoch
      - Notified of progress
      - Track stats within each layer of model
        - Mean and std of each weight layer.
        - Activation landscape of each layer
      - Plotting and visualizing the tracked stats
    - Find optimum hyper-parameters
    - annealing
      - scheduling parameters
      - discriminative learning
  - batch-norm paper implementation
  - lsuv

- Refactor
  - Refactor `Itemlist` to show tensor when indexing or iterating through it
    - Use `.show` method to view file with appropriate viewer
  - Refactor file_opener to include a `to_tensor` method.
  - Refactor `Pipeline` to take `Itemlist` as parameter instead of params for `Itemlist`
  - Refactor `Learner` to have more appropriate and less error prone callback names
- Callbacks
