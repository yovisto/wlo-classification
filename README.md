# WLO Discipline Classification

A [Docker](https://docker.com/)/[Python](https://www.python.org/)/[Keras](https://keras.io/)/[Tensorflow](https://www.tensorflow.org/) utility to train and predict *subject areas* for the [WLO project](https://github.com/openeduhub/) dataset.

 
## Prerequisites

- Install [Docker](https://docker.com/).

- Build the Docker container.

```
sh build.sh
```

## Training

- The following script retrieves and processes the latest [dataset](https://github.com/openeduhub/oeh-wlo-data-dump), which results in the `data/wirlernenonline.oeh.csv` file containing the relevant documents (documents with a discipline property).

```
sh prepareData.sh
```

- This script initiates the training, which results in the model file `data/wirlernenonline.oeh.h5` and the file with class labels `data/wirlernenonline.oeh.npy` (existing files will be overwritten without warning).

```
sh runTraining.sh
```

## Prediction

- To test the prediction just query the model with an arbitrary text.

```
sh runPrediction.sh "Der Satz des Pythagoras lautet: a^2 + b^2 = c^2."
```

The result is a list of tuples of a score and its corresponding class name (name of discipline). Only the top three items are retrieved in descending order.


