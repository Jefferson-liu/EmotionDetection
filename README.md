# Emotion Detection Project

A CNN project that can detect if you are happy, angry, surprised. or neutral

The dataset is a modified version of the FER2013 dataset from Kaggle without some of the emotions, namely sad, frightened, and disgust.

To use, all you need is the model.h5 and model.json files alongside the test.py file.
If you want to train your own model, run the train.py file.

## Comments on the FER2013 dataset
The dataset has some flaws, including the fact that a significant fraction of images in the dataset have large watermarks over the faces, and that some images have ambiguous emotions (i.e face closed and neutral mouth labelled sad, but that would be more similar to neutral). Also, there is a large difference in the amount of data available for some classes (i.e around 7000 images in the train data labelled "Happy", but only around 3000 labelled "Surprise". This problem is also true for the test dataset.

I fixed these problems on my end by first removing the classes that had ambiguous emotional faces, and then limiting the amount of data sampled from each training set to 3000, and limiting data from test set to 1000


