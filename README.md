# allophone-classifier
Classifies allophones of English stops using the TIMIT corpus

## Introduction
This repository contains the code used in my Honours Project in Cognitive Science at McGill University.
Specifically, it is a convolutional neural network programmed with the Keras machine learning library which classifies allophones of English stop phonemes.
It also includes the code to parse the idiosyncratic file format used in the TIMIT corpus which makes up the training and test data of the classifier.
Ultimately the classifier preforms fairly well on the TIMIT corpus, with accuracies ranging from 85 to 95 depending on the phoneme in question.


## Installation
The majority of the relative paths are defined in the `settings.py` file although there are a few other arbitrary paths hard coded throughout the code.
To run it, you need both the AutoVOT software, specifically the `VotFrontEnd2` binary, as well as access to the TIMIT corpus.

The code itself is written in Python 3, using Keras and Numpy primarily. 
