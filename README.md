# Copula Ordinal Regression
Copula Ordinal Regression is a statistical lerning method in which the goal is to predict a set of discrete variables that are ordinal. For example, predicting the intensity of different facial action units on a scale from 0 to 5 starts can be considered an mutli-output ordinal regression task.
This is a generalization of the multi-label classification task, where the set of classification problem is restricted to binary classification, and of the multi-class classification task.

##### Install instructions:
requires:
* Python (>= 2.6 or >= 3.3),
* Numpy
* Scipy
* Theano (>= 0.7)
* scikit-learn

First, get the code from Github:
```sh
git clone https://github.com/RWalecki/copula_ordinal_regression.git
```

Next, go into the directory where the clone was placed and run the installation script:
```
cd copula_ordinal_regression
python setup.py install
```

##### Test the installation:
Once you have installed copula_ordinal_regression, you should run the nosetests before using it.
Therefore, run:
```
nosetests .
```
The tests should not take longer than a few seconds. You are ready to use copula_ordinal_regression. Enjoy!

##### Quick-Start:
You should read through the scripts that are located in the __demo__ folder to understand how the models are applied. The files are heavily commented and are basically a small tutorial.
'__copula_classification.py__' contains an example of how to train the model and use it to predict structured outputs (Action Units).
The file '__copula_cross_validation.py__' contains an example of an exhaustive parameter grid search using cross validation.
___
##### License and Citations
Copula Ordinal Regression is free of charge for research purposes only.
If you use is, please cite:
* "Copula Ordinal Regression for Joint Estimation of Facial Action Unit Intensity", R. Walecki, O. Rudovic, V. Pavlovic, M. Pantic. Proceedings of IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2016). Las Vegas, Nevada, June 2016.
[[pdf](http://ibug.doc.ic.ac.uk/media/uploads/documents/copula_ordinal_regression__cvpr2016_final.pdf "pdf")]
