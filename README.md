# Copula Ordinal Regression
Copula Ordinal Regression denotes is a statistical learning methods in which the goal is to predict a set of discrete and ordinal variables. For example, predicting the intensity of different facial action units on a scale from 0 to 5 starts can be considered an mutli-task ordinal regression task.

##### Install instructions:
requires:
* Python (>= 2.6 or >= 3.3),
* Numpy
* Scipy
* Theano (>= 0.6)
* scikit-learn

First, get the code from Github:
```sh
git clone https://github.com/RWalecki/copula_ordinal_regression.git
```

Next, go into the directory where the clone was placed and run:
```
python setup.py install
```

##### Test the installation:
Once you have installed copula_ordinal_regression, you should run the nosetests to make sure that everything works as it should.
Therefore, run:
```
nosetests .
```
The tests should not take longer than a few seconds and now you are ready to use copula_ordinal_regression. Enjoy!

##### Run the demos:
'__copula_classification.py__' contains an example of how to train the model and use it to predict structured outputs (Action Units).
The file '__copula_cross_validation.py__' contains an example of an exhaustive parameter grid search using cross validation.

##### License and Citations
Copula Ordinal Regression is free of charge for research purposes only.
If you use is, please cite:
* "Copula Ordinal Regression for Joint Estimation of Facial Action Unit Intensity", R. Walecki, O. Rudovic, V. Pavlovic, M. Pantic. Proceedings of IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2016). Las Vegas, Nevada, June 2016.
