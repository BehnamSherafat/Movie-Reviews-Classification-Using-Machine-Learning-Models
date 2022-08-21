% Behnam Sherafat
% Project
% Machine Learning
% Support Vector Machines (SVM) Algorithm

initial_learning_rate = input('Insert the best learning rate calculated from cross validation section (i.e. 0.001 ):');
max_epoch_cross = input('Insert the number of epochs for training (i.e. 5 ):');
tradeoff = input('Insert the value of tradeoff C (i.e. 10^9 ):');

[Classifications, a] = SVM(initial_learning_rate, max_epoch_cross, tradeoff);