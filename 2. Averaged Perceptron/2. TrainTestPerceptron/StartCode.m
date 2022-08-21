% Behnam Sherafat
% Project
% Machine Learning
% Average Perceptron Algorithm

learning_rate = input('Insert the best learning rate calculated from cross validation section (i.e. 1, 0.5, 0.2, 0.1, 0.01, 0.001):');
max_epoch = input('Insert the number of epochs for training (i.e. 20 ):');

[submission_file, num_updates, results] = TrainTestAveragedPerceptron(learning_rate, max_epoch);