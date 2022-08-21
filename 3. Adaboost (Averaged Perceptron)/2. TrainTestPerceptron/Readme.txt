****For implementing all codes, please open "run.sh" file********

Pre-processing algorithm:
1)  Initial parameters are defined for different methods;
2) Data is read and stored it into tables;
3) Data is saved as sparse data for further use. I used a function named “Sparse”, which saves it as a sparse table. This table can be used for further processes;
4) Sparse data is loaded;
5) Features’ dimension of data is made the same as each other;
6) Sparse data is changed into full tables;
7) The important key point is that because the initial data set is large and consists of about 74000 features, the reading and processing time would be so long.
Thus, I used Principal Component Analysis (PCA) to reduce the dimension of the features.
The main idea of PCA is to reduce the dimensionality of a data set consisting of many variables correlated with each other, either heavily or lightly, while retaining the variation present in the data set up to the maximum extent.
Using PCA, I have reduced the number of features to different values (i.e. 100, 300, 500, 1000, 2000, and 10000) which can be defined at the beginning of code “desired_num_feat” to test the most optimum major features that decrease the computational time and increase the accuracy; and 8) Finally, specific algorithm for each method is implemented.