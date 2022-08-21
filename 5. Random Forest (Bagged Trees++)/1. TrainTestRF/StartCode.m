ntrees = input('Insert the number of forest trees to grow (e.g. 1000):');
maxDepth = input('Insert the maximum depth of the tree (e.g. 10):');
[Classifications, tree, votes, Percentage] = RandomForest(ntrees, maxDepth);