## Folder structure
- **code** contains our code used to create the data
- **csvData** contains (without explicitely deleting anything) data used to create plots (and much more).
- **selected_results** contains selected runs that can be benchmarked against our code.
  - boosting_2testreduced_ex.csv and boosting_2trainreduced_ex.csv were created with boosting2.py the way the file looks like now. (crossval=1, M up to 1000).
  - boosting_1testreduced_ex.csv and boosting_1trainreduced_ex.csv were created with boosting1.py the way the file looks like now. (crossval=1, M = 100, 200, 400).
  - ridgereduced.csv was created using PCA.py 
  - The files "nn_" were created using neural_network.py the way the file looks like now.
- **data** contains the data file(s) being analysed
- **figures** contains figures (and some more) that were used in the report.
- **report** contains the report.

## Explanation of code
__Each file in the code folder contains a "running example" in the bottom of the file explaining how it is executed.__

- **bagging.py** implements bagging and plots the train and test MAE as function of the number of trees. Parameters need to be cahnged within the program itself. Bagging is implemented with Scikit-Learn.
- **boosting_1.py** implements boosting and writes to an output file for different values of the tree depth. Parameters need to be changed within the program itself. Boosting is implemented with XGboost, the scikit-learn compatible implementation is used.
-  **boosting_2.py** implements boosting and writes to an output file the test/train MAE as function of M for different values of $\L1_{reg}$ and $\eta$.  This version is GPU-optimized and can run for larger M's, XGBoost's DMatrices are being used.
-  **decision_tree.py** plots the train and test MAE as function of tree depth for a Single Decision Tree, implemented with Scikit-Learn.
- **function-library.py** contains useful algorithms that are used by several of the other files.
- **neural_network.py** contains the implementation of an FFNN with keras/TensorFlow that writes to file the test and train error for different learning rates, regularization rates for L1 and L2, and the ELU/sigmoid activation function. 
- **PCA.py** does Ridge Regression and plots the MAE as function of the number of used PCA components. PCA is implemented with Scikit-Learn.
- the different **plot_XXX.py** functions either do very litte calculations and then plot what their filename says, or they simply read from the files created by the other programs.
- **random_forest.py** implements Random Forests and plots the train and test MAE as function of the number of trees. Parameters need to be cahnged within the program itself, such as the number of used predictors. Random Forests are implemented with Scikit-Learn.
- **test_functions.py** contains test functions.
