## Folder structure
- **code** contains our code used to create the data
- **csvData** contains (without explicitely deleting anything) data used to create plots (and much more).
- **selected_results** contains selected runs that can be benchmarked against our code.
  - The three "OLSMSE*"-files were created by running a.py as "python a.py 200 5 5 16 10 100 0.1".
  - The map "2b1_tanhADAM200010_epoch100" was created using b.py with the parameters that can be found in info.txt. Scikit Learn was turned off, so these files are empty.
  - d_benchmark Contains two different outputs from classifying NN. To compare, run d.py with BENCHMARK = True and all others false. Relevant code is found near the top.
  - e_benchmark Contains two different outputs from own and SciKit- Learn Multinomial logistic regression. To compare, run e.py with BENCHMARK = True and all others false. Relevant code is found near the bottom.
- **data** contains the data file(s) being analysed
- **figures** contains figures (and some more) that were used in the report.
- **report** contains the report.

## Explanation of code
__Each file in the code folder contains a "running example" in the bottom of the file explaining how it is executed.__

- **a.py** creates data that compares different stochastic gradient methods with each other as well as with the OLS fit, as a function of learning rate, number of epochs and batch size. Creates data
- **a_ridge.py** compares the optimal Ridge Regression fit with the fit for different Stochastic Gradient methods as a function of the learning rate and the regularization parameter. The function plots (but doesn't create data).
- **b.py** creates data (from the Korean geographic data) for our own NN and Scikit learn: The test MSE and the train MSE as a function of the learning rate and the regularization parameter are written to a file. 
- **d.py**  Contains four sections: 
            - Benchmark: Runs a small script testing the NN on classification
            - Activation comparison: Produces the plot in Figure 12 in the report.
            - Architecture Comparison: Produces the plot in Figure 11 in the report.
            - SciKit Learn: Gets result from the Scikit Learn implementation of a NN for classification
- **e.py**  Contains four sections: 
            - Benchmark: Runs a small script testing the Scikit- Learn and own implementation of Softmax regression
            - Wrong Class Example: Produces the image in Figure 13 in the report.
            - Accuracy v. Learning rate: Produces the plots in Figure 14 and 15 in the report.
            - Compare SciKit Learn: Gets results from the Scikit Learn- and own implementation of Softmax Regression. Produces Table 1 in the report.

- **function_library.py** contains the fundament of this article, namely, the Neural Network class, the SGD class and the Logistic Regression class, as well as  other useful and relevant functions.
- **plot_a.py** plots the data created in **a.py**
- **plot_b.py** plots the Scikit-Learn fit and the fit using our own NN in the same plot for both test and train data using SGD and the sigmoid activation function.
- **plot_b2.py** plots the fit using our own NN in the same plot for both test and train data comparing ADAM & RMSProp using the sigmoid activation function.
- **plot_b3.py** plots the fit using our own NN in the same plot for test data comparing the sigmoid, the tanh, the ReLU and the LeakyReLU activation functions. 
- **test_functions.py** contain test functions to check wether our implementations give expected solutions to simple problems.
