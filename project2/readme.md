## Folder structure
- **code** contains our code used to create the data
- **csvData** contains (without explicitely deleting anything) data used to create plots (and much more).
- **selected_results** contains selected runs that can be benchmarked against our code.
  - file 1 blablabla
  - file 2 blablabla
- **data** contains the data file(s) being analysed
- **figures** contains figures (and some more) that were used in the report.
- **report** contains the report.

## Explanation of code
__Each file in the code folder contains a "running example" in the bottom of the file explaining how it is executed.__

- **a.py** creates data that compares different stochastic gradient methods with each other as well as with the OLS fit, as a function of learning rate, number of epochs and batch size. Creates data
- **a_ridge.py** compares the optimal Ridge Regression fit with the fit for different Stochastic Gradient methods as a function of the learning rate and the regularization parameter. The function plots (but doesn't create data).
- **b.py** creates data (from the Korean geographic data) for our own NN and Scikit learn: The test MSE and the train MSE as a function of the learning rate and the regularization parameter are written to a file. 
- **plot_a.py** plots the data created in **a.py**
- **plot_b.py** plots the Scikit-Learn fit and the fit using our own NN in the same plot for both test and train data using SGD and the sigmoid activation function.
- **plot_b2.py** plots the fit using our own NN in the same plot for both test and train data comparing ADAM & RMSProp using the sigmoid activation function.
- **plot_b3.py** plots the fit using our own NN in the same plot for test data comparing the sigmoid, the tanh, the ReLU and the LeakyReLU activation functions. 
- **test_functions.py** contain test functions to check wether our implementations give expected solutions to simple problems.
