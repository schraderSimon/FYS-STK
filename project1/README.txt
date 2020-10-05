Here is an explanation of the folder layout:
The "code" folder contains the code used to produce the data.
The "selected_results" folder contains selected runs that can be benchmarked against our code.
 - Korea20000.csv contains data (the parameters can be found in the file) for fitting the geographic data with 20.000 points. The individual columns contain data for different polynomial degrees (starting at mindeg). Created with create_geographic_data.py 
- Korea50000_NOBOOTSTRAP.csv contains data contains data (the parameters can be found in the file) for fitting the geographic data with 20.000 points. Bootstrap was not used. The individual columns contain data for different polynomial degrees (starting at mindeg). Created with create_geographic_data.py
- MSE_data_Franke.csv contains the different MSE's for all three regression methods used. The individual columns contain data for different polynomial degrees (starting at 1). Created with calculate_different_MSE.py.
- OLS_data_sigma0.100data800.csv contains the different data required to plot the bias-variance-tradeoff plots. Created with a.py

The "report" map contains the report.
The "csvData" folder contains (without explicitely deleting anything) data used to create plots (and much more).
The "data" map contains .tif containing the Korea geographic data.
The "figures" map contains figures (and some more) that were used in the report.
Here is an explanation of what the single code files are and what they do. Each file contains a "running example" in the bottom of the file explaining how it is executed:

a.py - OLS Fit to the Franke Function. parameters are to be put in the Source code manually as there are so many.

a_betaConfidenceIntervals.py - Calculates the 95'th percentile confidence intervals of the estimators in OLS and ridge. Prints the maximum standard deviation found among the estimators.
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

c.py - Compares Test MSE for kfold- cross validation and Bootstrap for different model complexities
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

c_kfoldcross5to10vsbootstrap.py - compares test MSE for Bootstrap and kfoldCross with different values of k over model complexity

d.py - OUTPUTS CSV FILE CONTAINING MSE COMPARISONS BETWEEN BOOTSTRAP, KFOLD-OLS AND KFOLD-RIDGE OVER A SPAN OF POLYNOMIAL DEGREES
	outputs csv files containing either an MSE comparison between bootstrap, kfold with OLS and kfold with Ridge over multiple model complexities, or MSE from kfold with Ridge over several values for Lambda
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

d2.py - Ridge Fit to the Franke Function. parameters are to be put in the Source code manually as there are so many.

d_lambdaToNoise.py - Finds the optimal lambda for minimizing MSE for a fixed model complexity and varying function noise
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

d_lambdaToPolycomplx.py - Finds the optimal lambda for minimizing MSE for a fixed function noise and varying model complexity
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

optimal_k_kfoldcross.py - Plots MSE resulting from different values of k in kfold- cross validation using OLS with varying model complexity
	RUN EXAMPLES AT THE BOTTOM OF THE FILE

e.py - LASSO fit to the Franke Function. parameters are to be put in the Source code manually as there are so many.

plot_from_data.py - Plots the MSE from the data created using a, d2 and e.

plot_geographic_data.py - Plots MSE for the three methods using the file created by create_geographic_data.py
  takes as arguments: Filename and wether bootstrap was used creating the data.

create_picture.py - Recreates the picture using the ideal parameters from create_geographic_data.py for OLS and Ridge.

calculate_different_MSE.py - Plots the MSE for the Franke Function using the same parameters for all 3 Regression methods.
  parameters are to be put in the Source code manually as there are so many.

create_geographic_data.py - Finds the MSE using k-fold Cross validation, the MSE, R2-Score, Bias^2, Variance using Bootstrap for the Image data.
  Most parameters are to be put in the Source code manually as there are so many.

testfunksjoner.py - tests if functions are implemented correctly.

small_function_library.py - library containing important functions that were used throughout the report
