from small_function_library import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd 


""" PARAMETERS START"""
np.random.seed(sum([ord(c) for c in "corona"]))
k = 4
nr_lambdas = 80
smallest_lambda = -3
deg = 4
datapoints=200
nr_noiseVal = 40
""" PARAMETERS END"""


#Creating a list of values for function noise and lambda values
function_noise = np.linspace(0,1,nr_noiseVal)
lambda_val = np.logspace(smallest_lambda,1,nr_lambdas)

#Setting up the dataset apart from z, which appears in the for loop
#to get the average over two independent runs (This is only for the case of no random.seed
#which has been added to make the results reproducible)
x=np.random.uniform(0,1,datapoints)
y=np.random.uniform(0,1,datapoints)

#Initializing a matrix for the MSE's and a list for the lambda values 
#resulting in the smalles MSE at a given model complexity
MSE_test_kfoldRidge_lambda = np.zeros((nr_lambdas,nr_noiseVal))
min_lambda = np.zeros(nr_noiseVal)

#setting up design matrix
X=DesignMatrix_deg2(x,y,deg,True)

t = 0
#Run over all noise values in list
for noise in function_noise:
    #z is calculated for each noise value 
    z=FrankeFunction(x,y)+noise*np.random.normal(0,1, datapoints)
    for i in range(nr_lambdas):
        #Averaging over two runs to hopefully avoid extreme outliers (does not work with random.seed)
        MSE_test_kfoldRidge_lambda[i,t] = (KCrossValRidgeMSE(X,z,k,lambda_val[i]) + KCrossValRidgeMSE(X,z,k,lambda_val[i]))/2
    t += 1
    print(t, "out of ",nr_noiseVal)



#picking out the lambda that resulted in the smalles MSE
for i in range(len(min_lambda)):
    min_lambda[i] = min(MSE_test_kfoldRidge_lambda[:,i])

plt.plot(function_noise,np.log10(min_lambda))
plt.xlabel(r"Function noise ($\sigma$)")
plt.ylabel('$\log_{10}(\lambda)$'+' of minimum MSE')
plt.title(r"$\lambda$ resulting in smallest MSE for each $\sigma$ (poly. deg.= " +str(deg)+r', #datapoints = '+str(datapoints) +r' )' )
plt.show()

print('Random seed: '+'sum([ord(c) for c in "corona"]' + ',k = {0}, nr_lambdas = {1}, smallest_lambda = {2}, deg = {3}, datapoints = {4}, nr_noiseVal = {5}'.format(k , nr_lambdas  , smallest_lambda  , deg  , datapoints , nr_noiseVal ))

print('Lambda resulting in smallest MSE')
print(min_lambda)


"""
RUN EXAMPLES:

###############################################################################################################################################
Random seed: sum([ord(c) for c in "corona"],k = 4, nr_lambdas = 80, smallest_lambda = -3, deg = 4, datapoints = 200, nr_noiseVal = 40
Lambda resulting in smallest MSE
[0.00435159 0.0053001  0.00674357 0.01176623 0.01649331 0.02352091
 0.03208273 0.04074036 0.04821939 0.05245691 0.07219758 0.0742093
 0.09544288 0.13183089 0.1626366  0.15889123 0.17450586 0.20418628
 0.18672632 0.22714777 0.30777509 0.2829011  0.35567369 0.32183287
 0.3618193  0.51812621 0.53293298 0.55327763 0.57738223 0.44308535
 0.51168843 0.65366006 0.66805317 0.65787447 0.83007886 0.92389982
 0.97536661 0.91069425 0.77592796 0.92177808]
###############################################################################################################################################

"""


