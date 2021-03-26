# # A classification task: the prediction of  plant species from floreal measures (the iris tdataset)
# The task is to estimate the influence of several variables (like the weather, the season, the day of the week..) on the demand of shared bicycles, so that the authority in charge of the service can organise the service in the best way.
#
# Data origin:
# - original full dataset (by hour, not used here): [https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# - simplified dataset (by day, with some simple scaling): [https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec]( https://www.hds.utc.fr/~tdenoeux/dokuwiki/en/aec)
# - description: [https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/exam_2019_ace_.pdf)
# - data: [https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip](https://www.hds.utc.fr/~tdenoeux/dokuwiki/_media/en/bike_sharing_day.csv.zip)

# Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).

# ## Library and data loading

using LinearAlgebra, Random, Statistics, DataFrames, CSV, Plots, Pipe, BetaML, BenchmarkTools
import Distributions: Uniform
import DecisionTree, Flux ## For comparisions
using  Test     #src

using RDatasets, BetaML, BenckmarkTools

iris = dataset("datasets", "iris")




# ## Decision Trees and Random Forests
x = convert(Matrix,iris[:,1:4])
y = convert(Array{String,1},iris[:,5])
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3],rng=FIXEDRNG)

myForest       = buildForest(xtrain,ytrain,30,rng=FIXEDRNG)
ŷtrain         = predict(myForest, xtrain)
ŷtest          = predict(myForest, xtest)
trainAccuracy  = accuracy(ŷtrain,ytrain) # 1.00
testAccuracy   = accuracy(ŷtest,ytest)   # 0.956

# ## Perceptron

# ## Neural networks
