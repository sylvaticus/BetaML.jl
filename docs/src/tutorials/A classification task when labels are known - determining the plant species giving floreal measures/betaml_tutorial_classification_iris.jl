# # A classification task: the prediction of  plant species from floreal measures (the iris tdataset)
# The task is to estimate the species of a plant given some floreal measurements. It is a very
#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)

# Note that even if we are estimating a time serie, we are not using here a recurrent neural network as we assume the temporal dependence to be negligible (i.e. $Y_t = f(X_t)$ alone).

# ## Library and data loading

using LinearAlgebra, Random, Statistics, DataFrames, CSV, Plots, Pipe, BetaML, BenchmarkTools, RDatasets
import Distributions: Uniform
import DecisionTree, Flux ## For comparisions
using  Test     #src


# Differently from the [regression tutorial](../A%20regression%20task%20-%20sharing%20bike%20demand%20prediction/betaml_tutorial_regression_sharingBikes.html), we load the data here from `RDatasets`, a package providing standard datasets
iris = dataset("datasets", "iris")
describe(y)


# ## Decision Trees and Random Forests
x = convert(Matrix,iris[:,1:4])
y = convert(Array{String,1},iris[:,5])
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.7,0.3],rng=FIXEDRNG)

myForest       = buildForest(xtrain,ytrain,30,rng=FIXEDRNG);
ŷtrain,         = predict(myForest, xtrain)
ŷtest          = predict(myForest, xtest)
trainAccuracy  = accuracy(ŷtrain,ytrain) # 1.00
testAccuracy   = accuracy(ŷtest,ytest)   # 0.956

# ## Perceptron

# ## Neural networks
