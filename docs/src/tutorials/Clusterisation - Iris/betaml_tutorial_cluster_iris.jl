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


# Differently from the [regression tutorial](../A%20regression%20task%20-%20sharing%20bike%20demand%20prediction/betaml_tutorial_regression_sharingBikes.html), we load the data here from `RDatasets`, a package providing standard datasets.
iris = dataset("datasets", "iris")
describe(iris)

#=
# ## Decision Trees and Random Forests

# ### Data preparation
# The first step is to prepare the data for the analysis. This indeed depends already on the model we want to employ, as some models "accept" everything as input, no matter if the data is numerical or categorical, if it has missing values or not... while other models are instead much more exigents, and require more work to "clean up" our dataset.
# Here we start using  Decision Tree and Random Forest models that belong to the first group, so the only things we have to do is to select the variables in input (the "feature matrix", we wil lindicate it with "X") and those representing our output (the values we want to learn to predict, we call them "y"):

x = Matrix{Float64}(iris[:,1:4])
y = Vector{String}(iris[:,5])

# We can now split the dataset between the data we will use for training the algorithm (`xtrain`/`ytrain`), those for selecting the hyperparameters (`xval`/`yval`) and finally those for testing the quality of the algoritm with the optimal hyperparameters (`xtest`/`ytest`). We use the `partition` function specifying the share we want to use for these three different subsets, here 75%, 12.5% and 12.5 respectively. As the dataset is shuffled by default, to obtain replicable results we call `partition` with `rng=copy(FIXEDRNG)`, where `FIXEDRNG` is a fixed-seeded random number generator guaranteed to maintain the same stream of random numbers even between different julia versions. That's also what we use for our unit tests.

((xtrain,xval,xtest),(ytrain,yval,ytest)) = partition([x,y],[0.75,0.125,1-0.75-0.125],rng=copy(FIXEDRNG))
(ntrain, nval, ntest) = size.([ytrain,yval,ytest],1)



clusterOut  = gmm(x,3,mixtures=[FullGaussian() for i in 1:3],minVariance=0.002,rng=copy(FIXEDRNG))
clusterOut.BIC
accuracy(clusterOut.pₙₖ,y,ignoreLabels=true)


clusterOut  = kmeans(x2,3,rng=copy(FIXEDRNG))
accuracy(clusterOut[1],y,ignoreLabels=true)

clusterOut  = kmedoids(x2,3,rng=copy(FIXEDRNG))
accuracy(clusterOut[1],y,ignoreLabels=true)
=#
