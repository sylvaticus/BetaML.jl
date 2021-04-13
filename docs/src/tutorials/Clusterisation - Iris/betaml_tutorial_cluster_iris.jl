# # A classification task: the prediction of  plant species from floreal measures (the iris tdataset)
# The task is to estimate the species of a plant given some floreal measurements. It use the classical "Iris" dataset.
# Note that in this example we are using clustering approaches, so we try to understand the "structure" of our data, without relying to actually knowing the true labels ("classes" or "factors"). However we have chosen a dataset for which the true labels are actually known, so to compare the accuracy of the algorithms we use, but these labels will not be used during the algorithms training.

#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)


# ## Library and data loading

using  LinearAlgebra, Random, Statistics, DataFrames, CSV, Plots, Pipe, BetaML, BenchmarkTools, RDatasets
import Distributions: Uniform

using  Test     #src






# Differently from the [regression tutorial](@ref regression_tutorial), we load the data here from `RDatasets`, a package providing standard datasets.
iris = dataset("datasets", "iris")


describe(iris)


# ### Data preparation
# The first step is to prepare the data for the analysis.


x       = Matrix{Float64}(iris[:,1:4])
yLabels = unique(iris[:,5])
y       = integerEncoder(iris[:,5],factors=yLabels)

# We shuffle
((xtest,xval,xtest),(ytest,yval,ytest)) = partition([x,y],[0.98,0.01,0.01],rng=copy(FIXEDRNG))

# Run the gmm(em) algorithm for the various cases...
sphOut   = [gmm(xtest,3,mixtures=[SphericalGaussian() for i in 1:3],minVariance=v, minCovariance=cv, verbosity=NONE) for v in minVarRange, cv in minCovarRange[1:1]]
diagOut  = [gmm(xtest,3,mixtures=[DiagonalGaussian() for i in 1:3],minVariance=v, minCovariance=cv, verbosity=NONE)  for v in minVarRange, cv in minCovarRange[1:1]]
fullOut  = [gmm(xtest,3,mixtures=[FullGaussian() for i in 1:3],minVariance=v, minCovariance=cv, verbosity=NONE) for v in minVarRange, cv in minCovarRange]



# Get the Bayesian information criterion (AIC is also available)
sphBIC  = [sphOut[v,cv].BIC for v in 1:length(minVarRange), cv in 1:1]
diagBIC = [diagOut[v,cv].BIC for v in 1:length(minVarRange), cv in 1:1]
fullBIC = [fullOut[v,cv].BIC for v in 1:length(minVarRange), cv in 1:length(minCovarRange)]

# Compare the accuracy with true categories
sphAcc  = [accuracy(sphOut[v,cv].pₙₖ,y,ignoreLabels=true) for v in 1:length(minVarRange), cv in 1:1]
diagAcc = [accuracy(diagOut[v,cv].pₙₖ,y,ignoreLabels=true) for v in 1:length(minVarRange), cv in 1:1]
fullAcc = [accuracy(fullOut[v,cv].pₙₖ,y,ignoreLabels=true) for v in 1:length(minVarRange), cv in 1:length(minCovarRange)]

plot(minVarRange,[sphBIC diagBIC fullBIC[:,1] fullBIC[:,15] fullBIC[:,30]], markershape=:circle, label=["sph" "diag" "full (cov=0)" "full (cov=0.7)" "full (cov=1.45)"], title="BIC", xlabel="minVariance")
plot(minVarRange,[sphAcc diagAcc fullAcc[:,1] fullAcc[:,15] fullAcc[:,30]], markershape=:circle, label=["sph" "diag" "full (cov=0)" "full (cov=0.7)" "full (cov=1.45)"], title="Accuracies", xlabel="minVariance")





clusterOut  = gmm(x,3,mixtures=[FullGaussian() for i in 1:3],minVariance=0.002,rng=copy(FIXEDRNG))
clusterOut.BIC
accuracy(clusterOut.pₙₖ,y,ignoreLabels=true)


clusterOut  = kmeans(x2,3,rng=copy(FIXEDRNG))
accuracy(clusterOut[1],y,ignoreLabels=true)

clusterOut  = kmedoids(x2,3,rng=copy(FIXEDRNG))
accuracy(clusterOut[1],y,ignoreLabels=true)
