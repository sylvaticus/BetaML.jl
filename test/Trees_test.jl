using Test
using DelimitedFiles, LinearAlgebra

import Random:seed!
seed!(123)

using BetaML.Trees


println("*** Testing Decision trees/Random Forest algorithms...")


# ==================================
# NEW TEST
# ==================================
println("Testing classification on sepal database using gini...")


iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),",",skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5])
y_oh = oneHotEncoder(y)

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
ytrain_oh = y_oh[1:ntrain,:]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]
