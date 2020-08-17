using Test
using DelimitedFiles, LinearAlgebra

import Random:seed!
seed!(123)

using BetaML.Trees


println("*** Testing Decision trees/Random Forest algorithms...")

# ==================================
# NEW TEST
# ==================================
println("Testing basic classification of decision trees...")
# ---------------------
xtrain = [
    "Green"  3.0;
    "Yellow" 3.0;
    "Red"    1.0;
    "Red"    1.0;
    "Yellow" 3.0;
]

ytrain = ["Apple",  "Apple", "Grape", "Grape", "Lemon"]
myTree = buildTree(xtrain,ytrain)

#print(myTree)

ŷtrain = Trees.predict(myTree, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.8

xtest = [
    "Green"  3;
    "Yellow" 4;
    "Red"    2;
    "Red"    1;
    "Yellow" 3
]

ytest = ["Apple","Apple","Grape","Grape","Lemon"]
ŷtest = Trees.predict(myTree, xtest)
@test accuracy(ŷtest,ytest) >= 0.8

#print(myTree)

# ==================================
# NEW TEST
# ==================================

println("Testing classification of the sepal database using decision trees...")
iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = convert(Array{String,1}, iris[:,5])

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

myTree = buildTree(xtrain,ytrain, splittingCriterion="entropy");
ŷtrain = Trees.predict(myTree, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.99
ŷtest = Trees.predict(myTree, xtest)
@test accuracy(ŷtest,ytest)  >= 0.95
