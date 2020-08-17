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

#q1 = Question(1,"Green")
#q2 = Question(2,3)
#p1,p2 = partition(training_data,q1)
#infoGain(p1,p2,gini(training_data))
#p1,p2 = partition(training_data,q2)

#a = ["a","b","a","c","d"]
#c = class_counts(a)
#gini([1 2 3; 2 3 3; 4 3 3])
#gini([1 2 3; 2 3 1; 4 3 2; 2 3 4; 1 2 5])
#gini(training_data)
#findBestSplit(training_data)

myTree = buildTree(xtrain,ytrain)

#print(myTree)

ŷtrain = predict(myTree, xtrain)
accuracy(ŷtrain,ytrain)

xtest = [
    "Green"  3;
    "Yellow" 4;
    "Red"    2;
    "Red"    1;
    "Yellow" 3
]

ytest = ["Apple","Apple","Grape","Grape","Lemon"]
ŷtest = predict(myTree, xtest)
accuracy(ŷtest,ytest)

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

myTree = buildTree(xtrain,ytrain);
ŷtrain = predict(myTree, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.99
ŷtest = predict(myTree, xtest)
@test accuracy(ŷtest,ytest)  >= 0.95


print(myTree)
