using Test
using DelimitedFiles, LinearAlgebra

using StableRNGs
rng = StableRNG(123)
using BetaML.Trees

import BetaML.Trees:Leaf

Leaf([1.1,2.1,3.1],2)

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
println("Testing classification of the sepal database using decision trees...")
iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = convert(Array{String,1}, iris[:,5])

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

myTree = buildTree(xtrain,ytrain, splittingCriterion=entropy);
ŷtrain = Trees.predict(myTree, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.99
ŷtest = Trees.predict(myTree, xtest)
@test accuracy(ŷtest,ytest)  >= 0.95

# ==================================
# NEW TEST
println("Testing decision trees regression...")

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

myTree = buildTree(xtrain,ytrain, minGain=0.001, minRecords=2, maxDepth=3)
ŷtrain = Trees.predict(myTree, xtrain)
ŷtest = Trees.predict(myTree, xtest)
mreTrain = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.06
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.3

# ==================================
# NEW TEST
println("Testing classification of the sepal database using random forests...")
iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = convert(Array{String,1}, iris[:,5])

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

forestClassifier = buildForest(xtrain,ytrain,β=1,oob=true)
myForest = forestClassifier[:forest]
treesWeights = forestClassifier[:weights]
oobError = forestClassifier[:oobError]
ŷtrain = Trees.predict(myForest, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.99
ŷtest = Trees.predict(myForest, xtest)
@test accuracy(ŷtest,ytest)  >= 0.96
ŷtrain2 = Trees.predict(myForest, xtrain,weights=treesWeights)
@test accuracy(ŷtrain2,ytrain) >= 0.99
ŷtest2 = Trees.predict(myForest, xtest,weights=treesWeights)
@test accuracy(ŷtest2,ytest)  >= 0.96
@test oobError <= 0.07



# ==================================
# NEW TEST
println("Testing random forest regression...")

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

forestClassifier = buildForest(xtrain,ytrain, minGain=0.001, minRecords=2, maxDepth=3, β=100)
myForest         = forestClassifier[:forest]
treesWeights     = forestClassifier[:weights]

ŷtrain           = Trees.predict(myForest, xtrain)
ŷtest            = Trees.predict(myForest, xtest)
mreTrain         = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.4

ŷtrain2 = Trees.predict(myForest, xtrain,weights=treesWeights)
ŷtest2 = Trees.predict(myForest, xtest,weights=treesWeights)
mreTrain = meanRelError(ŷtrain2,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest2,ytest)
@test mreTest <= 0.4

# ==================================
# NEW TEST
println("Testing all possible combinations...")
xtrain = [1 "pippo" 1.5; 3 "topolino" 2.5; 1 "amanda" 5.2; 5 "zzz" 1.2]
ytrain = [x[2][1] <= 'q' ? 5*x[1]-2*x[3] : -5*x[1]+2*x[3] for x in eachrow(xtrain)]
ytrainInt = Int64.(round.(ytrain))
myTree1 = buildTree(xtrain,ytrain)
myTree2 = buildTree(xtrain,ytrainInt)
myTree3 = buildTree(xtrain,ytrainInt, forceClassification=true)

@test typeof(myTree1) <: Trees.DecisionNode && typeof(myTree2) <: Trees.DecisionNode && typeof(myTree3) <: Trees.DecisionNode
