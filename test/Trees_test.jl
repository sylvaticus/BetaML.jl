using Test
using DelimitedFiles, LinearAlgebra
import MLJBase
const Mlj = MLJBase
using StableRNGs
rng = StableRNG(123)
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

myForest = buildForest(xtrain,ytrain,β=0,maxDepth=20,oob=true)

trees = myForest.trees
treesWeights = myForest.weights
oobError = myForest.oobError
ŷtrain = Trees.predict(myForest, xtrain)
@test accuracy(ŷtrain,ytrain) >= 0.98
ŷtest = Trees.predict(myForest, xtest)
@test accuracy(ŷtest,ytest)  >= 0.96
updateTreesWeights!(myForest,xtrain,ytrain;β=1)
ŷtrain2 = Trees.predict(myForest, xtrain)
@test accuracy(ŷtrain2,ytrain) >= 0.98
ŷtest2 = Trees.predict(myForest, xtest)
@test accuracy(ŷtest2,ytest)  >= 0.96
@test oobError <= 0.1


# ==================================
# NEW TEST
println("Testing random forest regression...")

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

myForest         = buildForest(xtrain,ytrain, minGain=0.001, minRecords=2, maxDepth=3)
trees            = myForest.trees
treesWeights     = myForest.weights

ŷtrain           = Trees.predict(myForest, xtrain)
ŷtest            = Trees.predict(myForest, xtest)
mreTrain         = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.4

updateTreesWeights!(myForest,xtrain,ytrain;β=50)
ŷtrain2 = Trees.predict(myForest, xtrain)
ŷtest2 = Trees.predict(myForest, xtest)
mreTrain = meanRelError(ŷtrain2,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest2,ytest)
@test mreTest <= 0.4

# ==================================
# NEW TEST
println("Testing all possible combinations...")
xtrain = [1 "pippo" 1.5; 3 "topolino" 2.5; 1 "amanda" 5.2; 5 "zzz" 1.2; 7 "pippo" 2.2; 1 "zzz" 1.5; 3 "topolino" 2.1]
ytrain = [x[2][1] <= 'q' ? 5*x[1]-2*x[3] : -5*x[1]+2*x[3] for x in eachrow(xtrain)]
xtest = [2 "pippo" 3.4; 1 "amanda" 1.5; 4 "amanda" 0.5; 2 "topolino" 2.2; 7 "zzz" 3.2]
ytest = [x[2][1] <= 'q' ? 5*x[1]-2*x[3] : -5*x[1]+2*x[3] for x in eachrow(xtest)]
ytrainInt = Int64.(round.(ytrain))

myTree1 = buildTree(xtrain,ytrain)
myForest = buildForest(xtrain,ytrain,oob=true) # TO.DO solved 20201130: If I add here β=1 I have no problem, but local testing gives a crazy error!!!
oobError = myForest.oobError
ŷtrain = predict(myForest,xtrain)
ŷtest = predict(myForest,xtest)
mreTrain = meanRelError(ŷtrain,ytrain)
mreTest  = meanRelError(ŷtest,ytest)

xtrain[3,3] = missing
xtest[3,2] = missing
myForest = buildForest(xtrain,ytrain,oob=true,β=1)
oobError = myForest.oobError
ŷtrain = predict(myForest,xtrain)
ŷtest = predict(myForest,xtest)
mreTrain2 = meanRelError(ŷtrain,ytrain)
mreTest2  = meanRelError(ŷtest,ytest)
@test mreTest2 <= mreTest * 1.5

myTree2 = buildTree(xtrain,ytrainInt)
myTree3 = buildTree(xtrain,ytrainInt, forceClassification=true)
@test typeof(myTree1) <: Trees.DecisionNode && typeof(myTree2) <: Trees.DecisionNode && typeof(myTree3) <: Trees.DecisionNode

#=
# NEW Test
println("Testing MLJ interface for Trees models....")
X, y                           = Mlj.@load_boston
model_dtr                      = DecisionTreeRegressor()
regressor_dtr                  = Mlj.machine(model_dtr, X, y)
(fitresult_dtr, cache, report) = Mlj.fit(model_dtr, 0, X, y)
yhat_dtr                       = Mlj.predict(model_dtr, fitresult_dtr, X)
@test meanRelError(yhat_dtr,y) < 0.02

model_rfr                      = RandomForestRegressor()
regressor_rfr                  = Mlj.machine(model_rfr, X, y)
(fitresult_rfr, cache, report) = Mlj.fit(model_rfr, 0, X, y)
yhat_rfr                       = Mlj.predict(model_rfr, fitresult_rfr, X)
@test meanRelError(yhat_rfr,y) < 0.06

X, y                           = Mlj.@load_iris
model_dtc                      = DecisionTreeClassifier()
regressor_dtc                  = Mlj.machine(model_dtc, X, y)
(fitresult_dtc, cache, report) = Mlj.fit(model_dtc, 0, X, y)
yhat_dtc                       = Mlj.predict(model_dtc, fitresult_dtc, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat_dtc, y)) < 0.0002

model_rfc                      = RandomForestClassifier(maxFeatures=3)
regressor_rfc                  = Mlj.machine(model_rfc, X, y)
(fitresult_rfc, cache, report) = Mlj.fit(model_rfc, 0, X, y)
yhat_rfc                       = Mlj.predict(model_rfc, fitresult_rfc, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat_rfc, y)) < 0.04
=#

# Other MLJ classifier models
#=
import MLJ
X, y                       = Mlj.@load_iris
MLJ.models(MLJ.matching(X,y))
Model                      = MLJ.@load XGBoostClassifier #  DecisionTreeClassifier    #    XGBoostClassifier
model                      = Model()
regressor                  = MLJ.machine(model, X, y)
(fitresult, cache, report) = MLJ.fit(model, 0, X, y)
yhat                       = MLJ.predict(model, fitresult, X)
MLJ.mean(MLJ.LogLoss(tol=1e-4)(yhat, y))
MLJ.evaluate!(regressor, measure=MLJ.LogLoss())
#XGBoostClassifier:
#- fit: https://github.com/alan-turing-institute/MLJModels.jl/blob/3687491b132be8493b6f7a322aedf66008caaab1/src/XGBoost.jl#L600
#- predict :
=#
