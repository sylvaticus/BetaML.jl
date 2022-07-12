using Test
using DelimitedFiles, LinearAlgebra
import MLJBase
const Mlj = MLJBase
using StableRNGs
#rng = StableRNG(123)
using BetaML

#TESTRNG = FIXEDRNG # This could change...
TESTRNG = StableRNG(123)

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
myTree = buildTree(xtrain,ytrain,rng=copy(TESTRNG))
m = DTModel(rng=copy(TESTRNG))
train!(m,xtrain,ytrain)

ŷtrain  = predict(myTree, xtrain,rng=copy(TESTRNG))
ŷtrain2 = predict(m,xtrain)

@test accuracy(ŷtrain,ytrain,rng=copy(TESTRNG)) >= 0.8
@test ŷtrain == ŷtrain2

xtest = [
    "Green"  3;
    "Yellow" 4;
    "Red"    2;
    "Red"    1;
    "Yellow" 3
]

ytest  = ["Apple","Apple","Grape","Grape","Lemon"]
ŷtest  = predict(myTree, xtest,rng=copy(TESTRNG))
ŷtest2 = predict(m, xtest)

@test accuracy(ŷtest,ytest,rng=copy(TESTRNG)) >= 0.8
@test ŷtest == ŷtest2

@test report(m) == Dict(:jobIsRegression => 0,:maxDepth => 3, :dimensions => 2, :trainedRecords => 5, :avgDepth => 2.6666666666666665)
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

myTree = buildTree(xtrain,ytrain, splittingCriterion=entropy,rng=copy(TESTRNG))
ŷtrain = predict(myTree, xtrain,rng=copy(TESTRNG))
@test accuracy(ŷtrain,ytrain,rng=copy(TESTRNG)) >= 0.98
ŷtest = predict(myTree, xtest,rng=copy(TESTRNG))
@test accuracy(ŷtest,ytest,rng=copy(TESTRNG))  >= 0.95

# ==================================
# NEW TEST
println("Testing decision trees regression...")

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

myTree = buildTree(xtrain,ytrain, minGain=0.001, minRecords=2, maxDepth=3,rng=copy(TESTRNG))
ŷtrain = predict(myTree, xtrain,rng=copy(TESTRNG))
ŷtest = predict(myTree, xtest,rng=copy(TESTRNG))
mreTrain = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.06
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.3
m = DTModel(minGain=0.001,minRecords=2,maxDepth=3,rng=copy(TESTRNG))
train!(m,xtrain,ytrain)
@test predict(m,xtrain) == ŷtrain

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

myForest = buildForest(xtrain,ytrain,β=0,maxDepth=20,oob=true,rng=copy(TESTRNG))

trees = myForest.trees
treesWeights = myForest.weights
oobError = myForest.oobError
ŷtrain = predict(myForest, xtrain,rng=copy(TESTRNG))
@test accuracy(ŷtrain,ytrain,rng=copy(TESTRNG)) >= 0.96
ŷtest = predict(myForest, xtest,rng=copy(TESTRNG))
@test accuracy(ŷtest,ytest,rng=copy(TESTRNG))  >= 0.96
updateTreesWeights!(myForest,xtrain,ytrain;β=1,rng=copy(TESTRNG))
ŷtrain2 = predict(myForest, xtrain,rng=copy(TESTRNG))
@test accuracy(ŷtrain2,ytrain,rng=copy(TESTRNG)) >= 0.98
ŷtest2 = predict(myForest, xtest,rng=copy(TESTRNG))
@test accuracy(ŷtest2,ytest,rng=copy(TESTRNG))  >= 0.96
@test oobError <= 0.1

m = RFModel(maxDepth=20,oob=true,beta=0,rng=copy(TESTRNG))
train!(m,xtrain,ytrain)
m.opt.rng=copy(TESTRNG) 
ŷtrainNew = predict(m,xtrain)
@test ŷtrainNew == ŷtrain 
m.opt.rng=copy(TESTRNG) 
ŷtestNew = predict(m,xtest)
@test ŷtestNew == ŷtest 
#=
m.options.rng=copy(TESTRNG) 
m.learnableparameters.weights = updateTreesWeights!(myForest,xtrain,ytrain;β=1, rng=copy(TESTRNG))
m.learnableparameters.weights == myForest.weights


m.options.rng=copy(TESTRNG) 
ŷtrain2New = predict(m,xtrain)
ŷtrain2New == ŷtrain2

m.learnableparameters.trees == trees

m.options.rng=copy(TESTRNG) 
ŷtrain3 = predict(m, xtrain)
m.options.rng=copy(TESTRNG) 
ŷtest3  = predict(m, xtest)
ŷtrain2 == ŷtrain3

@test accuracy(ŷtest2,ytest,rng=copy(TESTRNG)) ≈ accuracy(ŷtest3,ytest,rng=copy(TESTRNG))
@test info(m)[:oobE] ≈ oobError
=#

predictionsByTree = [] # don't use weights...
for i in 1:30
    old = trees[i]
    new = m.par.trees[i]
    pold = predict(old,xtrain, rng=copy(TESTRNG))
    pnew = predict(old,xtrain, rng=copy(TESTRNG))
    push!(predictionsByTree,pold == pnew)
end

@test sum(predictionsByTree) == 30

# ==================================
# NEW TEST
println("Testing random forest regression...")

ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

myForest         = buildForest(xtrain,ytrain, minGain=0.001, minRecords=2, maxDepth=3,rng=copy(TESTRNG))
trees            = myForest.trees
treesWeights     = myForest.weights

ŷtrain           = predict(myForest, xtrain,rng=copy(TESTRNG))
ŷtest            = predict(myForest, xtest,rng=copy(TESTRNG))
mreTrain         = meanRelError(ŷtrain,ytrain)
@test mreTrain <= 0.08
mreTest  = meanRelError(ŷtest,ytest)
@test mreTest <= 0.4

updateTreesWeights!(myForest,xtrain,ytrain;β=50)
ŷtrain2 = predict(myForest, xtrain,rng=copy(TESTRNG))
ŷtest2 = predict(myForest, xtest,rng=copy(TESTRNG))
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

myTree1 = buildTree(xtrain,ytrain,rng=copy(TESTRNG))
myForest = buildForest(xtrain,ytrain,oob=true,rng=copy(TESTRNG)) # TO.DO solved 20201130: If I add here β=1 I have no problem, but local testing gives a crazy error!!!
oobError = myForest.oobError
ŷtrain = predict(myForest,xtrain,rng=copy(TESTRNG))
ŷtest = predict(myForest,xtest,rng=copy(TESTRNG))
mreTrain = meanRelError(ŷtrain,ytrain)
mreTest  = meanRelError(ŷtest,ytest)

xtrain[3,3] = missing
xtest[3,2] = missing
myForest = buildForest(xtrain,ytrain,oob=true,β=1,rng=copy(TESTRNG))
oobError = myForest.oobError
ŷtrain = predict(myForest,xtrain,rng=copy(TESTRNG))
ŷtest = predict(myForest,xtest,rng=copy(TESTRNG))
mreTrain2 = meanRelError(ŷtrain,ytrain)
mreTest2  = meanRelError(ŷtest,ytest)
@test mreTest2 <= mreTest * 1.5

m = RFModel(oob=true,beta=1,rng=copy(TESTRNG))
train!(m,xtrain,ytrain)
m.opt.rng=copy(TESTRNG) # the model RNG is consumed at each operation
ŷtest2 = predict(m,xtest)

@test meanRelError(ŷtest,ytest,normDim=false,normRec=false) ≈ meanRelError(ŷtest2,ytest,normDim=false,normRec=false)

myTree2 = buildTree(xtrain,ytrainInt,rng=copy(TESTRNG))
myTree3 = buildTree(xtrain,ytrainInt, forceClassification=true,rng=copy(TESTRNG))
@test typeof(myTree1) <: Trees.DecisionNode && typeof(myTree2) <: Trees.DecisionNode && typeof(myTree3) <: Trees.DecisionNode

# ==================================
# NEW TEST
println("Testing trees with unsortable and missing X values...")

abstract type AType end
mutable struct SortableType<:AType
    x::Int64
    y::Int64
end

mutable struct UnsortableType<:AType
    x::Int64
    y::Int64
end
isless(x::SortableType,y::SortableType) = x.x < y.x

SortableVector = [SortableType(2,4),SortableType(1,5),SortableType(1,8),SortableType(12,5),
    SortableType(6,2),SortableType(2,2),SortableType(2,2),SortableType(2,4),
    SortableType(6,2),SortableType(1,5),missing,SortableType(2,4),
    SortableType(1,8),SortableType(12,5)]
UnSortableVector = [UnsortableType(2,5),UnsortableType(1,3),UnsortableType(1,8),UnsortableType(2,6),
    UnsortableType(6,3),UnsortableType(7,9),UnsortableType(2,5),UnsortableType(2,6),
    missing,UnsortableType(3,2),UnsortableType(6,3),UnsortableType(2,5),
    UnsortableType(7,9),UnsortableType(7,9)]

data = Union{Missing,Float64, String,AType}[
    0.9 0.6 "black" "monitor" 10.1
    0.3 missing "white" "paper sheet" 2.3
    4.0 2.2 missing "monitor"  12.5
    0.6 0.5 "white" "monitor" 12.5
    3.8 2.1 "gray" "car" 54.2
    0.3 0.2 "red" "paper sheet" 2.6
    0.1 0.1 "white" "paper sheet" 2.5
    0.3 0.2 "black" "monitor" 11.3
    0.1 0.2 "black" "monitor" 9.8
    0.31 0.2 "white" "paper sheet" 3.7
    3.2 1.9 "gray" "car" 64.3
    0.4 0.25 "white" "paper" 2.7
    0.9 0.4 "black" "monitor" 12.5
    4.1 2.1 "gray" "monitor" 13.2
]

X = hcat(data[:,[1,2,3]],UnSortableVector)
y = convert(Vector{String},  data[:,4])

((xtrain,xtest),(ytrain,ytest)) = Utils.partition([X,y],[0.7,0.3],shuffle=false,rng=copy(TESTRNG))

modelβ = buildForest(xtrain,ytrain,5,rng=copy(TESTRNG))
ŷtestβ = predict(modelβ,xtest,rng=copy(TESTRNG))
accβ   = accuracy(ŷtestβ,ytest,rng=copy(TESTRNG))
@test accβ >= 0.25

# ==================================
# NEW TEST
println("Testing MLJ interface for Trees models....")
X, y                           = Mlj.@load_boston
model_dtr                      = DecisionTreeRegressor(rng=copy(TESTRNG))
regressor_dtr                  = Mlj.machine(model_dtr, X, y)
(fitresult_dtr, cache, reportobj) = Mlj.fit(model_dtr, 0, X, y)
yhat_dtr                       = Mlj.predict(model_dtr, fitresult_dtr, X)
@test meanRelError(yhat_dtr,y) < 0.02

model_rfr                      = RandomForestRegressor(rng=copy(TESTRNG))
regressor_rfr                  = Mlj.machine(model_rfr, X, y)
(fitresult_rfr, cache, reportObj) = Mlj.fit(model_rfr, 0, X, y)
yhat_rfr                       = Mlj.predict(model_rfr, fitresult_rfr, X)
@test meanRelError(yhat_rfr,y) < 0.06

X, y                           = Mlj.@load_iris
model_dtc                      = DecisionTreeClassifier(rng=copy(TESTRNG))
regressor_dtc                  = Mlj.machine(model_dtc, X, y)
(fitresult_dtc, cache, reportObj) = Mlj.fit(model_dtc, 0, X, y)
yhat_dtc                       = Mlj.predict(model_dtc, fitresult_dtc, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat_dtc, y)) < 0.0002

model_rfc                      = RandomForestClassifier(maxFeatures=3,rng=copy(TESTRNG))
regressor_rfc                  = Mlj.machine(model_rfc, X, y)
(fitresult_rfc, cache, reportObj) = Mlj.fit(model_rfc, 0, X, y)
yhat_rfc                       = Mlj.predict(model_rfc, fitresult_rfc, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat_rfc, y)) < 0.04


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
