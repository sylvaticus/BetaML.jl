using Test
using DelimitedFiles, LinearAlgebra

import Random:seed!
seed!(123)

using Bmlt.Nn


println("*** Testing Neural Network...")

# ==================================
# New test
# ==================================
println("Testing basic NN behaviour...")
x = [0.1,1]
y = [1,0]
l1   = DenseNoBiasLayer(2,2,w=[2 1;1 1],f=identity)
l2   = VectorFunctionLayer(2,2,f=softMax)
mynn = buildNetwork([l1,l2],squaredCost,name="Simple Multinomial logistic regression")
o1 = forward(l1,x)
#@code_warntype forward(l1,x)
o2 = forward(l2,o1)
orig = predict(mynn,x')[1,:]
#@code_warntype predict(mynn,x')[1,:]
ϵ = squaredCost(o2,y)
#@code_warntype  squaredCost(o2,y)
lossOrig = loss(mynn,x',y')
#@code_warntype  loss(mynn,x',y')
dϵ_do2 = dSquaredCost(o2,y)
#@code_warntype dSquaredCost(o2,y)
dϵ_do1 = backward(l2,o1,dϵ_do2)
#@code_warntype backward(l2,o1,dϵ_do2)
dϵ_dX = backward(l1,x,dϵ_do1)
#@code_warntype backward(l1,x,dϵ_do1)
l1w = getParams(l1)
#@code_warntype
l2w = getParams(l2)
#@code_warntype
w = getParams(mynn)
#@code_warntype getParams(mynn)
#typeof(w)
origW = deepcopy(w)
l2dw = getGradient(l2,o1,dϵ_do2)
#@code_warntype getGradient(l2,o1,dϵ_do2)
l1dw = getGradient(l1,x,dϵ_do1)
#@code_warntype
dw = getGradient(mynn,x,y)
#@code_warntype getGradient(mynn,x,y)
y_deltax1 = predict(mynn,[x[1]+0.001 x[2]])[1,:]
#@code_warntype predict(mynn,[x[1]+0.001 x[2]])[1,:]
lossDeltax1 = loss(mynn,[x[1]+0.001 x[2]],y')
#@code_warntype
deltaloss = dot(dϵ_dX,[0.001,0])
#@code_warntype
@test isapprox(lossDeltax1-lossOrig,deltaloss,atol=0.0000001)
l1wNew = l1w
l1wNew[1][1,1] += 0.001
setParams!(l1,l1wNew)
lossDelPar = loss(mynn,x',y')
#@code_warntype
deltaLossPar = 0.001*l1dw[1][1,1]
lossDelPar - lossOrig
@test isapprox(lossDelPar - lossOrig,deltaLossPar,atol=0.00000001)
η = 0.01
#w = gradientDescentSingleUpdate(w,dw,η)
#w = w - dw * η
w = gradSub.(w, gradMul.(dw,η))
#@code_warntype gradSub.(w, gradMul.(dw,η))
#@code_warntype
setParams!(mynn,w)
loss2 = loss(mynn,x',y')
#@code_warntype
@test loss2 < lossOrig
for i in 1:10000
    w  = getParams(mynn)
    dw = getGradient(mynn,x,y)
    w  = gradSub.(w,gradMul.(dw,η))
    setParams!(mynn,w)
end
lossFinal = loss(mynn,x',y')
@test predict(mynn,x')[1,1]>0.96
setParams!(mynn,origW)
train!(mynn,x',y',epochs=10000,batchSize=1,sequential=true,verbosity=NONE,optAlg=SGD(η=t->η,λ=1))
#@code_warntype train!(mynn,x',y',epochs=10000,batchSize=1,sequential=true,verbosity=NONE,optAlg=SGD(η=t->η,λ=1))
lossTraining = loss(mynn,x',y')
#@code_warntype
@test isapprox(lossFinal,lossTraining,atol=0.00001)


# ==================================
# NEW Test
println("Testing regression if it just works with manual derivatives...")

xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
xtest = [0.5 0.6; 0.14 0.2; 0.3 0.7; 2.0 4.0]
ytest = [1.1; 0.36; 1.0; 6.0]
l1 = DenseLayer(2,3,w=[1 1; 1 1; 1 1], wb=[0 0 0], f=tanh, df=dtanh)
l2 = DenseNoBiasLayer(3,2, w=[1 1 1; 1 1 1], f=relu, df=drelu)
l3 = DenseLayer(2,1, w=[1 1], wb=[0], f=identity,df=didentity)
mynn = buildNetwork([l1,l2,l3],squaredCost,name="Feed-forward Neural Network Model 1",dcf=dSquaredCost)
train!(mynn,xtrain,ytrain,batchSize=1,sequential=true,epochs=100,verbosity=NONE,optAlg=SGD(η=t -> 1/(1+t),λ=1))
#@benchmark train!(mynn,xtrain,ytrain,batchSize=1,sequential=true,epochs=100,verbosity=NONE,optAlg=SGD(η=t -> 1/(1+t),λ=1))
avgLoss = loss(mynn,xtest,ytest)
@test  avgLoss ≈ 1.599729991966362
expectedOutput = [0.7360644412052633, 0.7360644412052633, 0.7360644412052633, 2.47093434438514]
predicted = dropdims(predict(mynn,xtest),dims=2)
@test any(isapprox(expectedOutput,predicted))

# ==================================
# NEW TEST
# ==================================
println("Testing using AD...")


ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

l1   = DenseLayer(2,3,w=ones(3,2), wb=zeros(3))
l2   = DenseLayer(3,1, w=ones(1,3), wb=zeros(1))
mynn = buildNetwork([l1,l2],squaredCost,name="Feed-forward Neural Network Model 1")
train!(mynn,xtrain,ytrain,epochs=1000,sequential=true,batchSize=1,verbosity=NONE,optAlg=SGD(η=t->0.01,λ=1))
avgLoss = loss(mynn,xtest,ytest)
@test  avgLoss ≈ 0.0032018998005211886
expectedOutput = [0.4676699631752518,0.3448383593117405,0.4500863419692639,9.908883999376018]
predicted = dropdims(predict(mynn,xtest),dims=2)
@test any(isapprox(expectedOutput,predicted))
#predicted = dropdims(predict(mynn,xtrain),dims=2)
#ytrain

# ==================================
# NEW TEST
# ==================================
println("Going through Multinomial logistic regression (using softMax)...")
#=
using RDatasets
using Random
using DataFrames: DataFrame
using CSV
Random.seed!(123);
iris = dataset("datasets", "iris")
iris = iris[shuffle(axes(iris, 1)), :]
CSV.write(joinpath(@__DIR__,"data","iris_shuffled.csv"),iris)
=#

iris     = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5])
y_oh = oneHotEncoder(y)

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
ytrain_oh = y_oh[1:ntrain,:]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

l1   = DenseLayer(4,10, w=ones(10,4), wb=zeros(10),f=relu)
l2   = DenseLayer(10,3, w=ones(3,10), wb=zeros(3))
l3   = VectorFunctionLayer(3,3,f=softMax)
mynn = buildNetwork([l1,l2,l3],squaredCost,name="Multinomial logistic regression Model Sepal")
train!(mynn,scale(xtrain),ytrain_oh,epochs=500,batchSize=8,sequential=true,verbosity=LOW,optAlg=SGD(η=t->0.001,λ=1))

ŷtrain = predict(mynn,scale(xtrain))
ŷtest  = predict(mynn,scale(xtest))
trainAccuracy = accuracy(ŷtrain,ytrain,tol=1)
testAccuracy  = accuracy(ŷtest,ytest,tol=1)
@test testAccuracy >= 0.8 # set to random initialisation/training to have much better accuracy
