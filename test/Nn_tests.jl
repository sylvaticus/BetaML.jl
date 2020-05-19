using Test

import Random:seed!
seed!(1234)

using Bmlt.Utilities
using Bmlt.Nn
using Bmlt.NnDefaultLayers

println("*** Testing Neural Network...")
# ==================================
# TEST 1: no AD

println("Going through Test1...")
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
xtest = [0.5 0.6; 0.14 0.2; 0.3 0.7; 2.0 4.0]
ytest = [1.1; 0.36; 1.0; 6.0]
l1 = DenseLayer(tanh,2,3,w=[1 1; 1 1; 1 1], wb=[0 0 0], df=dtanh)
l2 = DenseNoBiasLayer(relu,3,2, w=[1 1 1; 1 1 1], df=drelu)
l3 = DenseLayer(linearf,2,1, w=[1 1], wb=[0], df=dlinearf)
mynn = buildNetwork([l1,l2,l3],squaredCost,name="Feed-forward Neural Network Model 1",dcf=dSquaredCost)
train!(mynn,xtrain,ytrain,maxepochs=100,η=nothing,rshuffle=false,nMsgs=0)
avgLoss = losses(mynn,xtest,ytest)
@test  avgLoss ≈ 1.599729991966362
expectedOutput = [0.7360644412052633, 0.7360644412052633, 0.7360644412052633, 2.47093434438514]
predicted = dropdims(predictSet(mynn,xtest),dims=2)
@test any(isapprox(expectedOutput,predicted))


# ==================================
# Test 2: using AD
# ==================================
println("Going through Test2...")


ϵtrain = [1.023,1.08,0.961,0.919,0.933,0.993,1.011,0.923,1.084,1.037,1.012]
ϵtest  = [1.056,0.902,0.998,0.977]
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtrain[i] for (i,x) in enumerate(eachrow(xtrain))]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*ϵtest[i] for (i,x) in enumerate(eachrow(xtest))]

l1   = DenseLayer(linearf,2,3,w=ones(3,2), wb=zeros(3))
l2   = DenseLayer(linearf,3,1, w=ones(1,3), wb=zeros(1))
mynn = buildNetwork([l1,l2],squaredCost,name="Feed-forward Neural Network Model 1")
train!(mynn,xtrain,ytrain,maxepochs=1000,η=0.01,rshuffle=false,nMsgs=0)
avgLoss = losses(mynn,xtest,ytest)
@test  avgLoss ≈ 0.0032018998005211886
expectedOutput = [0.4676699631752518,0.3448383593117405,0.4500863419692639,9.908883999376018]
predicted = dropdims(predictSet(mynn,xtest),dims=2)
@test any(isapprox(expectedOutput,predicted))


#for (i,r) in enumerate(eachrow(xtest))
#  println("x: $r ŷ: $(predict(mynn,r)[1]) y: $(ytest[i])")
#end

#=
# Challenging dataset with nonlinear relationship:
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]^2+0.2*x[2]+0.3)*rand(0.95:0.001:1.05) for x in eachrow(xtrain)]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]^2+0.2*x[2]+0.3)*rand(0.95:0.001:1.05) for x in eachrow(xtest)]



# ==================================
# Individual components debugging stuff
# ==================================
l1 = FullyConnectedLayer(relu,2,3,w=[1 2; -1 -2; 3 -3],wb=[1,-1,0],df=drelu)
l2 = NoBiasLayer(linearf,3,2,w=[1 2 3; -1 -2 -3],df=dlinearf)
X = [3,1]
Y = [10,0]
o1 = forward(l1,X)
o2 = forward(l2,o1)
ϵ = squaredCost(o2,Y)
d€_do2 = dSquaredCost(o2,Y)
d€_do1 = backward(l2,d€_o1,do2)
d€_dX = backward(l1,X,d€_do1)
l1w = getParams(l1)
l2w = getParams(l2)
l2dw = getGradient(l2,o1,d€_do2)
l1dw = getGradient(l1,X,d€_do1)
setParams!(l1,l1w)
setParams!(l2,l2w)
mynn = buildNetwork([l1,l2],squaredCost,dcf=dSquaredCost)
predict(mynn,X)
ϵ2 = loss(mynn,X,Y)
=#
