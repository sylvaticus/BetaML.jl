using Statistics
using Test
using DelimitedFiles

import Random:seed!
seed!(1234)

using BetaML.Perceptron

println("*** Testing Perceptron algorithms...")

# ==================================
# TEST 1: Normal perceptron
println("Going through Test1 (normal Perceptron)...")

perceptronData     = readdlm(joinpath(@__DIR__,"data/binary2DData.csv"),'\t')
x = copy(perceptronData[:,[2,3]])
y = convert(Array{Int64,1},copy(perceptronData[:,1]))
ntrain = Int64(round(length(y)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

out   = perceptron(xtrain, ytrain, rShuffle=false,nMsgs=0)
ŷtest = Perceptron.predict(xtest,out.θ,out.θ₀)
ŷavgtest = Perceptron.predict(xtest,out.avgθ,out.avgθ₀)
ϵ = error(ytest, ŷtest)
ϵavg = error(ytest, ŷavgtest)



# ==================================
# Test 2: Kernel Perceptron
# ==================================
println("Going through Test2 (Kernel Perceptron)...")


xtrain = [3 4 5; 5 3 5; 3 7 2; 8 5 3; 4 2 3; 3 2 1; 8 3 4; 3 5 1; 1 9 3; 4 2 1]
ytt    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtrain))]
ytrain = [i > median(ytt) ? 1 : -1 for i in ytt]
xtest  = [2 2 3; 3 2 2; 4 1 2; 4 3 2; 3 7 2]
#xtest = xtrain
ytt2    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtest))]
ytest = [i > median(ytt2) ? 1 : -1 for i in ytt2]
#out   = kernelPerceptron(xtrain, ytrain, K=polynomialKernel,rShuffle=true,nMsgs=100)
#ŷtest = predict(xtest,out[1][1],out[1][2],out[1][3], K=polynomialKernel)
out   = kernelPerceptron(xtrain, ytrain, K=radialKernel,rShuffle=false,nMsgs=0,α=ones(length(ytrain)))
ŷtest = Perceptron.predict(xtest,out.x,out.y,out.α, K=radialKernel)
ϵ = error(ytest, ŷtest)
ŷtestExpected = [-1,-1,-1,-1,1]
@test ϵ ≈ 0.2
#@test any(isapprox(ŷtestExpected,ŷtest))
@test any(ŷtestExpected == ŷtest )


# ==================================
# Test 3: Pegasus
# ==================================
println("Going through Test3 (Pegasus)...")


perceptronData     = readdlm(joinpath(@__DIR__,"data/binary2DData.csv"),'\t')
x = copy(perceptronData[:,[2,3]])
y = convert(Array{Int64,1},copy(perceptronData[:,1]))
xtrain = x[1:160,:]
ytrain = y[1:160]
xtest = x[161:end,:]
ytest = y[161:end]

out   = pegasus(xtrain, ytrain, rShuffle=false,nMsgs=0)
ŷtest = Perceptron.predict(xtest,out.θ,out.θ₀)
ŷavgtest = Perceptron.predict(xtest,out.avgθ,out.avgθ₀)
ϵ = error(ytest, ŷtest)
ϵavg = error(ytest, ŷavgtest)
