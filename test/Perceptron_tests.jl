using Statistics
using Test
using DelimitedFiles
import MLJBase
const Mlj = MLJBase
using StableRNGs
using BetaML

#TESTRNG = FIXEDRNG # This could change...
TESTRNG = StableRNG(123)

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
classes = unique(y)
out      = perceptron(xtrain, ytrain, shuffle=false,nMsgs=0)
ŷtrain    = predict(xtrain,out.θ,out.θ₀,out.classes)
ϵtrain    = error(ytrain, mode(ŷtrain))
ŷtest    = predict(xtest,out.θ,out.θ₀,classes)
outTest  = perceptron(xtrain, ytrain, shuffle=false,nMsgs=0,returnMeanHyperplane=true)
ŷavgtest = predict(xtest,outTest.θ,outTest.θ₀,outTest.classes)
ϵ        = error(ytest, mode(ŷtest))
ϵavg     = error(ytest, mode(ŷavgtest))
@test ϵ    < 0.03
@test ϵavg < 0.2
m      =  PerceptronClassic(shuffle=false, verbosity=NONE, rng=copy(TESTRNG))
fit!(m,xtrain,ytrain)
ŷtrain2 = predict(m) 
ŷtrain3 = predict(m,xtrain)
ϵtrain    = error(ytrain, mode(ŷtrain3))
@test ŷtrain == ŷtrain2 == ŷtrain3

# Test save/load
model_save("test.jld2"; m, m2=m)
models   = model_load("test.jld2")
ŷtrain4  = predict(models["m2"]) 
mb       = model_load("test.jld2","m") 
(mc, md) = model_load("test.jld2","m", "m2") 
ŷtrain5 = predict(mb) 
ŷtrain6 = predict(mc) 
ŷtrain7 = predict(md) 
@test ŷtrain == ŷtrain4 == ŷtrain5 == ŷtrain6 == ŷtrain7

pars = parameters(m)
pars.weigths[1,1] = 10
pars.weigths[2,1] = -10
ŷtrain8 = predict(m,xtrain) 
@test ŷtrain8 != ŷtrain 

hpars = hyperparameters(m)
hpars.epochs = 10
@test m.hpar.epochs == 10

opt = options(m)
opt.descr="Hello"
@test m.opt.descr == "Hello"

println("Testing multiple classes...")

#xtrain = [3 4 5; 5 3 5; 3 7 2; 8 5 3; 4 2 3; 3 2 1; 8 3 4; 3 5 1; 1 9 3; 4 2 1]
xtrain = rand(TESTRNG,100,3)
ytt    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtrain))]
ytrain = [i > median(ytt)*1.1 ? "big" :  i > median(ytt)*0.9 ? "avg" : "small" for i in ytt]
#xtest  = [2 2 3; 3 2 2; 4 1 2; 4 3 2; 3 7 2]
xtest = rand(TESTRNG,20,3)
ytt2   = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtest))]
ytest  = [i > median(ytt2)*1.1 ? "big" :  i > median(ytt2)*0.9 ? "avg" : "small" for i in ytt2]
out    = perceptron(xtrain,  ytrain, shuffle=false,nMsgs=0)
out2   = perceptron(xtrain,ytrain,θ₀=[0.0, 0.0, 0.0],θ=[[0.0, 0.0, 0.0], [0.0,0.0,0.0], [0.0,0.0,0.0]])
@test out == out2
ŷtrain = predict(xtrain,out.θ,out.θ₀,out.classes)
ŷtest  = predict(xtest,out.θ,out.θ₀,out.classes)
ϵtrain = error(ytrain, mode(ŷtrain))
ϵtest  = error(ytest, mode(ŷtest))

@test ϵtrain  < 0.4
@test ϵavg    < 0.4

m      =  PerceptronClassic(shuffle=false, verbosity=NONE, rng=copy(TESTRNG))
fit!(m,xtrain,ytrain)
ŷtrain2 = predict(m) 
ŷtrain3 = predict(m,xtrain)
@test all([ŷtrain[r][k] ≈ ŷtrain2[r][k] ≈ ŷtrain3[r][k] for k in keys(ŷtrain[1]) for r in 1:length(ŷtrain)])

# ==================================
# Test 2: Kernel Perceptron
# ==================================
println("Going through Test2 (Kernel Perceptron)...")


xtrain = [3 4 5; 5 3 5; 3 7 2; 8 5 3; 4 2 3; 3 2 1; 8 3 4; 3 5 1; 1 9 3; 4 2 1]
ytt    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtrain))]
ytrain = [i > median(ytt) ? 1 : -1 for i in ytt]
xtest  = [ 3 7 2; 2 2 3; 3 2 2; 4 1 2; 4 3 2;]
#xtest = xtrain
ytt2    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtest))]
ytest = [i > median(ytt2) ? 1 : -1 for i in ytt2]
#out   = kernelPerceptron(xtrain, ytrain, K=polynomialKernel,rShuffle=true,nMsgs=100)
#ŷtest = predict(xtest,out[1][1],out[1][2],out[1][3], K=polynomialKernel)
out   = kernelPerceptronBinary(xtrain, ytrain, K=radialKernel,shuffle=false,nMsgs=0,α=ones(Int64,length(ytrain)))
# the same: out   = kernelPerceptronBinary(xtrain, ytrain, K=radialKernel,shuffle=false,nMsgs=0)
ŷtest = predict(xtest,out.x,out.y,out.α, K=out.K)
ϵ = error(ytest, ŷtest)
ŷtestExpected = [1,-1,-1,-1,-1]
@test ϵ ≈ 0.2
#@test any(isapprox(ŷtestExpected,ŷtest))
@test any(ŷtestExpected == ŷtest )

# Multiclass..
outMultiClass   = kernelPerceptron(xtrain, ytrain, K=radialKernel,shuffle=false,nMsgs=0)
ŷtest = predict(xtest,outMultiClass.x,outMultiClass.y,outMultiClass.α, outMultiClass.classes,K=outMultiClass.K)
ϵ = error(ytest, mode(ŷtest))
ŷtestExpected = [1,-1,-1,-1,-1]
@test ϵ ≈ 0.2
#@test any(isapprox(ŷtestExpected,ŷtest))
@test any(ŷtestExpected == mode(ŷtest) )


xtrain = rand(TESTRNG,100,3)
ytt    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtrain))]
ytrain = [i > median(ytt)*1.1 ? "big" :  i > median(ytt)*0.9 ? "avg" : "small" for i in ytt]
#xtest  = [2 2 3; 3 2 2; 4 1 2; 4 3 2; 3 7 2]
xtest = rand(TESTRNG,20,3)
ytt2   = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtest))]
ytest  = [i > median(ytt2)*1.1 ? "big" :  i > median(ytt2)*0.9 ? "avg" : "small" for i in ytt2]

out    = kernelPerceptron(xtrain,  ytrain, shuffle=false,nMsgs=0,T=1000)
ŷtrain = predict(xtrain,out.x,out.y,out.α, out.classes,K=out.K)
ŷtest  = predict(xtest,out.x,out.y,out.α, out.classes,K=out.K)
ϵtrain = error(ytrain, mode(ŷtrain))
ϵtest  = error(ytest, mode(ŷtest))

@test ϵtrain  < 0.1
@test ϵtest   < 0.8

m = KernelPerceptron(shuffle=false,verbosity=NONE, rng=copy(TESTRNG))
fit!(m,xtrain,ytrain)
ŷtrain2 = predict(m)
ŷtrain3 = predict(m,xtrain)
@test all([ŷtrain[r][k] ≈ ŷtrain2[r][k] ≈ ŷtrain3[r][k] for k in keys(ŷtrain[1]) for r in 1:length(ŷtrain)])
ŷtest2 = predict(m,xtest)
@test all([ŷtest[r][k] ≈ ŷtest2[r][k] for k in keys(ŷtest[1]) for r in 1:length(ŷtest)])


# ==================================
# Test 3: Pegasos
# ==================================
println("Going through Test3 (Pegasos)...")


perceptronData     = readdlm(joinpath(@__DIR__,"data/binary2DData.csv"),'\t')
x = copy(perceptronData[:,[2,3]])
y = convert(Array{Int64,1},copy(perceptronData[:,1]))
xtrain = x[1:160,:]
ytrain = y[1:160]
xtest  = x[161:end,:]
ytest  = y[161:end]

out   = pegasos(xtrain, ytrain, shuffle=false, nMsgs=0)
ŷtest = predict(xtest,out.θ,out.θ₀,out.classes)
outAvg   = pegasos(xtrain, ytrain, shuffle=false, nMsgs=0, returnMeanHyperplane=true)
ŷavgtest = predict(xtest,outAvg.θ,outAvg.θ₀,outAvg.classes)
ϵ = error(ytest, mode(ŷtest))
ϵavg = error(ytest, mode(ŷavgtest))
@test ϵ ≈ 0.025
@test ϵavg ≈ 0.1

println("Testing pegasos with multiple classes...")

#xtrain = [3 4 5; 5 3 5; 3 7 2; 8 5 3; 4 2 3; 3 2 1; 8 3 4; 3 5 1; 1 9 3; 4 2 1]
xtrain = rand(TESTRNG,100,3)
ytt    = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtrain))]
ytrain = [i > median(ytt)*1.1 ? "big" :  i > median(ytt)*0.9 ? "avg" : "small" for i in ytt]
#xtest  = [2 2 3; 3 2 2; 4 1 2; 4 3 2; 3 7 2]
xtest = rand(TESTRNG,20,3)
ytt2   = [(0.5*x[1]+0.2*x[2]^2+0.3*x[3]+1) for (i,x) in enumerate(eachrow(xtest))]
ytest  =  [i > median(ytt2)*1.1 ? "big" :  i > median(ytt)*0.9 ? "avg" : "small" for i in ytt2]

out    = pegasos(xtrain,  ytrain, shuffle=false,nMsgs=0)
ŷtrain = predict(xtrain,out.θ,out.θ₀,out.classes)
ŷtest  = predict(xtest,out.θ,out.θ₀,out.classes)
ϵtrain = error(ytrain, mode(ŷtrain))
ϵtest  = error(ytest, mode(ŷtest))

@test ϵtrain  <= 0.8 # this relation is not linear, normal error is big
@test ϵtest   <= 0.8

m = Pegasos(shuffle=false,verbosity=NONE, rng=copy(TESTRNG))
fit!(m,xtrain,ytrain)
ŷtrain2 = predict(m)
ŷtrain3 = predict(m,xtrain)
@test all([ŷtrain[r][k] ≈ ŷtrain2[r][k] ≈ ŷtrain3[r][k] for k in keys(ŷtrain[1]) for r in 1:length(ŷtrain)])
ŷtest2 = predict(m,xtest)
@test all([ŷtest[r][k] ≈ ŷtest2[r][k] for k in keys(ŷtest[1]) for r in 1:length(ŷtest)])


# ==================================
# NEW TEST
println("Testing classification of the sepal database using perceptron algorithms...")
iris = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
x = convert(Array{Float64,2}, iris[:,1:4])
y = convert(Array{String,1}, iris[:,5])

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

model = perceptron(xtrain,ytrain)
ŷtrain = predict(xtrain,model.θ,model.θ₀,model.classes)
@test accuracy(mode(ŷtrain),ytrain) >= 0.79
ŷtest = predict(xtest,model.θ,model.θ₀,model.classes)
@test accuracy(mode(ŷtest),ytest)  >= 0.9

model = kernelPerceptron(xtrain,ytrain)
ŷtrain = predict(xtrain,model.x,model.y,model.α,model.classes)
@test accuracy(mode(ŷtrain),ytrain) >= 0.9
ŷtest = predict(xtest,model.x,model.y,model.α,model.classes)
@test accuracy(mode(ŷtest),ytest)  >= 0.9

model = pegasos(xtrain,ytrain)
ŷtrain = predict(xtrain,model.θ,model.θ₀,model.classes)
@test accuracy(mode(ŷtrain),ytrain) >= 0.64
ŷtest = predict(xtest,model.θ,model.θ₀,model.classes)
@test accuracy(mode(ŷtest),ytest)  >= 0.76

# ==================================
# NEW TEST
println("Testing MLJ interface for Perceptron models....")

X, y                           = Mlj.@load_iris

model                          = PerceptronClassifier(rng=copy(TESTRNG))
regressor                      = Mlj.machine(model, X, y)
(fitresult, cache, report)     = Mlj.fit(model, 0, X, y)
yhat                           = Mlj.predict(model, fitresult, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat, y)) < 3.1

model                          = KernelPerceptronClassifier(rng=copy(TESTRNG))
regressor                      = Mlj.machine(model, X, y)
(fitresult, cache, report)     = Mlj.fit(model, 0, X, y)
yhat                           = Mlj.predict(model, fitresult, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat, y)) < 0.5

model                          = PegasosClassifier(rng=copy(TESTRNG))
regressor                      = Mlj.machine(model, X, y)
(fitresult, cache, report)     = Mlj.fit(model, 0, X, y)
yhat                           = Mlj.predict(model, fitresult, X)
@test Mlj.mean(Mlj.LogLoss(tol=1e-4)(yhat, y)) < 1.3
