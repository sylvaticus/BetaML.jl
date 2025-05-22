using Test
using DelimitedFiles, LinearAlgebra, Statistics #, MLDatasets

#using StableRNGs
#rng = StableRNG(123)
using BetaML

import BetaML.Nn: buildNetwork, forward, loss, backward, train!, get_nparams, _get_n_layers_weights, random_init
import BetaML.Nn: ConvLayer, ReshaperLayer # todo: to export it and remove this when completed

TESTRNG = FIXEDRNG # This could change...
#TESTRNG = StableRNG(123)

println("*** Testing Neural Network...")


# ==================================
# New test
# ==================================
println("Testing Learnable structure...")


L1a = Learnable(([1.0 2; 3 4.0], [1.1,2.2], Float64[], [1.1,2.2], [1.1]))
L1b = Learnable(([1.0 2; 3 4.0], [1.1,2.2], Float64[], [1.1,2.2], [1.1]))
L1c = Learnable(([1.0 2; 3 4.0], [1.1,2.2], Float64[], [1.1,2.2], [1.1]))
foo = (((((sum(L1a,L1b,L1c) - (L1a + L1b + L1c)) + 10) - 4) * 2 ) / 3) *
      (((((sum(L1a,L1b,L1c) - (L1a + L1b + L1c)) + 10) - 4) * 2 ) / 3)

@test foo.data[1][2] == 16.0 && foo.data[5][1] == 16.00
@test (L1a -1).data[1] ≈ (-1 * (1 - L1a)).data[1]
@test (2 / (L1a / 2)).data[2] ≈ (4/L1a).data[2]
@test sqrt(L1a).data[1][2,2] == 2.0

l1_1   = DenseLayer(2,3, rng=copy(TESTRNG),f=identity,w=ones(3,2),wb=[10,10,10])
l1_2   = ConvLayer((14,8),(6,3),3,2,stride=(6,3))
l1_3   = ReplicatorLayer(2)
l1     = GroupedLayer([l1_1,l1_2,l1_3])
p12    = deepcopy(get_params(l1_2))
BetaML.Nn.random_init!(l1;rng=deepcopy(TESTRNG))
@test !(p12.data[1] ≈ get_params(l1_2).data[1])


# ==================================
# New test
# ==================================
println("Testing basic NN behaviour...")
x = [0.1,1]
y = [1,0]
l1   = DenseNoBiasLayer(2,2,w=[2 1;1 1],f=identity,rng=copy(TESTRNG))
l2   = VectorFunctionLayer(2,f=softmax)
mynn = buildNetwork([l1,l2],squared_cost,name="Simple Multinomial logistic regression")
o1 = forward(l1,x)
@test o1 == [1.2,1.1]
#@code_warntype forward(l1,x)
o2 = forward(l2,o1)
@test o2 ≈ [0.5249791874789399, 0.47502081252106] ≈ softmax([1.2,1.1])
orig = predict(mynn,x')[1,:]
@test orig == o2
#@code_warntype Nn.predict(mynn,x')[1,:]
ϵ = squared_cost(y,o2)
#@code_warntype  squared_cost(o2,y)
lossOrig = loss(mynn,x',y')
@test ϵ == lossOrig
#@code_warntype  loss(mynn,x',y')
dϵ_do2 = dsquared_cost(y,o2)
@test dϵ_do2 == [-0.4750208125210601,0.47502081252106]
#@code_warntype dsquared_cost(o2,y)
dϵ_do1 = backward(l2,o1,dϵ_do2) # here takes long as needs Zygote (because Vector Function layer has dfw that stil luse zygote)
@test dϵ_do1 ≈ [-0.23691761847142412, 0.23691761847142412]
#@code_warntype backward(l2,o1,dϵ_do2)
dϵ_dX = backward(l1,x,dϵ_do1)
@test dϵ_dX ≈ [-0.23691761847142412, 0.0]
#@code_warntype backward(l1,x,dϵ_do1)
l1w = get_params(l1)
#@code_warntype
l2w = get_params(l2)
#@code_warntype
w = get_params(mynn)
#@code_warntype get_params(mynn)
#typeof(w)2-element Vector{Float64}:
origW = deepcopy(w)
l2dw = get_gradient(l2,o1,dϵ_do2)
@test length(l2dw.data) == 0
#@code_warntype get_gradient(l2,o1,dϵ_do2)
l1dw = get_gradient(l1,x,dϵ_do1)
@test l1dw.data[1] ≈ [-0.023691761847142414 -0.23691761847142412; 0.023691761847142414 0.23691761847142412]
#@code_warntype
dw = get_gradient(mynn,x,y)
#@code_warntype get_gradient(mynn,x,y)
y_deltax1 = predict(mynn,[x[1]+0.001 x[2]])[1,:]
#@code_warntype Nn.predict(mynn,[x[1]+0.001 x[2]])[1,:]
lossDeltax1 = loss(mynn,[x[1]+0.001 x[2]],y')
#@code_warntype
deltaloss = dot(dϵ_dX,[0.001,0])
#@code_warntype
@test isapprox(lossDeltax1-lossOrig,deltaloss,atol=0.0000001)
l1wNew = l1w
l1wNew.data[1][1,1] += 0.001
set_params!(l1,l1wNew)
lossDelPar = loss(mynn,x',y')
#@code_warntype
deltaLossPar = 0.001*l1dw.data[1][1,1]
lossDelPar - lossOrig
@test isapprox(lossDelPar - lossOrig,deltaLossPar,atol=0.00000001)
η = 0.01
#w = gradientDescentSingleUpdate(w,dw,η)
#w = w - dw * η
w = w - dw * η
#@code_warntype gradSub.(w, gradMul.(dw,η))
#@code_warntype
set_params!(mynn,w)
loss2 = loss(mynn,x',y')
#@code_warntype
@test loss2 < lossOrig
for i in 1:10000
    local w  = get_params(mynn)
    local dw = get_gradient(mynn,x,y)
    w  = w - dw * η
    set_params!(mynn,w)
end
lossFinal = loss(mynn,x',y')
@test predict(mynn,x')[1,1]>0.96
set_params!(mynn,origW)
train!(mynn,x',y',epochs=10000,batch_size=1,sequential=true,verbosity=NONE,opt_alg=SGD(η=t->η,λ=1),rng=copy(TESTRNG))
#@code_warntype train!(mynn,x',y',epochs=10000,batch_size=1,sequential=true,verbosity=NONE,opt_alg=SGD(η=t->η,λ=1))
lossTraining = loss(mynn,x',y')
#@code_warntype
@test isapprox(lossFinal,lossTraining,atol=0.00001)

li   = DenseLayer(2,2,w=[2 1;1 1],f=identity,rng=copy(TESTRNG))
@test get_nparams(li) == 6

# Testing ScalarFunctionLayer with no weigths and identity function (aka "Replicator") layer 
l1    = DenseNoBiasLayer(2,2,w=[2 1;1 1],f=identity,rng=copy(TESTRNG))
l1b   = ReplicatorLayer(2) 
l2    = VectorFunctionLayer(2,f=softmax)
mynn2 = buildNetwork([l1,l1b,l2],squared_cost,name="Simple Multinomial logistic regression")
train!(mynn2,x',y',epochs=10000,batch_size=1,sequential=true,verbosity=NONE,opt_alg=SGD(η=t->η,λ=1),rng=copy(TESTRNG))
lossTraining2 = loss(mynn2,x',y')
#@code_warntype
@test isapprox(lossTraining,lossTraining2,atol=0.00001)


# ==================================
# NEW Test
println("Testing regression if it just works with manual derivatives...")
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
xtest = [0.5 0.6; 0.14 0.2; 0.3 0.7; 2.0 4.0]
ytest = [1.1; 0.36; 1.0; 6.0]
l1 = DenseLayer(2,3,w=[1 1; 1 1; 1 1], wb=[0,0,0], f=tanh, df=dtanh,rng=copy(TESTRNG))
l2 = DenseNoBiasLayer(3,2, w=[1 1 1; 1 1 1], f=relu, df=drelu,rng=copy(TESTRNG))
l3 = DenseLayer(2,1, w=[1 1], wb=[0], f=identity,df=didentity,rng=copy(TESTRNG))
mynn = buildNetwork(deepcopy([l1,l2,l3]),squared_cost,name="Feed-forward Neural Network Model 1",dcf=dsquared_cost)
train!(mynn,xtrain,ytrain, opt_alg=SGD(η=t -> 1/(1+t),λ=1), batch_size=1,sequential=true,epochs=100,verbosity=NONE,rng=copy(TESTRNG),onfail="continue") # 
#@benchmark train!(mynn,xtrain,ytrain,batch_size=1,sequential=true,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1))
avgLoss = loss(mynn,xtest,ytest)
@test  avgLoss ≈ 1.599729991966362
expectedŷtest= [0.7360644412052633, 0.7360644412052633, 0.7360644412052633, 2.47093434438514]
ŷtrain = dropdims(predict(mynn,xtrain),dims=2)
ŷtest = dropdims(predict(mynn,xtest),dims=2)
@test any(isapprox(expectedŷtest,ŷtest))

m = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,dloss=dsquared_cost,batch_size=1,shuffle=false,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1),rng=copy(TESTRNG),descr="First test",onfail="continue")
fit!(m,xtrain,ytrain)
ŷtrain2 =  predict(m)
ŷtrain3 =  predict(m,xtrain)
@test ŷtrain ≈ ŷtrain2 ≈ ŷtrain3
ŷtest2 =  predict(m,xtest)
@test ŷtest ≈ ŷtest2 


# With the ADAM optimizer...
l1 = DenseLayer(2,3,w=[1 1; 1 1; 1 1], wb=[0,0,0], f=tanh, df=dtanh,rng=copy(TESTRNG))
l2 = DenseNoBiasLayer(3,2, w=[1 1 1; 1 1 1], f=relu, df=drelu,rng=copy(TESTRNG))
l3 = DenseLayer(2,1, w=[1 1], wb=[0], f=identity,df=didentity,rng=copy(TESTRNG))
mynn = buildNetwork([l1,l2,l3],squared_cost,name="Feed-forward Neural Network with ADAM",dcf=dsquared_cost)
train!(mynn,xtrain,ytrain,batch_size=1,sequential=true,epochs=100,verbosity=NONE,opt_alg=ADAM(),rng=copy(TESTRNG))
avgLoss = loss(mynn,xtest,ytest)
@test  avgLoss ≈ 0.9497779759064725
expectedOutput = [1.7020525792404175, -0.1074729043392682, 1.4998367847079956, 3.3985794704732717]
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

l1   = DenseLayer(2,3,w=ones(3,2), wb=zeros(3),rng=copy(TESTRNG))
l2   = DenseLayer(3,1, w=ones(1,3), wb=zeros(1),rng=copy(TESTRNG))
mynn = buildNetwork([l1,l2],squared_cost,name="Feed-forward Neural Network Model 1")
train!(mynn,xtrain,ytrain,epochs=1000,sequential=true,batch_size=1,verbosity=NONE,opt_alg=SGD(η=t->0.01,λ=1),rng=copy(TESTRNG))
avgLoss = loss(mynn,xtest,ytest)
@test  avgLoss ≈ 0.0032018998005211886
ŷtestExpected = [0.4676699631752518,0.3448383593117405,0.4500863419692639,9.908883999376018]
ŷtrain = dropdims(predict(mynn,xtrain),dims=2)
ŷtest = dropdims(predict(mynn,xtest),dims=2)
@test any(isapprox(ŷtest,ŷtestExpected))
mreTrain = relative_mean_error(ytrain,ŷtrain,normrec=true)
@test mreTrain <= 0.06
mreTest  = relative_mean_error(ytest,ŷtest,normrec=true)
@test mreTest <= 0.05

m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
fit!(m,xtrain,ytrain)
ŷtrain2 =  predict(m)
mreTrain = relative_mean_error(ytrain,ŷtrain,normrec=true)
@test mreTrain <= 0.06



#predicted = dropdims(Nn.predict(mynn,xtrain),dims=2)
#ytrain

# ==================================
# NEW TEST
# ==================================
println("Going through Multinomial logistic regression (using softmax)...")
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
y_oh = fit!(OneHotEncoder(),y)

ntrain = Int64(round(size(x,1)*0.8))
xtrain = x[1:ntrain,:]
ytrain = y[1:ntrain]
ytrain_oh = y_oh[1:ntrain,:]
xtest = x[ntrain+1:end,:]
ytest = y[ntrain+1:end]

xtrain = fit!(Scaler(),xtrain)
#pcaOut = pca(xtrain,error=0.01)
#(xtrain,P) = pcaOut.X, pcaOut.P
xtest = fit!(Scaler(),xtest)
#xtest = xtest*P

l1   = DenseLayer(4,10, w=ones(10,4), wb=zeros(10),f=celu,rng=copy(TESTRNG))
l2   = DenseLayer(10,3, w=ones(3,10), wb=zeros(3),rng=copy(TESTRNG))
l3   = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3],squared_cost,name="Multinomial logistic regression Model Sepal")
train!(mynn,xtrain,ytrain_oh,epochs=254,batch_size=8,sequential=true,verbosity=NONE,opt_alg=SGD(η=t->0.001,λ=1),rng=copy(TESTRNG))
ŷtrain = predict(mynn,xtrain)
ŷtest  = predict(mynn,xtest)
trainAccuracy = accuracy(ytrain,ŷtrain,tol=1)
testAccuracy  = accuracy(ytest,ŷtest,tol=1)
@test testAccuracy >= 0.8 # set to random initialisation/training to have much better accuracy

# With ADAM
l1   = DenseLayer(4,10, w=ones(10,4), wb=zeros(10),f=celu,rng=copy(TESTRNG))
l2   = DenseLayer(10,3, w=ones(3,10), wb=zeros(3),rng=copy(TESTRNG),df=nothing) # testing AD
l3   = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork(deepcopy([l1,l2,l3]),squared_cost,name="Multinomial logistic regression Model Sepal",dcf=nothing)
train!(mynn,xtrain,ytrain_oh,epochs=10,batch_size=8,sequential=true,verbosity=NONE,opt_alg=ADAM(η=t -> 1/(1+t), λ=0.5),rng=copy(TESTRNG))
ŷtrain = predict(mynn,xtrain)
ŷtest  = predict(mynn,xtest)
trainAccuracy = accuracy(ytrain,ŷtrain,tol=1)
testAccuracy  = accuracy(ytest,ŷtest,tol=1)
@test testAccuracy >= 1

m = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,dloss=nothing,batch_size=8,shuffle=false,epochs=10,verbosity=NONE,opt_alg=ADAM(η=t -> 1/(1+t), λ=0.5),rng=copy(TESTRNG),descr="Iris classification")
fit!(m,xtrain,ytrain_oh)
ŷtrain2 =  predict(m)
@test ŷtrain ≈ ŷtrain2
reset!(m)
fit!(m,xtrain,ytrain_oh)
ŷtrain3 =  predict(m)
@test ŷtrain ≈ ŷtrain2 ≈ ŷtrain3
reset!(m)
m.hpar.epochs = 5
fit!(m,xtrain,ytrain_oh)
#fit!(m,xtrain,ytrain_oh)
ŷtrain4 =  predict(m)
acc = accuracy(ytrain,ŷtrain4,tol=1)
@test acc >= 0.95

m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
fit!(m,xtrain,ytrain_oh)
ŷtrain5 = predict(m)
acc = accuracy(ytrain,ŷtrain5,tol=1, rng=copy(TESTRNG))
@test acc >= 0.78 # 0.9

# ------------------------------------------------------------------------------
# Testing GroupedLayer
println("Testing GroupedLayer and ReplicatorLayer...")
X      = Float64.(collect(transpose(reshape(1:3*7,7,3))))
Y      = Float64.(collect(1:3))

l1_1   = DenseLayer(2,3, rng=copy(TESTRNG),f=identity,w=ones(3,2),wb=[10,10,10])
l1_2   = DenseNoBiasLayer(3,3, rng=copy(TESTRNG),f=identity, w=ones(3,3))
l1_3   = ReplicatorLayer(2)
l1     = GroupedLayer([l1_1,l1_2,l1_3])
l1bis  = GroupedLayer([l1_3,l1_1,l1_2])
l2     = DenseLayer(8,1,f=identity,w=ones(1,8),rng=copy(TESTRNG))
o1     = forward(l1,X[1,:])
o1bis  = forward(l1bis,X[1,:])
@test o1 == [13,13,13,12,12,12,6,7]
@test o1bis == [1,2,17,17,17,18,18,18]
o1_1   = forward(l1_1,X[1,1:2])
@test typeof(o1) == typeof(o1_1)

dϵ_dI    = backward(l1,X[1,:], o1 ./ 10)
dϵ_dIbis = backward(l1bis,X[1,:], o1bis ./ 10)
@test dϵ_dI    ≈ [3.9, 3.9,3.6, 3.6, 3.6,0.6, 0.7]
@test dϵ_dIbis ≈ [0.1,0.2,5.1,5.1,5.4,5.4,5.4]

dϵ_dI_1_1 =  backward(l1_1,X[1,1:2], o1_1 ./ 10)
@test typeof(dϵ_dI) == typeof(dϵ_dI_1_1)

g1    = get_gradient(l1,X[1,:], o1 ./ 10)
g1bis = get_gradient(l1bis,X[1,:], o1bis ./ 10)
g1_1  = get_gradient(l1_1,X[1,1:2], o1_1 ./ 10)
@test all(g1.data .≈ ([1.3 2.6; 1.3 2.6; 1.3 2.6], [1.3, 1.3, 1.3], [3.6 4.8 6.0; 3.6 4.8 6.0; 3.6 4.8 6.0]))
@test all(g1bis.data .≈ ([5.1 6.8; 5.1 6.8; 5.1 6.8], [1.7, 1.7, 1.7], [9.0 10.8 12.6; 9.0 10.8 12.6; 9.0 10.8 12.6]))
@test typeof(g1.data[1:2]) == typeof(g1_1.data)

p      = get_params(l1)
pbis   = get_params(l1bis)

set_params!(l1,p)
set_params!(l1bis,pbis)
o1_after        = forward(l1,X[1,:])
o1bis_after     = forward(l1bis,X[1,:])
@test o1_after  == o1
@test o1bis_after  == o1bis
@test size(l1) == ((7,),(8,))

treenn =  buildNetwork([l1,l2],squared_cost)

predict(treenn,X)
loss(treenn,X,Y)
get_gradient(treenn,X[1,:],Y[1,:])
train!(treenn,X,Y,epochs=5000,rng=copy(TESTRNG))

Ŷ = predict(treenn,X)
rme = relative_mean_error(Y,Ŷ) 
@test rme = relative_mean_error(Y,Ŷ)<= 0.3

#=
if "all" in ARGS
    # ==================================
    # NEW TEST
    # ==================================
    println("Testing colvolution layer with MINST data...")
    train_x, train_y = MNIST.traindata()
    test_x,  test_y  = MNIST.testdata()

    test = train_x[:,:,1]

    eltype(test)

    test .+ 1
end
=#


# ==================================
# NEW Test
if VERSION >= v"1.6"
    println("Testing VectorFunctionLayer with pool1d function...")
    println("Attention: this test requires at least Julia 1.6")

    x        = rand(copy(TESTRNG),300,5)
    y        = [norm(r[1:3])+2*norm(r[4:5],2) for r in eachrow(x) ]
    (N,D)    = size(x)
    l1       = DenseLayer(D,8, f=relu,rng=copy(TESTRNG))
    l2       = VectorFunctionLayer(size(l1)[2][1],f=(x->pool1d(x,2,f=mean)))
    l3       = DenseLayer(size(l2)[2][1],1,f=relu, rng=copy(TESTRNG))
    mynn     = buildNetwork([l1,l2,l3],squared_cost,name="Regression with a pooled layer")
    train!(mynn,x,y,epochs=50,verbosity=NONE,rng=copy(TESTRNG))
    ŷ        = predict(mynn,x)
    rmeTrain = relative_mean_error(y,ŷ,normrec=false)
    @test rmeTrain  < 0.14
end

# ==================================
# NEW TEST
println("Testing ConvLayer....")
d2convl = ConvLayer((14,8),(6,3),3,2,stride=(6,3))
@test d2convl.padding_start == [2,1]
@test d2convl.padding_end   == [2,0]
@test size(d2convl) == ((14,8,3),(3,3,2))


d2convl = ConvLayer((14,8),(6,3),3,2,stride=3)
@test d2convl.padding_start == [2,1]
@test d2convl.padding_end   == [2,0]

@test size(d2convl) == ((14, 8, 3), (5, 3, 2))
d2convl = ConvLayer((14,8),(6,3),3,2,stride=3, padding=((2,1),(1,0)))
@test size(d2convl) == ((14, 8, 3), (4, 3, 2))

d2convl = ConvLayer((13,8),(6,3),3,2,stride=3)
@test d2convl.padding_start == [3,1]
@test d2convl.padding_end   == [2,0]
@test size(d2convl) == ((13, 8, 3), (5, 3, 2))

d2convl = ConvLayer((7,5),(4,3),3,2,stride=2)
@test d2convl.input_size == [7,5,3]
@test d2convl.ndims == 2
@test size(d2convl.weight) == (4,3,3,2) 
@test d2convl.stride == [2,2]

d2conv = ConvLayer((4,4),(2,2),3,2,kernel_init=reshape(1:24,(2,2,3,2)),bias_init=[1,1])
x = ones(4,4,3)
preprocess!(d2conv)
y = forward(d2conv,x)
# The syntax for tensor hard coded in this way wants Julia >= 1.7
if VERSION >= v"1.7"
  @test y[1,1,1] == dot([0 0; 0 1;;; 0 0; 0 1;;; 0 0; 0 1 ],selectdim(d2conv.weight,4,1)) + d2conv.bias[1] == 25
  @test y[2,3,1] == dot([1 1; 1 1;;; 1 1; 1 1;;; 1 1; 1 1 ],selectdim(d2conv.weight,4,1)) + d2conv.bias[1] == 79
end

de_dy = y ./100
de_dw = get_gradient(d2conv,x,de_dy)
de_dx = backward(d2conv,x,de_dy)

d1conv  = ConvLayer(8,3,1,1,stride=3,kernel_init=reshape(1:3,(3,1,1)),bias_init=[10,])
d1conv2 = ConvLayer(8,3,1,1,stride=2,kernel_init=reshape(1:3,(3,1,1)),bias_init=[10,])
x = collect(1:8)
preprocess!(d1conv)
preprocess!(d1conv2)
#@btime preprocess!(d1conv)
y = forward(d1conv,x)
@test y[1,1] == dot([0,1,2],[1,2,3]) + 10
@test y[3,1] == dot([6,7,8],[1,2,3]) + 10


# The syntax for tensor hard coded in this way wants Julia >= 1.7
if VERSION >= v"1.7"
    de_dy = [1.0; 2.0; 3.0;;]
    de_dw = get_gradient(d1conv,x,de_dy)
    de_dx = backward(d1conv,x,de_dy)
    @test de_dw.data[1] == [24.0; 30.0; 36.0;;;]
    @test de_dw.data[2] == [6]
    @test de_dx == [2.0; 3; 2; 4; 6; 3; 6; 9;;]
    # using d1conv2 (some overlapping filters for the same x)
    de_dy = [1.0; 2.0; 3.0; 4.0;;]
    de_dw = get_gradient(d1conv2,x,de_dy)
    de_dx = backward(d1conv2,x,de_dy)
    @test de_dw.data[1] == [40.0; 50.0; 60.0;;;]
    @test de_dw.data[2] == [10.0]
    @test de_dx == [2.0;5;4;9;6;13;8;12;;]
end

#=
x = reshape(1:12*12*3,12,12,3)
l = ConvLayer((12,12,3),(4,4),5)
preprocess!(l)
@btime preprocess!(l)
y = forward(l,x)
@btime forward($l,$x)
de_dy = y ./ 100
de_dw = get_gradient(l,x,de_dy)
@btime get_gradient($l,$x,$de_dy)
de_dx = backward(l,x,de_dy)
@btime backward($l,$x,$de_dy)
@profile get_gradient(l,x,de_dy)
=#


x     = collect(1:12)
l1    = ReshaperLayer((12,1),(3,2,2))
l2    = ConvLayer((3,2),(2,2),2,1,kernel_init=ones(2,2,2,1),bias_init=[1])
l3    = ConvLayer(size(l2)[2],(2,2),1,kernel_init=ones(2,2,1,1),bias_init=[1]) # alternative constructor
l4    = ReshaperLayer((3,2,1))
preprocess!.([l2,l3])
l1y   = forward(l1,x)
l2y   = forward(l2,l1y)
l3y   = forward(l3,l2y)
l4y   = forward(l4,l3y)
truey =  [8.0, 31.0, 43.0, 33.0, 101.0, 149.0]

mynn  = buildNetwork([l1,l2,l3,l4],squared_cost)
ŷ     = predict(mynn,x')
e     = loss(mynn,x',truey')
@test e ≈ 4


#x        = rand(copy(TESTRNG),100,3*3*2)
x        = convert(Matrix{Float32},reshape(1:100*3*3*2,100,3*3*2) ./ 100)
y        = convert(Vector{Float32},[norm(r[1:9])+2*norm(r[10:18],2) for r in eachrow(x) ])
(N,D)    = size(x)
l1       = ReshaperLayer((D,1),(3,3,2))
l2       = ConvLayer((3,3),(2,2),2,3,rng=copy(TESTRNG),kernel_eltype=Float32)
l3       = ConvLayer(size(l2)[2],(2,2),8,rng=copy(TESTRNG),kernel_eltype=Float32)
l4       = ReshaperLayer(size(l3)[2])
l5       = DenseLayer(size(l4)[2][1],1,f=relu, w_eltype=Float32, rng=copy(TESTRNG))
layers   = [l1,l2,l3,l4,l5]
mynn     = buildNetwork(layers,squared_cost,name="Regression with a convolutional layer")
preprocess!(mynn)
x1_hat = predict(mynn,x[1,:]')
@test typeof(x1_hat) == Matrix{Float32}
train!(mynn,x,y,epochs=60,verbosity=NONE,rng=copy(TESTRNG))
ŷ        = predict(mynn,x)
rmeTrain = relative_mean_error(y,ŷ,normrec=false)
@test rmeTrain  < 0.01

#using BenchmarkTools
#@btime train!($mynn,$x,$y,epochs=60,verbosity=NONE,rng=copy($TESTRNG))
#240.604 ms (1056544 allocations: 107.66 MiB)
#314.504 ms (774762 allocations: 99.39 MiB)


# ==================================
# NEW TEST
println("Testing PoolingLayer....")
d2pooll = PoolingLayer((14,8),(6,3),3)
@test d2pooll.padding_start == [2,1]
@test d2pooll.padding_end   == [2,0]

@test size(d2pooll) == ((14, 8, 3), (3, 3, 3))
d2pooll = PoolingLayer((14,8),(6,3),3,stride=3, padding=((2,1),(1,0)))
@test size(d2pooll) == ((14, 8, 3), (4, 3, 3))

d2pooll = PoolingLayer((13,8),(6,3),3,stride=3)
@test d2pooll.padding_start == [3,1]
@test d2pooll.padding_end   == [2,0]
@test size(d2pooll) == ((13, 8, 3), (5, 3, 3))

d2pooll = PoolingLayer((7,5),(4,3),3,stride=2)
@test d2pooll.input_size == [7,5,3]
@test d2pooll.ndims == 2
@test d2pooll.kernel_size == [4,3,3,3] 
@test d2pooll.stride == [2,2]

d2pool = PoolingLayer((4,4),(2,2),3,f=mean)
x = reshape(1:(4*4*3),4,4,3)
preprocess!(d2pool)
@test d2pool.y_to_x_ids[2,2,2] == [(3,3,2),(4,3,2),(3,4,2),(4,4,2)]
y = forward(d2pool,x)
@test y[1,2,1] == 11.5

de_dy = y ./10
de_dw = get_gradient(d2pool,x,de_dy)
de_dx = backward(d2pool,x,de_dy)
@test de_dx[2,3,3] == (de_dy ./ 4)[1,2,3]

# full example - x as classical tabular data
x        = reshape(1:100*6*6*2,100,6*6*2) ./ 100
y        = [norm(r[1:36])+2*norm(r[37:72],2) for r in eachrow(x) ]
(N,D)    = size(x)
l1       = ReshaperLayer((D,1),(6,6,2))
l2       = ConvLayer((6,6),(2,2),2,4,rng=copy(TESTRNG))
l3       = PoolingLayer((6,6,4),(2,2))
l4       = ConvLayer(size(l3)[2],(2,2),8,rng=copy(TESTRNG))
l5       = ReshaperLayer(size(l4)[2])
l6       = DenseLayer(size(l5)[2][1],1,f=identity, rng=copy(TESTRNG))
layers   = [l1,l2,l3,l4,l5,l6]
mynn     = buildNetwork(layers,squared_cost,name="Regression with a convolutional layer")
preprocess!(mynn)
dummyx   = x[1,:]
nnout    = BetaML.predict(mynn,dummyx')
l1y = forward(l1,dummyx)
l2y = forward(l2,l1y)
l3y = forward(l3,l2y)
l4y = forward(l4,l3y)
l5y = forward(l5,l4y)
l6y = forward(l6,l5y)
@test l6y[1] == nnout[1,1]
train!(mynn,x,y,epochs=40,verbosity=NONE,rng=copy(TESTRNG))
ŷ        = BetaML.predict(mynn,x)
rmeTrain = relative_mean_error(y,ŷ,normrec=false)
@test rmeTrain  < 0.1

# ==================================
# NEW TEST
println("Testing onfail strategies....")

xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
l1 = DenseLayer(2,3,w=[1 1; 1 1; 1 1], wb=[0,0,0], f=tanh, df=dtanh,rng=copy(TESTRNG))
l2 = DenseNoBiasLayer(3,2, w=[1 1 1; 1 1 1], f=relu, df=drelu,rng=copy(TESTRNG))
l3 = DenseLayer(2,1, w=[1 1], wb=[0], f=identity,df=didentity,rng=copy(TESTRNG))


m1 = NeuralNetworkEstimator(layers=deepcopy([l1,l2,l3]),loss=squared_cost,dloss=dsquared_cost,batch_size=1,shuffle=false,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1),rng=copy(TESTRNG),descr="First test",onfail="error")
m2 = NeuralNetworkEstimator(layers=deepcopy([l1,l2,l3]),loss=squared_cost,dloss=dsquared_cost,batch_size=1,shuffle=false,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1),rng=copy(TESTRNG),descr="First test",onfail="continue")
m3 = NeuralNetworkEstimator(layers=deepcopy([l1,l2,l3]),loss=squared_cost,dloss=dsquared_cost,batch_size=1,shuffle=false,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1),rng=copy(TESTRNG),descr="First test",onfail="stop")
m4 = NeuralNetworkEstimator(layers=deepcopy([l1,l2,l3]),loss=squared_cost,dloss=dsquared_cost,batch_size=1,shuffle=false,epochs=100,verbosity=NONE,opt_alg=SGD(η=t -> 1/(1+t),λ=1),rng=copy(TESTRNG),descr="First test",onfail="tryagain")

@test_throws ErrorException("Fitting the model on the data failed. Loss is not decreasing.") fit!(m1,xtrain,ytrain)
@test relative_mean_error(ytrain,fit!(m2,xtrain,ytrain)) >= 0.4
@test relative_mean_error(ytrain,fit!(m3,xtrain,ytrain)) >= 0.4
@test relative_mean_error(ytrain,fit!(m4,xtrain,ytrain)) <= 0.1

a = 1
#l1       = ReshaperLayer((D,1),(6,6,2))
#l2       = ConvLayer((6,6),(2,2),2,4,rng=copy(TESTRNG))
#l3       = PoolingLayer((6,6,4),(2,2))
#l4       = ConvLayer(size(l3)[2],(2,2),8,rng=copy(TESTRNG))
#l5       = ReshaperLayer(size(l4)[2])
#l6       = DenseLayer(size(l5)[2][1],1,f=identity, rng=copy(TESTRNG))
#layers   = [l1,l2,l3,l4,l5,l6]
#mynn     = buildNetwork(layers,squared_cost,name="Regression with a convolutional layer")
#@btime train!(mynn,x,y,epochs=5,verbosity=NONE,rng=copy(TESTRNG))
#ŷ        = BetaML.predict(mynn,x)
#rmeTrain = relative_mean_error(y,ŷ,normrec=false)


#=
# x organised as multidimensional array TODO
x        = reshape(1:100*6*6*2,100,6,6,2) ./ 100
y        = collect(1:100)
(N,D)    = size(x)
l2       = ConvLayer((6,6),(2,2),2,4,rng=copy(TESTRNG))
l3       = PoolingLayer((6,6,4),(2,2))
l4       = ConvLayer(size(l3)[2],(2,2),8,rng=copy(TESTRNG))
l5       = ReshaperLayer(size(l4)[2])
l6       = DenseLayer(size(l5)[2][1],1,f=identity, rng=copy(TESTRNG))
layers   = [l2,l3,l4,l5,l6]
mynn     = buildNetwork(layers,squared_cost,name="Regression with a convolutional layer")
preprocess!(mynn)
dummyx = selectdim(x,1,1)
nnout  = predict(mynn,dummyx)
l1y = forward(l1,dummyx)
l2y = forward(l2,l1y)
l3y = forward(l3,l2y)
l4y = forward(l4,l3y)
l5y = forward(l5,l4y)
l6y = forward(l6,l5y)
@test l6y == nnout
train!(mynn,x,y,epochs=10,verbosity=FULL,rng=copy(TESTRNG))
ŷ        = predict(mynn,x)
rmeTrain = relative_mean_error(y,ŷ,normrec=false)
@test rmeTrain  < 0.01
=#


# ==================================
# NEW TEST
println("Testing MLJ interface for FeedfordwarNN....")
import MLJBase
const Mlj = MLJBase
import StatisticalMeasures
X, y                           = Mlj.@load_boston
model                          = BetaML.Bmlj.NeuralNetworkRegressor(rng=copy(TESTRNG))
regressor                      = Mlj.machine(model, X, y)
(fitresult, cache, report)     = Mlj.fit(model, -1, X, y)
yhat                           = Mlj.predict(model, fitresult, X)
@test relative_mean_error(y,yhat,normrec=true) < 0.2

X, y                           = Mlj.@load_boston
y2d = [y y]
model                          = BetaML.Bmlj.MultitargetNeuralNetworkRegressor(rng=copy(TESTRNG))
regressor                      = Mlj.machine(model, X, y2d)
(fitresult, cache, report)     = Mlj.fit(model, -1, X, y2d)
yhat                           = Mlj.predict(model, fitresult, X)
@test relative_mean_error(y2d,yhat,normrec=true) < 0.2

X, y                           = Mlj.@load_iris
model                          = BetaML.Bmlj.NeuralNetworkClassifier(rng=copy(TESTRNG),epochs=500,batch_size=64)
regressor                      = Mlj.machine(model, X, y)
(fitresult, cache, report)     = Mlj.fit(model, -1, X, y)
yhat                           = Mlj.predict(model, fitresult, X)
#@test Mlj.mean(StatisticalMeasures.LogLoss(tol=1e-4)(yhat, y)) < 0.25
@test sum(Mlj.mode.(yhat) .== y)/length(y) >= 0.98

#=
x = Mlj.matrix(X)
sm = Scaler()  
xs = fit!(sm,x)
m = AutoEncoder() 
x2 = fit!(m,xs)
x̂s = inverse_predict(m,x2)
x̂  = inverse_predict(sm,x̂s)
rme = relative_mean_error(x,x̂)
=#
