# ## Library and data loading
using Dates #src
println(now(), " ", "*** Start iris clustering tutorial..." )  #src

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

using BetaML
using Random
Random.seed!(123);
using DelimitedFiles
using Statistics
using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using Flux: @epochs
using MLDatasets # For loading the training data
#using Images, FileIO, ImageTransformations # For loading the actual images

TESTRNG = FIXEDRNG # This could change...

x_train, y_train = MLDatasets.MNIST.traindata()
x_train          = permutedims(x_train,(3,2,1))
x_train          = convert(Array{Float32,3},x_train)
x_train          = reshape(x_train,size(x_train,1),size(x_train,2)*size(x_train,3))
ohm              = OneHotEncoder()
y_train_oh       = fit!(ohm,y_train)

x_test, y_test = MLDatasets.MNIST.testdata()
x_test          = permutedims(x_test,(3,2,1))
x_test          = convert(Array{Float32,3},x_test)
x_test          = reshape(x_test,size(x_test,1),size(x_test,2)*size(x_test,3))
y_test_oh       = predict(ohm,y_test)


#=
(N,D)    = size(x_train)
l1       = ReshaperLayer((D,1),(28,28,1))
l2       = ConvLayer((28,28),(5,5),1,8,padding=2,stride=2,rng=copy(TESTRNG))
size(l2)
l3       = ConvLayer(size(l2)[2],(3,3),16,padding=2,stride=2,rng=copy(TESTRNG))
size(l3)
l4       = ConvLayer(size(l3)[2],(3,3),32,padding=1,stride=2,rng=copy(TESTRNG))
size(l4)
l5       = ConvLayer(size(l4)[2],(3,3),32,padding=1,stride=2,rng=copy(TESTRNG))
size(l5)
l6       = PoolingLayer(size(l5)[2],(2,2),f=mean)
size(l6)
l7       = ReshaperLayer(size(l6)[2])
size(l7)
l8       = DenseLayer(size(l7)[2][1],10,f=BetaML.relu, rng=copy(TESTRNG))
size(l8)
l9       = VectorFunctionLayer(size(l8)[2][1],f=BetaML.softmax)
size(l9)
layers   = [l1,l2,l3,l4,l5,l6,l7,l8,l9]

m = NeuralNetworkEstimator(layers=layers,loss=squared_cost,verbosity=NONE,batch_size=64,epochs=1)

(x_debug,x_other),(y_debug_oh,y_other_oh)  = partition([x_train,y_train_oh],[0.005,0.995])

ŷ = fit!(m,x_debug,y_debug_oh)

mode(ŷ)

accuracy(predict(y_debug_oh),mode(ŷ))
hcat(y_train[1:100],mode(ŷ))
=#


(N,D)    = size(x_train)
l1       = ReshaperLayer((D,1),(28,28,1))
size(l1)
l2       = ConvLayer(size(l1)[2],(5,5),8,stride=2,rng=copy(TESTRNG))
size(l2)
l3       = PoolingLayer(size(l2)[2],(2,2))
size(l3)
l4       = ConvLayer(size(l3)[2],(3,3),16,stride=2,rng=copy(TESTRNG))
size(l4)
l5       = PoolingLayer(size(l4)[2],(2,2))
size(l5)
l6       = ReshaperLayer(size(l5)[2])
size(l6)
l7       = DenseLayer(size(l6)[2][1],10,f=BetaML.relu, rng=copy(TESTRNG))
size(l7)
l8      = VectorFunctionLayer(size(l7)[2][1],f=BetaML.softmax)
size(l8)
layers   = [l1,l2,l3,l4,l5,l6,l7,l8]
m = NeuralNetworkEstimator(layers=layers,loss=squared_cost,verbosity=HIGH,batch_size=64,epochs=5)

(x_debug,x_other),(y_debug_oh,y_other_oh)  = partition([x_train,y_train_oh],[0.01,0.99],rng=copy(TESTRNG))

ŷ = fit!(m,x_debug,y_debug_oh)

y_true  = inverse_predict(ohm,convert(Matrix{Bool},y_debug_oh))
ŷ_nonoh = inverse_predict(ohm,ŷ)
accuracy(y_true,ŷ_nonoh)
hcat(y_true,ŷ_nonoh)

ŷtest   = predict(m,x_test)
ytest_true  = inverse_predict(ohm,convert(Matrix{Bool},y_test_oh))
ŷtest_nonoh = inverse_predict(ohm,ŷtest)
accuracy(ytest_true,ŷtest_nonoh)
hcat(ytest_true,ŷtest_nonoh)

cm = ConfusionMatrix()
fit!(cm,ytest_true,ŷtest_nonoh)
print(cm)
