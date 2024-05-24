# [A dimensionality reduction task: imagee encoding (the MNIST dataset)](@id dimensionality_reduction_tutorial)
# For several reasons, including reducing the risk to incur into the [curse of the dimensionality](), we need to reduce the dimensionality of our data.

# Currently BetaML provides two methods for dimensionality reduction, using Principal Component Analysis ([`PCAEncoder`](@ref)), that linearlyreproject the data toward the axis of greeatest variance, or using an AutoEncoder ([`AutoEncoder`](@ref)) that try to learn, unsupervised, the characteristics of the data using neural network.

# We will apply them to 

#
# Data origin:
# - dataset description: [https://en.wikipedia.org/wiki/Iris_flower_data_set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
# - data source we use here: [https://github.com/JuliaStats/RDatasets.jl](https://github.com/JuliaStats/RDatasets.jl)


# ## Library and data loading
using Dates #src
println(now(), " ", "*** Start image recognition tutorial..." )  #src

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))
using Random
using BetaML
using MLDatasets # For loading the training data

Random.seed!(123);
TESTRNG = FIXEDRNG # This could change...

x,y = MLDatasets.MNIST()[:]
x          = permutedims(x,(3,2,1))
x          = convert(Array{Float64,3},x)
x          = reshape(x,size(x,1),size(x,2)*size(x,3))
ohm        = OneHotEncoder()
y_oh       = fit!(ohm,y)
(N,D) = size(x)

x2 = collect(x[1:10,:])

e_layers   = [
    ReshaperLayer((D,1),(28,28,1)), # 784x1 => 28x28x1
    ConvLayer((28,28,1),(5,5),4,stride=2,f=relu,rng=copy(TESTRNG)), # 28x28x1 => 14x14x4
    ConvLayer((14,14,4),(3,3),8,stride=2,f=relu,rng=copy(TESTRNG)), # 14x14x4 => 7x7x8
    ConvLayer((7,7,8),(3,3),8,stride=2,f=relu,rng=copy(TESTRNG)),   # 7x7x8 => 4x4x8
    ReshaperLayer((4,4,8),(128,1)), # 4x4x8 => 128x1 
    DenseLayer(128,2,f=relu,rng=copy(TESTRNG)) # 128x1 => 2x1
]

d_layers   = [
    DenseLayer(2,16,f=relu,rng=copy(TESTRNG)) 
    DenseLayer(16,784,f=relu,rng=copy(TESTRNG))
]


ae_mod     = AutoEncoder(encoded_size=2, e_layers=e_layers, d_layers=d_layers, epochs=4, cache=false)
x_ae       = fit!(ae_mod,x2)   

predict(ae_mod,x2)

xtemp = copy(x2)
xi = x[1,:]
for el in ae_mod.par.fullnn.par.nnstruct.layers[1:ae_mod.par.n_el]
    xi = forward(el,xi)
    println(typeof(xi))
    println(size(xi))

   # xtemp = vcat([forward(el,r) for r in eachrow(xtemp)]'...)
end
return xtemp|> makematrix




DenseLayer(2,4*4*8,f=relu,rng=copy(TESTRNG)) # 2x1 => 128x1 

ReshaperLayer((4*4*8,1),(4,4,8)) # 128x1 => 4x4x8
a = ConvLayer((4,4,8),(3,3),8,stride=1,padding=3,f=relu,rng=copy(TESTRNG)) # 4x4x8 => 8x8x8
a = ConvLayer((8,8,8),(3,3),8,stride=1,padding=3,f=relu,rng=copy(TESTRNG)) # 4x4x8 => 8x8x8

ReshaperLayer((D,1),(28,28,1))


l1     = ReshaperLayer((D,1),(28,28,1))
## 28x28x1 => 14x14x8
l2     = ConvLayer(size(l1)[2],(5,5),8,stride=2,f=relu,rng=copy(TESTRNG))
## 14x14x8 => 7x7x16
l3     = ConvLayer(size(l2)[2],(3,3),16,stride=2,f=relu,rng=copy(TESTRNG))
## 7x7x16 => 4x4x32
l4     = ConvLayer(size(l3)[2],(3,3),32,stride=2,f=relu,rng=copy(TESTRNG))
## 4x4x32 => 2x2x32
l5     = ConvLayer(size(l4)[2],(3,3),32,stride=2,f=relu,rng=copy(TESTRNG))
## 2x2x32 => 1x1x32 (global per layer mean)
l6     = PoolingLayer(size(l5)[2],(2,2),stride=(2,2),f=mean)
## 1x1x32 => 32x1 
l7     = ReshaperLayer(size(l6)[2])
## 32x1 => 10x1 
l8     = DenseLayer(size(l7)[2][1],10,f=identity, rng=copy(TESTRNG))



pca_mod    = PCAEncoder()  
#x_pca      = fit!(pca_mod,x[1:20000,200:end])
e_layers   = [DenseLayer(784,30)]
d_layers   = [DenseLayer(30,784)]
ae_mod     = AutoEncoder(encoded_size=2)
x_ae       = fit!(ae_mod,x[1:200,:])   

e_layers = 



x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(3,2,1))
x_train          = convert(Array{Float64,3},x_train)
x_train          = reshape(x_train,size(x_train,1),size(x_train,2)*size(x_train,3))
ohm              = OneHotEncoder()
y_train_oh       = fit!(ohm,y_train)

x_test, y_test  = MLDatasets.MNIST(split=:test)[:]
x_test          = permutedims(x_test,(3,2,1))
x_test          = convert(Array{Float64,3},x_test)
x_test          = reshape(x_test,size(x_test,1),size(x_test,2)*size(x_test,3))
y_test_oh       = predict(ohm,y_test)
(N,D)  = size(x_train)


using DelimitedFiles
using Statistics
using BenchmarkTools
using Plots
using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using MLDatasets # For loading the training data
#using Images, FileIO, ImageTransformations # For loading the actual images

TESTRNG = FIXEDRNG # This could change...

x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(3,2,1))
x_train          = convert(Array{Float64,3},x_train)
x_train          = reshape(x_train,size(x_train,1),size(x_train,2)*size(x_train,3))
ohm              = OneHotEncoder()
y_train_oh       = fit!(ohm,y_train)

x_test, y_test  = MLDatasets.MNIST(split=:test)[:]
x_test          = permutedims(x_test,(3,2,1))
x_test          = convert(Array{Float64,3},x_test)
x_test          = reshape(x_test,size(x_test,1),size(x_test,2)*size(x_test,3))
y_test_oh       = predict(ohm,y_test)
(N,D)  = size(x_train)

# Building the model:

## 784x1 => 28x28x1
l1     = ReshaperLayer((D,1),(28,28,1))
## 28x28x1 => 14x14x8
l2     = ConvLayer(size(l1)[2],(5,5),8,stride=2,f=relu,rng=copy(TESTRNG))
## 14x14x8 => 7x7x16
l3     = ConvLayer(size(l2)[2],(3,3),16,stride=2,f=relu,rng=copy(TESTRNG))
## 7x7x16 => 4x4x32
l4     = ConvLayer(size(l3)[2],(3,3),32,stride=2,f=relu,rng=copy(TESTRNG))
## 4x4x32 => 2x2x32
l5     = ConvLayer(size(l4)[2],(3,3),32,stride=2,f=relu,rng=copy(TESTRNG))
## 2x2x32 => 1x1x32 (global per layer mean)
l6     = PoolingLayer(size(l5)[2],(2,2),stride=(2,2),f=mean)
## 1x1x32 => 32x1 
l7     = ReshaperLayer(size(l6)[2])
## 32x1 => 10x1 
l8     = DenseLayer(size(l7)[2][1],10,f=identity, rng=copy(TESTRNG))
## 10x1 => 10x1 
l9     = VectorFunctionLayer(size(l8)[2][1],f=BetaML.softmax)
layers = [l1,l2,l3,l4,l5,l6,l7,l8,l9]
m      = NeuralNetworkEstimator(layers=layers,loss=squared_cost,verbosity=HIGH,batch_size=128,epochs=4)

# We train the model only on a subset of the training data, otherwise it is too long for the automated building of this page.
# Training the whole MINST set takes approximatly 16 minutes on a mid-level laptop (on CPU), leading to a test accuracy of 0.969
(x_debug,x_other),(y_debug_oh,y_other_oh)  = partition([x_train,y_train_oh],[0.01,0.99],rng=copy(TESTRNG))

#preprocess!.(layers)
# 0.131836 seconds (477.02 k allocations: 53.470 MiB, 72.73% compilation time)
#@code_warntype preprocess!(l5)

ŷ = fit!(m,x_debug,y_debug_oh)
#@btime fit!(m,x_debug,y_debug_oh)
# 1%: 15.909 s (1940246 allocations: 1.39 GiB)
#     17.509 s (1039126 allocations: 1.37 GiB)
#     15.766 s (1039111 allocations: 1.37 GiB)
#     14.669 s (3129139 allocations: 1.64 GiB) (w threads)
#     18.119 s (1039121 allocations: 1.37 GiB)
#     14.966 s (1039123 allocations: 1.37 GiB) (whout threads)
#      19.357 s (1039123 allocations: 1.37 GiB)

#println(now(), " ", "*** prefit..." )  #src
#ŷ = fit!(m,x_train,y_train_oh)
#println(now(), " ", "*** postfit..." )  #src

#y_true  = inverse_predict(ohm,convert(Matrix{Bool},y_train_oh))
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

res = info(cm)

heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

# -----------------------------------------------------------
# ## Flux implementation
# This is the equivalent workflow in Flux.
# Fitting on the whole training dataset lead to a test accuracy of 0.9658, so likely not statistically different than BetaML, but with still a much faster comutation time, as it takes only 2 minutes instead of 16...


x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(2,1,3)); # For correct img axis
#x_train          = convert(Array{Float32,3},x_train);
x_train          = reshape(x_train,(28,28,1,60000));
y_train          = Flux.onehotbatch(y_train, 0:9)
train_data       = Flux.Data.DataLoader((x_train, y_train), batchsize=128)
#x_test, y_test   = MLDatasets.MNIST.testdata(dir = "data/MNIST")
x_test, y_test   = MLDatasets.MNIST(split=:test)[:]
x_test           = permutedims(x_test,(2,1,3)); # For correct img axis
#x_test           = convert(Array{Float32,3},x_test);
x_test           = reshape(x_test,(28,28,1,10000));
y_test           = Flux.onehotbatch(y_test, 0:9)

model = Chain(
    ## 28x28 => 14x14
    Conv((5, 5), 1=>8, pad=2, stride=2, Flux.relu),
    ## 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=1, stride=2, Flux.relu),
    ## 7x7 => 4x4
    Conv((3, 3), 16=>32, pad=1, stride=2, Flux.relu),
    ## 4x4 => 2x2
    Conv((3, 3), 32=>32, pad=1, stride=2, Flux.relu),
    ## Average pooling on each width x height feature map
    GlobalMeanPool(),
    Flux.flatten,
    Dense(32, 10),
    Flux.softmax
)



myaccuracy(y,ŷ) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
myloss(x, y)     = Flux.crossentropy(model(x), y)

opt = Flux.ADAM()
ps  = Flux.params(model)
number_epochs = 4

[(println(e); Flux.train!(myloss, ps, train_data, opt)) for e in 1:number_epochs]

ŷtrain =   model(x_train)
ŷtest  =   model(x_test)
myaccuracy(y_train,ŷtrain)
myaccuracy(y_test,ŷtest)

plot(Gray.(x_train[:,:,1,2]))

cm = ConfusionMatrix()
fit!(cm,Flux.onecold(y_test) .-1, Flux.onecold(ŷtest) .-1 )
println(cm)

res = info(cm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

