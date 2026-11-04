# # [Image recognition with convolutional neural networks](@id image_recognition_tutorial)

# In this tutorial we classify the handwritten digits of the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset with a convolutional neural network, first with BetaML and then with the equivalent model in [Flux.jl](https://github.com/FluxML/Flux.jl).

using Dates #src
println(now(), " ", "*** Start image recognition tutorial..." )  #src

# ## Library and data loading

# Activating the local environment specific to BetaML documentation
using Pkg
Pkg.activate(joinpath(@__DIR__,"..","..",".."))

using CUDA
using BetaML
using Random
Random.seed!(123);
using Statistics
using Plots
import Flux
using MLDatasets # For loading the training data

using Test #src

TESTRNG = FIXEDRNG # This could change...

# MLDatasets downloads the data the first time it is used. We accept the download without being asked:
ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# We load the training and the testing data. Each image is a 28×28 matrix: we flatten it into a vector, so that each record (row) of `x_train` and `x_test` is an image, and we one-hot encode the labels (the digits).
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

# ## BetaML implementation

# We build the model with a sequence of convolutional layers that progressively reduce the size of the image while increasing the number of channels, followed by a pooling layer that averages each channel and by a dense layer with a softmax to get the probabilities of each digit:

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
m      = NeuralNetworkEstimator(layers=layers,loss=squared_cost,verbosity=HIGH,batch_size=128,epochs=4,device=:gpu,rng=copy(TESTRNG))

# We train the model only on 10% of the training data, to keep the automated building of this page fast.
# Training on the whole MNIST set (4 epochs) takes about 75 seconds on a laptop CPU with 4 threads, leading to a test accuracy of about 0.97.
(x_debug,x_other),(y_debug_oh,y_other_oh)  = partition([x_train,y_train_oh],[0.1,0.9],rng=copy(TESTRNG))

ŷ = fit!(m,x_debug,y_debug_oh)
ŷ = convert(Matrix{Float64},ŷ)

# We measure the accuracy on the training subset...
y_true  = inverse_predict(ohm,convert(Matrix{Bool},y_debug_oh))
ŷ_nonoh = inverse_predict(ohm,ŷ)
accuracy(y_true,ŷ_nonoh)

# ...and on the test set:
ŷtest       = predict(m,x_test)
ytest_true  = inverse_predict(ohm,convert(Matrix{Bool},y_test_oh))
ŷtest_nonoh = inverse_predict(ohm,convert(Matrix{Float64},ŷtest))
testAccuracy = accuracy(ytest_true,ŷtest_nonoh)
@test testAccuracy > 0.7 #src

# We look in more detail at which digits are confused with which with a [`ConfusionMatrix`](@ref):
cm = ConfusionMatrix()
fit!(cm,ytest_true,ŷtest_nonoh)
print(cm)

res = info(cm)

heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

# ## Flux implementation

# This is the equivalent workflow in Flux, trained on the whole training set. It reaches a similar test accuracy (about 0.97), but its training is faster, taking about 15 seconds on the same laptop.
# In Flux, the images are kept as 28×28 matrices with a further channel dimension, the records are on the last dimension, and the data is in `Float32`, the type of the Flux model parameters:

x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(2,1,3)); # For correct img axis
x_train          = reshape(Float32.(x_train),(28,28,1,60000));
y_train          = Flux.onehotbatch(y_train, 0:9)
train_data       = Flux.DataLoader((x_train, y_train), batchsize=128, shuffle=true)
x_test, y_test   = MLDatasets.MNIST(split=:test)[:]
x_test           = permutedims(x_test,(2,1,3)); # For correct img axis
x_test           = reshape(Float32.(x_test),(28,28,1,10000));
y_test           = Flux.onehotbatch(y_test, 0:9)

# We define the same network structure:
model = Flux.Chain(
    ## 28x28 => 14x14
    Flux.Conv((5, 5), 1=>8, pad=2, stride=2, Flux.relu),
    ## 14x14 => 7x7
    Flux.Conv((3, 3), 8=>16, pad=1, stride=2, Flux.relu),
    ## 7x7 => 4x4
    Flux.Conv((3, 3), 16=>32, pad=1, stride=2, Flux.relu),
    ## 4x4 => 2x2
    Flux.Conv((3, 3), 32=>32, pad=1, stride=2, Flux.relu),
    ## Average pooling on each width x height feature map
    Flux.GlobalMeanPool(),
    Flux.flatten,
    Flux.Dense(32, 10),
    Flux.softmax
)

# We use the "explicit" Flux API: the loss function takes the model as its first argument, and the state of the optimisation algorithm is set up once, before the training loop:
myaccuracy(y,ŷ) = mean(Flux.onecold(ŷ) .== Flux.onecold(y))
myloss(m, x, y) = Flux.crossentropy(m(x), y)
opt_state       = Flux.setup(Flux.Adam(), model)
number_epochs   = 4

for e in 1:number_epochs
    println("Epoch $e")
    Flux.train!(myloss, model, train_data, opt_state)
end

# We measure the accuracy on the training and the test sets:
ŷtrain =   model(x_train)
ŷtest  =   model(x_test)
myaccuracy(y_train,ŷtrain)
#-
fluxTestAccuracy = myaccuracy(y_test,ŷtest)
@test fluxTestAccuracy > 0.9 #src

# This is one of the images (the second one of the training set):
heatmap(x_train[:,:,1,2], yflip=true, color=:grays, aspect_ratio=1, legend=false)

# And we look at the confusion matrix also for the Flux model:
cm = ConfusionMatrix()
fit!(cm,Flux.onecold(y_test) .-1, Flux.onecold(ŷtest) .-1 )
println(cm)

res = info(cm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

println(now(), " ", "- DONE image recognition tutorial..." )  #src
