"""
  nn.jl

Neural Network algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/lmlj.jl/blob/master/src/nn.jl) - [Julia Package](https://github.com/sylvaticus/lmlj.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/nn.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/lmlj.jl/master?filepath=notebooks%2Fnn.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

FullyConnectedLayer and NoBiasLayer are already implemented and one can choose
them with predefined activation functions or provide your own (optionally including its derivative)

Alternativly you can implement your own layers.
Each user-implemented layer must define the following methods:

forward(layer,I) --> O
- Predict the output of the layer given the input

backward(layer,dϵ/dI^(l+1),I^l) --> dϵ/dI^l
- return the evaluated derivative of the error for the input given (a) the derivative of the
error with respect to the input of the next layer and (b) the input to this layer

getW(layer) --> tuple
- return the current weigths as tuple of matrices. Is is up to you to decide how
to organise this tuple, as long you are consistent with the getDW() and setW() functions

getDW(layer,dϵ/dI^(l+1),I^l) --> tuple
- return the evaluated current weigths as tuple of matrices given (a) the derivative of the
error with respect to the input of the next layer and (b) the input to this layer

setW!(layer, tuple)
- update the weights of the layer with the new values given

size(layer)
- return a tuple (number of dimensions in input, number of dimensions in output)
- not needed if your layer has a `w` weigth matrix defined as member in the standard (to,from) shape
- it would require `import Base.size` in your code

"""

# ==================================
# Neural Network Library
# ==================================

include(joinpath(@__DIR__,"utilities.jl"))



using Random, Zygote

## Some utility functions..


abstract type Layer end

# ------------------------------------------------------------------------------
# Provide default FullyConnectedLayer and NoBiasLayer

"""
   FullyConnectedLayer

Representation of a layer in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct FullyConnectedLayer <: Layer
     w::Array{Float64,2}
     wb::Array{Float64,1}
     f::Function
     df::Union{Function,Nothing}
     """
        FullyConnectedLayer(f,n,nₗ;w,wb,df)

     Instantiate a new FullyConnectedLayer

     Positional arguments:
     * `f`:  Activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     Keyword arguments:
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     * `wb`: Initial weigths with respect to bias [default: `rand(n)`]
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]

     """
     function FullyConnectedLayer(f,nₗ,n;w=rand(n,nₗ),wb=rand(n),df=nothing)
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         wb = reshape(wb,n)
         return new(w,wb,f,df)
     end
end

"""
   NoBiasLayer

Representation of a layer without bias in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct NoBiasLayer <: Layer
     w::Array{Float64,2}
     f::Function
     df::Union{Function,Nothing}
     """
        NoBiasLayer(f,nₗ,n;w,df)

     Instantiate a new NoBiasLayer

     Positional arguments:
     * `f`:  Activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     Keyword arguments:
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     * `df`: Derivative of the activation function [default: `nothing` (i.e. use AD)]
     """
     function NoBiasLayer(f,nₗ,n;w=rand(n,nₗ),df=nothing)
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         return new(w,f,df)
     end
end

"""
   forward(layer,x)

Layer prediction

# Parameters:
* `layer`:  Worker layer
* `x`:      Input to the layer
"""
function forward(layer::FullyConnectedLayer,x)
  return layer.f.(layer.w * x + layer.wb)
end
"""
   forward(layer,x)

Layer prediction of a single data point

# Parameters:
* `layer`:  Worker layer
* `x`:      Input to the layer
"""
function forward(layer::NoBiasLayer,x)
  return layer.f.(layer.w * x)
end


function backward(layer::FullyConnectedLayer,nextGradient,x)
   z = layer.w * x + layer.wb
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dI = layer.w' * dϵ_dz
end

function backward(layer::NoBiasLayer,nextGradient,x)
   z = layer.w * x
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dI = layer.w' * dϵ_dz
end

"""
   getW(layer)

Retrieve current weigthts

# Parameters:
* `layer`: Worker layer

# Return:
* The output is a tuples of the layer's input weigths and bias weigths
"""
function getW(layer::FullyConnectedLayer)
  return (layer.w,layer.wb)
end
function getW(layer::NoBiasLayer)
  return (layer.w,)
end

function getDw(layer::FullyConnectedLayer,nextGradient,x)
   z      = layer.w * x + layer.wb
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x'
   dϵ_dwb = dϵ_dz
   return (dϵ_dw,dϵ_dwb)
end

function getDw(layer::NoBiasLayer,nextGradient,x)
   z      = layer.w * x
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* nextGradient
    else
       dϵ_dz = layer.f'.(z) .* nextGradient # using AD
    end
   dϵ_dw  = dϵ_dz * x'
   return (dϵ_dw,)
end

"""
   setW!(layer,w)

Set weigths of the layer

# Parameters:
* `layer`: Worker layer
* `w`:   The new weights to set (tuple)
"""
function setW!(layer::FullyConnectedLayer,w)
   layer.w = w[1]
   layer.wb = w[2]
end
function setW!(layer::NoBiasLayer,w)
   layer.w = w[1]
end

import Base.size
""" size(Layer) - Return a touple (n dim in input, n dim in oputput)"""
function size(layer::Layer)
    return size(layer.w')
end

# ------------------------------------------------------------------------------
# NN-related functions
"""
   NN

Representation of a Neural Network

# Fields:
* `layers`:  Array of layers objects
* `cf`:      Cost function
* `dcf`:     Derivative of the cost function
* `trained`: Control flag for trained networks
"""
mutable struct NN
    layers::Array{Layer,1}
    cf::Function
    dcf::Union{Function,Nothing}
    trained::Bool
    name::String
end

"""
   buildNetwork

Instantiate a new Feedforward Neural Network

Parameters:
* `layers`:  Array of layers objects
* `cf`:      Cost function
* `dcf`:     Derivative of the cost function

# Notes:
* Even if the network ends with a single output note, the cost function and its
derivative should always expect y and ŷ as column vectors.
"""
function buildNetwork(layers,cf;dcf=nothing,name="Neural Network")
    return NN(layers,cf,dcf,false,name)
end


"""
   predict(nn,x)

Network prediction of a single data point

# Parameters:
* `nn`:  Worker network
* `x`:   Input to the network
"""
function predict(nn::NN,x)
    makeColVector(x)
    values = x
    for l in nn.layers
        values = forward(l,values)
    end
    return values
end

"""
   error(nn,x,y)

Compute network loss on a single data point

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network
* `y`:   Label input
"""
function error(nn::NN,x,y)
    x = makeColVector(x)
    y = makeColVector(y)
    ŷ = predict(nn,x)
    return nn.cf(ŷ,y)
end

"""
   errors(fnn,x,y)

Compute avg. network loss on a test set

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network (n) or (n x d)
* `y`:   Label input (n) or (n x d)
"""
function errors(nn::NN,x,y)
    x = makeMatrix(x)
    y = makeMatrix(y)
    nn.trained ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `train!(nn,x,y)`"
    ϵ = 0
    for i in 1:size(x)[1]
        xᵢ = x[i,:]'
        yᵢ = y[i,:]'
        ϵ += error(nn,xᵢ,yᵢ)
    end
    return ϵ/size(x)[1]
end

"""
   getW(nn)

Retrieve current weigthts

# Parameters:
* `nn`: Worker network

# Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getW(fnn)
  w = Tuple[]
  for l in fnn.layers
      push!(w,getW(l))
  end
  return w
end


"""
   getDw(nn,x,y)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network (d,1)
* `y`:   Label input (d,1)

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getDw(nn,x,y)

  x = makeColVector(x)
  y = makeColVector(y)

  nLayers = length(nn.layers)

  # Stap 1: Forward pass
  forwardStack = Array{Float64,1}[]
  push!(forwardStack,x)
  for l in nn.layers
      push!(forwardStack, forward(l,forwardStack[end]))
  end

  # Step 2: Backpropagation pass
  backwardStack = Array{Float64,1}[]
  if nn.dcf != nothing
    push!(backwardStack,nn.dcf(forwardStack[end],y)) # adding d€_dHatY
  else
    push!(backwardStack,gradient(nn.cf,forwardStack[end],y)[1]) # using AD from Zygote
  end
  for lidx in nLayers:-1:1
     l = nn.layers[lidx]
     d€_do = backward(l,backwardStack[end],forwardStack[lidx])
     push!(backwardStack,d€_do)
  end
  backwardStack = backwardStack[end:-1:1] # reversing it,

  # Step 3: Computing gradient of weigths
  dWs = Tuple[]
  for lidx in 1:nLayers
     l = nn.layers[lidx]
     dW = getDw(l,backwardStack[lidx+1],forwardStack[lidx])
     push!(dWs,dW)
  end

  return dWs
end

"""
   setW!(nn,w)

Update weigths of the network

# Parameters:
* `nn`: Worker network
* `w`:  The new weights to set
"""
function setW!(nn,w)
    for lidx in 1:length(nn.layers)
        setW!(nn.layers[lidx],w[lidx])
    end
end


"""
   train!(nn,x,y;epochs,η,rshuffle)

Train a fnn with the given x,y data

# Parameters:
* `nn`:      Worker network
* `x`:        Training input to the network (records x dimensions)
* `y`:        Label input (records x dimensions)
* `epochs`:   Number of passages over the training set [def = `1000`]
* `η`:        Learning rate. If not provided 1/(1+epoch) is used [def = `nothing`]
* `rshuffle`: Whether to random shuffle the training set at each epoch [def = `true`]
"""
function train!(nn,x,y;maxepochs=1000, η=nothing, rshuffle=true, nMsgs=10, tol=0)
    x = makeMatrix(x)
    y = makeMatrix(y)
    if nMsgs != 0
        println("***\n*** Training $(nn.name) for maximum $maxepochs epochs. Random shuffle: $rshuffle")
    end
    dyn_η = η == nothing ? true : false
    (ϵ,ϵl) = (0,Inf)
    converged = false
    for t in 1:maxepochs
        if rshuffle
           # random shuffle x and y
           ridx = shuffle(1:size(x)[1])
           x = x[ridx, :]
           y = y[ridx , :]
        end
        ϵ = 0
        η = dyn_η ? 1/(1+t) : η
        for i in 1:size(x)[1]
            xᵢ = x[i,:]'
            yᵢ = y[i,:]'
            W  = getW(nn)
            dW = getDw(nn,xᵢ,yᵢ)
            for (lidx,l) in enumerate(nn.layers)
                oldW = W[lidx]
                dw = dW[lidx]
                newW = oldW .- η .* dw
                setW!(l,newW)
            end
            ϵ += error(nn,xᵢ,yᵢ)
        end
        if nMsgs != 0 && (t % ceil(maxepochs/nMsgs) == 0 || t == 1 || t == maxepochs)
          println("Avg. error after epoch $t : $(ϵ/size(x)[1])")
        end

        if abs(ϵl/size(x)[1] - ϵ/size(x)[1]) < (tol * abs(ϵl/size(x)[1]))
            if nMsgs != 0
                println((tol * abs(ϵl/size(x)[1])))
                println("*** Avg. error after epoch $t : $(ϵ/size(x)[1]) (convergence reached")
            end
            converged = true
            break
        else
            ϵl = ϵ
        end
    end
    if nMsgs != 0 && converged == false
        println("*** Avg. error after epoch $maxepochs : $(ϵ/size(x)[1]) (convergence not reached)")
    end
    nn.trained = true
end

import Base.print

function print(io::IO, nn::NN)
  trainedString = nn.trained == true ? "trained" : "non trained"
  println("*** $(nn.name) ($(length(nn.layers)) layers, $(trainedString))\n")
  println("#\t # In \t # Out \t Type")
  for (i,l) in enumerate(nn.layers)
    shapes = size(l)
    println("$i \t $(shapes[1]) \t\t $(shapes[2]) \t\t $(typeof(l)) ")
  end
end


# ==================================
# Specific implementation - FNN definition
# ==================================

l1 = FullyConnectedLayer(tanh,2,3,w=[1 1; 1 1; 1 1], wb=[0 0 0], df=dtanh)
l2 = FullyConnectedLayer(tanh,3,2, w=[1 1 1; 1 1 1], wb=[0 0])
l3 = FullyConnectedLayer(linearf,2,1, w=[1 1], wb=[0])
mynn = buildNetwork([l1,l2,l3],squaredCost,name="Feed-forward Neural Network Model 1")
println(mynn)

# ==================================
# Usage of the FNN
# ==================================

# ------------------
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
xtest = [0.5 0.6; 0.14 0.2; 0.3 0.7; 2.0 4.0]
ytest = [1.1; 0.36; 1.0; 6.0]

train!(mynn,xtrain,ytrain,maxepochs=10000,η=0.01,rshuffle=false,nMsgs=10)
errors(mynn,xtest,ytest) # 0.000196
for (i,r) in enumerate(eachrow(xtest))
  println("x: $r ŷ: $(predict(mynn,r)[1]) y: $(ytest[i])")
end


# ==================================
# Harder case
# ==================================

import Random:seed!
seed!(1234)

xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1; 0.4 0.2; 0.3 0.3; 0.6 0.9; 0.3 0.4; 0.9 0.8]
ytrain = [(0.1*x[1]+0.2*x[2]+0.3)*rand(0.9:0.001:1.1) for x in eachrow(xtrain)]
xtest  = [0.5 0.6; 0.14 0.2; 0.3 0.7; 20.0 40.0;]
ytest  = [(0.1*x[1]+0.2*x[2]+0.3)*rand(0.9:0.001:1.1) for x in eachrow(xtest)]

l1   = FullyConnectedLayer(linearf,2,3,w=ones(3,2), wb=zeros(3))
l2   = FullyConnectedLayer(linearf,3,1, w=ones(1,3), wb=zeros(1))
mynn = buildNetwork([l1,l2],squaredCost,name="Feed-forward Neural Network Model 1")
train!(mynn,xtrain,ytrain,maxepochs=10000,η=0.01,rshuffle=false,nMsgs=10)
errors(mynn,xtest,ytest) # 0.000196
for (i,r) in enumerate(eachrow(xtest))
  println("x: $r ŷ: $(predict(mynn,r)[1]) y: $(ytest[i])")
end

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
d€_do1 = backward(l2,d€_do2,o1)
d€_dX = backward(l1,d€_do1,X)
l1w = getW(l1)
l2w = getW(l2)
l2dw = getDw(l2,d€_do2,o1)
l1dw = getDw(l1,d€_do1,X)
setW!(l1,l1w)
setW!(l2,l2w)
mynn = buildNetwork([l1,l2],squaredCost,dcf=dSquaredCost)
predict(mynn,X)
ϵ2 = error(mynn,X,Y)
