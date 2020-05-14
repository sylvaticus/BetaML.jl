"""
  nn.jl

Neural Network algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/lmlj.jl/blob/master/src/nn.jl) - [Julia Package](https://github.com/sylvaticus/lmlj.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/nn.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/lmlj.jl/master?filepath=notebooks%2Fnn.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

FullyConnectedLayer and NoBiasLayer are already implemented and one can choose
them with predefined activation functions or provide your own (including its derivative!)

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

"""

# ==================================
# Neural Network Library
# ==================================

include(joinpath(@__DIR__,"utilities.jl"))



using Random

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
     df::Function
     """
        FullyConnectedLayer(f,df,n,nₗ;w,wb)

     Instantiate a new FullyConnectedLayer

     Parameters:
     * `f`:  Activation function
     * `df`: Derivative of the activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     * `wb`: Initial weigths with respect to bias [default: `rand(n)`]

     """
     function FullyConnectedLayer(f,df,nₗ,n;w=rand(n,nₗ),wb=rand(n))
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
     df::Function
     """
        NoBiasLayer(f,df,nₗ,n;w)

     Instantiate a new NoBiasLayer

     Parameters:
     *
     * `f`:  Activation function
     * `df`: Derivative of the activation function
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     * `w`:  Initial weigths with respect to input [default: `rand(n,nₗ)`]
     """
     function NoBiasLayer(f,df,nₗ,n;w=rand(n,nₗ))
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
   dϵ_dz = layer.df.(z) .* nextGradient
   dϵ_dI = layer.w' * dϵ_dz
end

function backward(layer::NoBiasLayer,nextGradient,x)
   z = layer.w * x
   dϵ_dz = layer.df.(z) .* nextGradient
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
   dϵ_dz  = layer.df.(z) .* nextGradient
   dϵ_dw  = dϵ_dz * x'
   dϵ_dwb = dϵ_dz
   return (dϵ_dw,dϵ_dwb)
end

function getDw(layer::NoBiasLayer,nextGradient,x)
   z      = layer.w * x
   dϵ_dz  = layer.df.(z) .* nextGradient
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
    dcf::Function
    trained::Bool
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
function buildNetwork(layers,cf,dcf)
    return NN(layers,cf,dcf,false)
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
  push!(backwardStack,nn.dcf(forwardStack[end],y)) # adding d€_dHatY
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
function train!(nn,x,y;epochs=1000, η=nothing, rshuffle=true)
    x = makeMatrix(x)
    y = makeMatrix(y)
    logStep = Int64(ceil(epochs/100))
    dyn_η = η == nothing ? true : false
    for t in 1:epochs
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
        (t % logStep == 0) || t == 1 || t == epochs ? println("Avg. error after epoch $t : $(ϵ/size(x)[1])") : ""
    end
    nn.trained = true
end

# ==================================
# Specific implementation - FNN definition
# ==================================

l1 = FullyConnectedLayer(tanh,dtanh,2,3,w=[1 1; 1 1; 1 1], wb=[0 0 0])
l2 = FullyConnectedLayer(tanh,dtanh,3,2, w=[1 1 1; 1 1 1], wb=[0 0])
l3 = FullyConnectedLayer(linearf,dlinearf,2,1, w=[1 1], wb=[0])
mynn = buildNetwork([l1,l2,l3],squaredCost,dSquaredCost)

# ==================================
# Usage of the FNN
# ==================================

# ------------------
xtrain = [0.1 0.2; 0.3 0.5; 0.4 0.1; 0.5 0.4; 0.7 0.9; 0.2 0.1]
ytrain = [0.3; 0.8; 0.5; 0.9; 1.6; 0.3]
xtest = [0.5 0.6; 0.14 0.2; 0.3 0.7]
ytest = [1.1; 0.36; 1.0]

train!(mynn,xtrain,ytrain,epochs=10000,η=0.01, rshuffle=false)
errors(mynn,xtest,ytest) # 0.000196
for (i,r) in enumerate(eachrow(xtest))
  println("x: $r ŷ: $(predict(mynn,r)[1]) y: $(ytest[i])")
end

# ==================================
# Individual components debugging stuff
# ==================================

l1 = FullyConnectedLayer(relu,drelu,2,3,w=[1 2; -1 -2; 3 -3],wb=[1,-1,0])
l2 = NoBiasLayer(linearf,dlinearf,3,2,w=[1 2 3; -1 -2 -3])
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
mynn = buildNetwork([l1,l2],squaredCost,dSquaredCost)
predict(mynn,X)
ϵ2 = error(mynn,X,Y)
