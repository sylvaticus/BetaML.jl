"""
  nn.jl

Neural Network algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/lmlj.jl/blob/master/src/nn.jl) - [Julia Package](https://github.com/sylvaticus/lmlj.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/nn.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/lmlj.jl/master?filepath=notebooks%2Fnn.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

Dense and DenseNoBias are already implemented and one can choose them with
predefined activation functions or provide your own (optionally including its derivative)

Alternativly you can implement your own layers.
Each user-implemented layer must define the following methods:

* forward(layer,x)
* backward(layer,x,nextGradient)
* getParams(layer)
* getGradient(layer,x,nextGradient)
* setParams!(layer,w)
* size(layer)

Use the help system to get more info about these methods.

All high-level functions (except the low-level ones) expect x and y as (nRecords × nDimensions) matrices.

"""
module Nn


# ==================================
# Neural Network Module
# ==================================


using Random, Zygote, ProgressMeter
#using ..Utils
import ..Utils: relu, drelu, didentity, dtanh, sigmoid, dsigmoid, softMax,
      dSoftMax, autoJacobian,
      squaredCost,dSquaredCost,
      accuracy,
      makeMatrix, makeColVector, gradientDescentSingleUpdate, oneHotEncoder,
      getScaleFactors, scale
import Base.size

export Layer, forward, backward, getParams, getGradient, setParams!, size, NN,
       buildNetwork, predict, loss, train!, getindex,
       DenseLayer, DenseNoBiasLayer, VectorFunctionLayer,
       relu, drelu, didentity, dtanh, sigmoid, dsigmoid, softMax, dSoftMax,
       autoJacobian,
       accuracy,
       squaredCost, dSquaredCost, makeMatrix, makeColVector, oneHotEncoder,
       getScaleFactors, scale



## Sckeleton for the layer functionality.
# See nn_default_layers.jl for actual implementations

abstract type Layer end

include("Nn_default_layers.jl")

"""
   forward(layer,x)

Predict the output of the layer given the input

# Parameters:
* `layer`:  Worker layer
* `x`:      Input to the layer

# Return:
- An Array{T,1} of the prediction (even for a scalar)
"""
function forward(layer::Layer,x)
 error("Not implemented for this kind of layer. Please implement `forward(layer,x)`.")
end

"""
   backward(layer,x,nextGradient)

Compute backpropagation for this layer

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `nextGradient`: Derivative of the overaall loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer inputs

"""
function backward(layer::Layer,x,nextGradient)
    error("Not implemented for this kind of layer. Please implement `backward(layer,x,nextGradient)`.")
end

"""
   getParams(layer)

Get the layers current value of its trainable parameters

# Parameters:
* `layer`:  Worker layer

# Return:
* The current value of the layer's trainable parameters as tuple of matrices.
It is up to you to decide how to organise this tuple, as long you are consistent
with the getGradient() and setParams() functions.
"""
function getParams(layer::Layer)
  error("Not implemented for this kind of layer. Please implement `getParams(layer)`.")
end

"""
   getGradient(layer,x,nextGradient)

Compute backpropagation for this layer

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `nextGradient`: Derivative of the overaall loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer's trainable parameters
as tuple of matrices. It is up to you to decide how to organise this tuple, as long you are consistent
with the getParams() and setParams() functions.
"""
function getGradient(layer::Layer,x,nextGradient)
    error("Not implemented for this kind of layer. Please implement `getGradient(layer,x,nextGradient)`.")
  end

"""
     setParams!(layer,w)

Set the trainable parameters of the layer with the given values

# Parameters:
* `layer`: Worker layer
* `w`:   The new parameters to set (tuple)

# Notes:
*  The format of the tuple with the parameters must be consistent with those of
the getParams() and getGradient() functions.
"""
function setParams!(layer::Layer,w)
    error("Not implemented for this kind of layer. Please implement `setParams!(layer,w)`.")
end


"""
    size(layer)

SGet the dimensions of the layers in terms of (dimensions in input , dimensions in output)

# Notes:
* You need to use `import Base.size` before defining this function for your layer
"""
function size(layer::Layer)
    error("Not implemented for this kind of layer. Please implement `size(layer)`.")
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

Network predictions

# Parameters:
* `nn`:  Worker network
* `x`:   Input to the network (n × d)
"""
#=
function predict(nn::NN,x)
    makeColVector(x)
    values = x
    for l in nn.layers
        values = forward(l,values)
    end
    return values
end
=#

function predict(nn::NN,x)
    x = makeMatrix(x)
    # get the output dimensions
    n = size(x)[1]
    d = size(nn.layers[end])[2]
    out = zeros(n,d)
    for i in 1:size(x)[1]
        values = x[i,:]
        for l in nn.layers
            values = forward(l,values)
        end
        out[i,:] = values
    end
    return out
end

"""
   loss(fnn,x,y)

Compute avg. network loss on a test set (or a single (1 × d) data point)

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network (n) or (n x d)
* `y`:   Label input (n) or (n x d)
"""
function loss(nn::NN,x,y)
    x = makeMatrix(x)
    y = makeMatrix(y)
    (n,d) = size(x)
    (nn.trained || n == 1) ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `train!(nn,x,y)`"
    ϵ = 0
    for i in 1:n
        ŷ = predict(nn,x[i,:]')[1,:]
        ϵ += nn.cf(ŷ,y[i,:])
    end
    return ϵ/n
end

"""
   getParams(nn)

Retrieve current weigthts

# Parameters:
* `nn`: Worker network

# Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getParams(nn::NN)
  w = Tuple[]
  for l in nn.layers
      push!(w,getParams(l))
  end
  return w
end


"""
   getGradient(nn,x,y)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network (d,1)
* `y`:   Label input (d,1)

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getGradient(nn::NN,x,y)

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
    push!(backwardStack,nn.dcf(forwardStack[end],y)) # adding dϵ_dHatY
  else
    push!(backwardStack,gradient(nn.cf,forwardStack[end],y)[1]) # using AD from Zygote
  end
  for lidx in nLayers:-1:1
     l = nn.layers[lidx]
     dϵ_do = backward(l,forwardStack[lidx],backwardStack[end])
     push!(backwardStack,dϵ_do)
  end
  backwardStack = backwardStack[end:-1:1] # reversing it,

  # Step 3: Computing gradient of weigths
  dWs = Tuple[]
  for lidx in 1:nLayers
     l = nn.layers[lidx]
     dW = getGradient(l,forwardStack[lidx],backwardStack[lidx+1])
     push!(dWs,dW)
  end

  return dWs
end

"""
   setParams!(nn,w)

Update weigths of the network

# Parameters:
* `nn`: Worker network
* `w`:  The new weights to set
"""
function setParams!(nn::NN,w)
    for lidx in 1:length(nn.layers)
        setParams!(nn.layers[lidx],w[lidx])
    end
end


"""
   train!(nn,x,y;epochs,η,rshuffle,nMsg,tol)

Train a neural network with the given x,y data

# Parameters:
* `nn`:       Worker network
* `x`:        Training input to the network (records x dimensions)
* `y`:        Label input (records x dimensions)
* `epochs`:   Number of passages over the training set [def: `1000`]
* `η`:        Learning rate as a function of the epoch [def: `t -> 1/(1+t)`]
* `λ`:        Multiplicative term of the learning rate
* `rShuffle`: Whether to random shuffle the training set at each epoch [def: `true`]
* `nMsg`:     Maximum number of messages to show if all epochs are done [def: `10`]
* `tol`:      A tollerance to stop when the losses stop decreasing [def: `0`]

# Notes:
- use `η = t->k` if you want a learning rate constant to `k`
"""


function show(nn::NN)
  trainedString = nn.trained == true ? "trained" : "non trained"
  println("*** $(nn.name) ($(length(nn.layers)) layers, $(trainedString))\n")
  println("#\t # In \t # Out \t Type")
  for (i,l) in enumerate(nn.layers)
    shapes = size(l)
    println("$i \t $(shapes[1]) \t\t $(shapes[2]) \t\t $(typeof(l)) ")
  end
end

Base.getindex(n::NN, i::AbstractArray) = NN(n.layers[i]...)

# ------------------------------------------------------------------------------
# Optimisation-related functions
abstract type OptimisationAlgorithm end

include("Nn_default_optalgs.jl")


function train!(nn::NN,x,y,optAlg::OptimisationAlgorithm)
   error("Not implemented for this kind of optimisation algorithm. Please implement `train!(NN,x,y,optAlg)`.")
end


end # end module
