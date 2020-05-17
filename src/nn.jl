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

"""

# ==================================
# Neural Network Library
# ==================================

include(joinpath(@__DIR__,"utilities.jl"))

using Random, Zygote

## Sckeleton for the layer functionality.
# See nn_default_layers.jl for actual implementations

abstract type Layer end

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

import Base.size
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

function predictSet(nn::NN,x)
    # get the output dimensions
    n = size(x)[1]
    d = size(nn.layers[end])[2]
    out = zeros(n,d)
    for i in 1:size(x)[1]
        out[i,:] = predict(nn,x[i,:])
    end
    return out
end

"""
   loss(nn,x,y)

Compute network loss on a single data point

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network
* `y`:   Label input
"""
function loss(nn::NN,x,y)
    x = makeColVector(x)
    y = makeColVector(y)
    ŷ = predict(nn,x)
    return nn.cf(ŷ,y)
end

"""
   losses(fnn,x,y)

Compute avg. network loss on a test set

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network (n) or (n x d)
* `y`:   Label input (n) or (n x d)
"""
function losses(nn::NN,x,y)
    x = makeMatrix(x)
    y = makeMatrix(y)
    nn.trained ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `train!(nn,x,y)`"
    ϵ = 0
    for i in 1:size(x)[1]
        xᵢ = x[i,:]'
        yᵢ = y[i,:]'
        ϵ += loss(nn,xᵢ,yᵢ)
    end
    return ϵ/size(x)[1]
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
    push!(backwardStack,nn.dcf(forwardStack[end],y)) # adding d€_dHatY
  else
    push!(backwardStack,gradient(nn.cf,forwardStack[end],y)[1]) # using AD from Zygote
  end
  for lidx in nLayers:-1:1
     l = nn.layers[lidx]
     d€_do = backward(l,forwardStack[lidx],backwardStack[end])
     push!(backwardStack,d€_do)
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
function train!(nn::NN,x,y;maxepochs=1000, η=nothing, rshuffle=true, nMsgs=10, tol=0)
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
            W  = getParams(nn)
            dW = getGradient(nn,xᵢ,yᵢ)
            for (lidx,l) in enumerate(nn.layers)
                oldW = W[lidx]
                dw = dW[lidx]
                newW = oldW .- η .* dw
                setParams!(l,newW)
            end
            ϵ += loss(nn,xᵢ,yᵢ)
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
