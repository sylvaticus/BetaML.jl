"""
  nn.jl File

Neural Network implementation (Module BetaML.Nn)

`?BetaML.Nn` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Nn.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/BetaML.jl/blob/master/notebooks/Nn.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/BetaML.jl/master?filepath=notebooks%2FNn.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    BetaML.Nn module

Implement the functionality required to define an artificial Neural Network, train it with data, forecast data and assess its performances.

Common type of layers and optimisation algorithms are already provided, but you can define your own ones subclassing respectively the `Layer` and `OptimisationAlgorithm` abstract types.

The module provide the following type or functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition:

- `DenseLayer`: Classical feed-forward layer with user-defined activation function
- `DenseNoBiasLayer`: Classical layer without the bias parameter
- `VectorFunctionLayer`: Parameterless layer whose activation function run over the ensable of its nodes rather than on each one individually
- `buildNetwork`: Build the chained network and define a cost function
- `getParams(nn)`: Retrieve current weigthts
- `getGradient(nn)`: Retrieve the current gradient of the weights
- `setParams!(nn)`: Update the weigths of the network
- `show(nn)`: Print a representation of the Neural Network

Each layer can use a default activation function, one of the functions provided in the `Utils` module (`relu`, `tanh`, `softmax`,...) or you can specify your own function. The derivative of the activation function can be optionally be provided, in such case training will be quicker, altought this difference tends to vanish with bigger datasets.
You can alternativly implement your own layers defining a new type as subtype of the abstract type `Layer`. Each user-implemented layer must define the following methods:

- A suitable constructor
- `forward(layer,x)`
- `backward(layer,x,nextGradient)`
- `getParams(layer)`
- `getGradient(layer,x,nextGradient)`
- `setParams!(layer,w)`
- `size(layer)`

# Model training:

- `trainingInfo(nn)`: Default callback function during training
- `train!(nn)`:  Training function
- `singleUpdate!(θ,▽;optAlg)`: The parameter update made by the specific optimisation algorithm
- `SGD`: The default optimisation algorithm
- `ADAM`: A faster moment-based optimisation algorithm (added in v0.2.2)

To define your own optimisation algorithm define a subtype of `OptimisationAlgorithm` and implement the function `singleUpdate!(θ,▽;optAlg)` and eventually `initOptAlg(⋅)` specific for it.

# Model predictions and assessment:

- `predict(nn)`: Return the output given the data
- `loss(nn)`: Compute avg. network loss on a test set
- `Utils.accuracy(ŷ,y)`: Categorical output accuracy

While high-level functions operating on the dataset expect it to be in the standard format (nRecords × nDimensions matrices) it is custom to represent the chain of a neural network as a flow of column vectors, so all low-level operations (operating on a single datapoint) expect both the input and the output as a column vector.
"""
module Nn

import Base.Threads.@spawn

using Random, Zygote, ProgressMeter, Reexport
import Distributions: Uniform

using ForceImport
@force using ..Api
@force using ..Utils

import Base.size
import Base: +, -, *, /, sum, sqrt

# module own functions
export Layer, forward, backward, getParams, getNParams, getGradient, setParams!, size, NN,
       buildNetwork, predict, loss, train!, getindex, initOptAlg!, singleUpdate!,
       DenseLayer, DenseNoBiasLayer, VectorFunctionLayer,
       Learnable,
       show

# for working on gradient as e.g [([1.0 2.0; 3.0 4.0], [1.0,2.0,3.0]),([1.0,2.0,3.0],1.0)]
"""
Learnable(data)

Structure representing the learnable parameters of a layer or its gradient.

The learnable parameters of a layers are given in the form of a N-tuple of Array{Float64,N2} where N2 can change (e.g. we can have a layer with the first parameter being a matrix, and the second one being a scalar).
We wrap the tuple on its own structure a bit for some efficiency gain, but above all to define standard mathematic operations on the gradients without doing "type pyracy" with respect to Base tuples.
"""
mutable struct Learnable
    data::Tuple{Vararg{Array{Float64,N} where N}}
    function Learnable(data)
        return new(data)
    end
end
function +(items::Learnable...)
  values = collect(items[1].data)
  N = length(values)
  @inbounds for item in items[2:end]
      @inbounds @simd for n in 1:N
          values[n] += item.data[n]
      end
  end
  return Learnable(Tuple(values))
end
sum(items::Learnable...)  = +(items...)
function -(items::Learnable...)
  values = collect(items[1].data)
  N = length(values)
  @inbounds for item in items[2:end]
      @inbounds @simd for n in 1:N
          values[n] -= item.data[n]
      end
  end
  return Learnable(Tuple(values))
end
function *(items::Learnable...)
  values = collect(items[1].data)
  N = length(values)
  @inbounds for item in items[2:end]
      @inbounds @simd for n in 1:N
          values[n] = values[n] .* item.data[n]
      end
  end
  return Learnable(Tuple(values))
end
+(item::Learnable,sc::Number) =  Learnable(Tuple([item.data[i] .+ sc for i in 1:length(item.data)]))
+(sc::Number, item::Learnable) = +(item,sc)
-(item::Learnable,sc::Number) = Learnable(Tuple([item.data[i] .- sc for i in 1:length(item.data)]))
-(sc::Number, item::Learnable) = (-(item,sc)) * -1
*(item::Learnable,sc::Number) = Learnable(item.data .* sc)
*(sc::Number, item::Learnable) = Learnable(sc .* item.data)
/(item::Learnable,sc::Number) = Learnable(item.data ./ sc)
/(sc::Number,item::Learnable,) = Learnable(Tuple([sc ./ item.data[i] for i in 1:length(item.data)]))
sqrt(item::Learnable) = Learnable(Tuple([sqrt.(item.data[i]) for i in 1:length(item.data)]))
/(item1::Learnable,item2::Learnable) = Learnable(Tuple([item1.data[i] ./ item2.data[i] for i in 1:length(item1.data)]))

#=
# not needed ??
function Base.iterate(iter::Learnable, state=(iter.data[1], 1))
           element, count = state
           if count > length(iter)
               return nothing
           elseif count == length(iter)
               return (element, (iter.data[count], count + 1))
           end
           return (element, (iter.data[count+1], count + 1))
end
Base.length(iter::Learnable) = length(iter.data)
#Base.eltype(iter::Learnable) = Int
=#

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
* The current value of the layer's trainable parameters as tuple of matrices. It is up to you to decide how to organise this tuple, as long you are consistent with the `getGradient()` and `setParams()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.
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
* The evaluated gradient of the loss with respect to this layer's trainable parameters as tuple of matrices. It is up to you to decide how to organise this tuple, as long you are consistent with the `getParams()` and `setParams()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.
"""
function getGradient(layer::Layer,x,nextGradient)
    error("Not implemented for this kind of layer. Please implement `getGradient(layer,x,nextGradient)`.")
  end

"""
     setParams!(layer,w)

Set the trainable parameters of the layer with the given values

# Parameters:
* `layer`: Worker layer
* `w`:     The new parameters to set (Learnable)

# Notes:
*  The format of the tuple wrapped by Learnable must be consistent with those of the `getParams()` and `getGradient()` functions.
"""
function setParams!(layer::Layer,w)
    error("Not implemented for this kind of layer. Please implement `setParams!(layer,w)`.")
end


"""
    size(layer)

Get the dimensions of the layers in terms of (dimensions in input , dimensions in output)

# Notes:
* You need to use `import Base.size` before defining this function for your layer
"""
function size(layer::Layer)
    error("Not implemented for this kind of layer. Please implement `size(layer)`.")
end

"""getNParams(layer)

Return the number of parameters of a layer.

It doesn't need to be implemented by each layer type, as it uses getParams().
"""
function getNParams(layer::Layer)
    pars = getParams(layer)
    nP = 0
    for p in pars.data
        nP += *(size(p)...)
    end
    return nP
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
   buildNetwork(layers,cf;dcf,name)

Instantiate a new Feedforward Neural Network

Parameters:
* `layers`: Array of layers objects
* `cf`:     Cost function
* `dcf`:    Derivative of the cost function [def: `nothing`]
* `name`:   Name of the network [def: "Neural Network"]

# Notes:
* Even if the network ends with a single output note, the cost function and its derivative should always expect y and ŷ as column vectors.
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
    #(nn.trained || n == 1) ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `train!(nn,x,y)`"
    ϵ = 0.0
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
@inline function getParams(nn::NN)
  return [getParams(l) for l in nn.layers]
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
function getGradient(nn::NN,x::Union{T,AbstractArray{T,1}},y::Union{T2,AbstractArray{T2,1}}) where { T <: Number, T2 <: Number}

  x = makeColVector(x)
  y = makeColVector(y)

  nLayers = length(nn.layers)

  # Stap 1: Forward pass
  forwardStack = Vector{Vector{Float64}}(undef,nLayers+1)

  forwardStack[1] = x
  @inbounds for (i,l) in enumerate(nn.layers)
      forwardStack[i+1] = forward(l,forwardStack[i])
  end

  # Step 2: Backpropagation pass
  backwardStack = Vector{Vector{Float64}}(undef,nLayers+1)
  if nn.dcf != nothing
    backwardStack[end] = nn.dcf(forwardStack[end],y) # adding dϵ_dHatY
  else
    backwardStack[end] = gradient(nn.cf,forwardStack[end],y)[1] # using AD from Zygote
  end
  @inbounds for lidx in nLayers:-1:1
     l = nn.layers[lidx]
     dϵ_do = backward(l,forwardStack[lidx],backwardStack[lidx+1])
     backwardStack[lidx] = dϵ_do
  end

  # Step 3: Computing gradient of weigths
  dWs = Array{Learnable,1}(undef,nLayers)
  @inbounds for lidx in 1:nLayers
     dWs[lidx] = getGradient(nn.layers[lidx],forwardStack[lidx],backwardStack[lidx+1])
  end

  return dWs
end

"""
   getGradient(nn,xbatch,ybatch)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `nn`:      Worker network
* `xbatch`:  Input to the network (n,d)
* `ybatch`:  Label input (n,d)

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getGradient(nn,xbatch::AbstractArray{T,2},ybatch::AbstractArray{T2,2}) where {T <: Number, T2 <: Number}
    #return [getGradient(nn,xbatch[j,:],ybatch[j,:]) for j in 1:size(xbatch,1)]
    bSize = size(xbatch,1)
    gradients = Array{Vector{Learnable},1}(undef,bSize)
    # Note: in Julia 1.6 somehow the multithreading is less efficient than in Julia 1.5
    # Using @inbounds @simd result faster than using 4 threads, so reverting to it.
    # But to keep following the evolution, as there seems to be some issues on performances
    # in Julia 1.6: https://discourse.julialang.org/t/drop-of-performances-with-julia-1-6-0-for-interpolationkernels/58085
    # Maybe when that's solved it will be again more convenient to use multi-threading
    #Threads.@threads
    @inbounds @simd for j in 1:bSize
       gradients[j] =  getGradient(nn,xbatch[j,:],ybatch[j,:])
    end
    return gradients
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
  show(nn)

Print a representation of the Neural Network (layers, dimensions..)

# Parameters:
* `nn`: Worker network
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

"getNParams(nn) - Return the number of trainable parameters of the neural network."
function getNParams(nn::NN)
    nP = 0
    for l in nn.layers
        nP += getNParams(l)
    end
    return nP
end


Base.getindex(n::NN, i::AbstractArray) = NN(n.layers[i]...)

# ------------------------------------------------------------------------------
# Optimisation-related functions

"""
    OptimisationAlgorithm

Abstract type representing an Optimisation algorithm.

Currently supported algorithms:
- `SGD` (Stochastic) Gradient Descent

See `?[Name OF THE ALGORITHM]` for their details

You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `singleUpdate(⋅)` (type `?singleUpdate` for details).

"""
abstract type OptimisationAlgorithm end

include("Nn_default_optalgs.jl")

"""
   trainingInfo(nn,x,y;n,batchSize,epochs,verbosity,nEpoch,nBatch)

Default callback funtion to display information during training, depending on the verbosity level

# Parameters:
* `nn`: Worker network
* `x`:  Batch input to the network (batchSize,d)
* `y`:  Batch label input (batchSize,d)
* `n`: Size of the full training set
* `nBatches` : Number of baches per epoch
* `epochs`: Number of epochs defined for the training
* `verbosity`: Verbosity level defined for the training (NONE,LOW,STD,HIGH,FULL)
* `nEpoch`: Counter of the current epoch
* `nBatch`: Counter of the current batch

#Notes:
* Reporting of the error (loss of the network) is expensive. Use `verbosity=NONE` for better performances
"""
function trainingInfo(nn,x,y;n,nBatches,epochs,verbosity,nEpoch,nBatch)
   if verbosity == NONE
       return false # doesn't stop the training
   end

   nMsgDict = Dict(LOW => 0, STD => 10,HIGH => 100, FULL => n)
   nMsgs = nMsgDict[verbosity]
   batchSize = size(x,1)

   if verbosity == FULL || ( nBatch == nBatches && ( nEpoch == 1  || nEpoch % ceil(epochs/nMsgs) == 0))

      ϵ = loss(nn,x,y)
      println("Training.. \t avg ϵ on (Epoch $nEpoch Batch $nBatch): \t $(ϵ)")
   end
   return false
end

"""
   train!(nn,x,y;epochs,batchSize,sequential,optAlg,verbosity,cb)

Train a neural network with the given x,y data

# Parameters:
* `nn`:         Worker network
* `x`:          Training input to the network (records x dimensions)
* `y`:          Label input (records x dimensions)
* `epochs`:     Number of passages over the training set [def: `100`]
* `batchSize`:  Size of each individual batch [def: `min(size(x,1),32)`]
* `sequential`: Wether to run all data sequentially instead of random [def: `false`]
* `optAlg`:     The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]
* `verbosity`:  A verbosity parameter for the trade off information / efficiency [def: `STD`]
* `cb`:         A callback to provide information. [def: `trainingInfo`]
* `rng`:        Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Return:
- A named tuple with the following information
  - `epochs`: Number of epochs actually ran
  - `ϵ_epochs`: The average error on each epoch (if `verbosity > LOW`)
  - `θ_epochs`: The parameters at each epoch (if `verbosity > STD`)

# Notes:
- Currently supported algorithms:
    - `SGD`, the classical (Stochastic) Gradient Descent optimiser
    - `ADAM`,  an adaptive moment estimation optimiser
- Look at the individual optimisation algorithm (`?[Name OF THE ALGORITHM]`) for info on its parameter, e.g. [`?SGD`](@ref SGD) for the Stochastic Gradient Descent.
- You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `singleUpdate!(⋅)` (type `?singleUpdate!` for details).
- You can implement your own callback function, altought the one provided by default is already pretty generic (its output depends on the `verbosity` parameter). See [`trainingInfo`](@ref) for informations on the cb parameters.
- Both the callback function and the [`singleUpdate!`](@ref) function of the optimisation algorithm can be used to stop the training algorithm, respectively returning `true` or `stop=true`.
- The verbosity can be set to any of `NONE`,`LOW`,`STD`,`HIGH`,`FULL`.
- The update is done computing the average gradient for each batch and then calling `singleUpdate!` to let the optimisation algorithm perform the parameters update
"""
function train!(nn::NN,x,y; epochs=100, batchSize=min(size(x,1),32), sequential=false, verbosity::Verbosity=STD, cb=trainingInfo, optAlg::OptimisationAlgorithm=ADAM(),rng = Random.GLOBAL_RNG)#,   η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0optAlg::SD=SD())
    if verbosity > STD
        @codeLocation
    end
    x = makeMatrix(x)
    y = makeMatrix(y)
    (n,d)     = size(x)
    batchSize = min(size(x,1),batchSize)
    if verbosity > NONE # Note that are two "Verbosity type" objects. To compare with numbers use Int(NONE) > 1
        println("***\n*** Training $(nn.name) for $epochs epochs with algorithm $(typeof(optAlg)).")
    end
    ϵ_epoch_l = Inf
    θ_epoch_l = getParams(nn)
    ϵ_epoch   = loss(nn,x,y)
    θ_epoch   = getParams(nn)
    ϵ_epochs  = Float64[]
    θ_epochs  = []

    initOptAlg!(optAlg::OptimisationAlgorithm;θ=getParams(nn),batchSize=batchSize,x=x,y=y)

    timetoShowProgress = verbosity > NONE ? 1 : typemax(Int64)
    @showprogress timetoShowProgress "Training the Neural Network..." for t in 1:epochs
       batches = batch(n,batchSize,sequential=sequential,rng=rng)
       nBatches = length(batches)
       if t == 1
           if (verbosity >= STD) push!(ϵ_epochs,ϵ_epoch); end
           if (verbosity > STD) push!(θ_epochs,θ_epoch); end
       end
       for (i,batch) in enumerate(batches)
           xbatch = x[batch, :]
           ybatch = y[batch, :]
           θ   = getParams(nn)
           # remove @spawn and fetch (on next row) to get single thread code
           # note that there is no random number issue here..
           #gradients   = @spawn getGradient(nn,xbatch,ybatch)
           #sumGradient = sum(fetch(gradients))
           gradients   = getGradient(nn,xbatch,ybatch)
           sumGradient = sum(gradients)

           ▽   = sumGradient / length(batch)
           #▽   = gradDiv.(gradSum([getGradient(nn,xbatch[j,:],ybatch[j,:]) for j in 1:batchSize]), batchSize)
           res = singleUpdate!(θ,▽;nEpoch=t,nBatch=i,nBatches=nBatches,xbatch=xbatch,ybatch=ybatch,optAlg=optAlg)
           setParams!(nn,res.θ)
           cbOut = cb(nn,xbatch,ybatch,n=d,nBatches=nBatches,epochs=epochs,verbosity=verbosity,nEpoch=t,nBatch=i)
           if(res.stop==true || cbOut==true)
               nn.trained = true
               return (epochs=t,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs)
           end
       end
       if (verbosity >= STD)
           ϵ_epoch_l = ϵ_epoch
           ϵ_epoch = loss(nn,x,y)
           push!(ϵ_epochs,ϵ_epoch);
       end
       if (verbosity > STD)
           θ_epoch_l = θ_epoch
           θ_epoch = getParams(nn)
           push!(θ_epochs,θ_epoch); end
    end

    if (verbosity > NONE)
        if verbosity == LOW
            ϵ_epoch = loss(nn,x,y)
        end
        println("Training of $epochs epoch completed. Final epoch error: $(ϵ_epoch).");
     end
    nn.trained = true
    return (epochs=epochs,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs)
end

"""
   singleUpdate!(θ,▽;nEpoch,nBatch,batchSize,xbatch,ybatch,optAlg)

Perform the parameters update based on the average batch gradient.

# Parameters:
- `θ`:         Current parameters
- `▽`:         Average gradient of the batch
- `nEpoch`:    Count of current epoch
- `nBatch`:    Count of current batch
- `nBatches`:  Number of batches per epoch
- `xbatch`:    Data associated to the current batch
- `ybatch`:    Labels associated to the current batch
- `optAlg`:    The Optimisation algorithm to use for the update

# Notes:
- This function is overridden so that each optimisation algorithm implement their
own version
- Most parameters are not used by any optimisation algorithm. They are provided
to support the largest possible class of optimisation algorithms
- Some optimisation algorithms may change their internal structure in this function
"""
function singleUpdate!(θ,▽;nEpoch,nBatch,nBatches,xbatch,ybatch,optAlg::OptimisationAlgorithm)
   return singleUpdate!(θ,▽,optAlg;nEpoch=nEpoch,nBatch=nBatch,nBatches=nBatches,xbatch=xbatch,ybatch=ybatch)
end

function singleUpdate!(θ,▽,optAlg::OptimisationAlgorithm;nEpoch,nBatch,nBatches,xbatch,ybatch)
    error("singleUpdate() not implemented for this optimisation algorithm")
end

"""
   initOptAlg!(optAlg;θ,batchSize,x,y)

Initialize the optimisation algorithm

# Parameters:
- `optAlg`:    The Optimisation algorithm to use
- `θ`:         Current parameters
- `batchSize`:    The size of the batch
- `x`:   The training (input) data
- `y`:   The training "labels" to match
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Notes:
- Only a few optimizers need this function and consequently ovverride it. By default it does nothing, so if you want write your own optimizer and don't need to initialise it, you don't have to override this method
"""
initOptAlg!(optAlg::OptimisationAlgorithm;θ,batchSize,x,y,rng = Random.GLOBAL_RNG) = nothing

#=
        if rShuffle
           # random shuffle x and y
           ridx = shuffle(1:size(x)[1])
           x = x[ridx, :]
           y = y[ridx , :]
        end
        ϵ = 0
        #η = dyn_η ? 1/(1+t) : η
        ηₜ = η(t)*λ
        for i in 1:size(x)[1]
            xᵢ = x[i,:]'
            yᵢ = y[i,:]'
            W  = getParams(nn)
            dW = getGradient(nn,xᵢ,yᵢ)
            newW = gradientDescentSingleUpdate(W,dW,ηₜ)
            setParams!(nn,newW)
            ϵ += loss(nn,xᵢ,yᵢ)
        end
        if nMsgs != 0 && (t % ceil(maxEpochs/nMsgs) == 0 || t == 1 || t == maxEpochs)
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
        println("*** Avg. error after epoch $maxEpochs : $(ϵ/size(x)[1]) (convergence not reached)")
    end
    nn.trained = true
end

 =#

end # end module
