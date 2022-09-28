"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
    BetaML.Nn module

Implement the functionality required to define an artificial Neural Network, train it with data, forecast data and assess its performances.

Common type of layers and optimisation algorithms are already provided, but you can define your own ones subclassing respectively the `AbstractLayer` and `OptimisationAlgorithm` abstract types.

The module provide the following types or functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition:

- `DenseLayer`: Classical feed-forward layer with user-defined activation function
- `DenseNoBiasLayer`: Classical layer without the bias parameter
- `VectorFunctionLayer`: Parameterless layer whose activation function run over the ensable of its nodes rather than on each one individually
- `NeuralNetworkEstimator`: Build the chained network and define a cost function


Each layer can use a default activation function, one of the functions provided in the `Utils` module (`relu`, `tanh`, `softmax`,...) or you can specify your own function. The derivative of the activation function can be optionally be provided, in such case training will be quicker, altought this difference tends to vanish with bigger datasets.
You can alternativly implement your own layer defining a new type as subtype of the abstract type `AbstractLayer`. Each user-implemented layer must define the following methods:

- A suitable constructor
- `forward(layer,x)`
- `backward(layer,x,next_gradient)`
- `get_params(layer)`
- `get_gradient(layer,x,next_gradient)`
- `set_params!(layer,w)`
- `size(layer)`

# Model fitting:

- `fit!(nn,X,Y)`:  fitting function
- `fitting_info(nn)`: Default callback function during fitting
- `SGD`:  The classical optimisation algorithm
- `ADAM`: A faster moment-based optimisation algorithm 

To define your own optimisation algorithm define a subtype of `OptimisationAlgorithm` and implement the function `single_update!(θ,▽;opt_alg)` and eventually `init_optalg(⋅)` specific for it.

# Model predictions and assessment:

- `predict(nn)` or `predict(nn,X)`: Return the output given the data
- `loss(nn)`: Compute avg. network loss on a test set
- `accuracy(ŷ,y)`: Categorical output accuracy

While high-level functions operating on the dataset expect it to be in the standard format (n_records × n_dimensions matrices) it is customary to represent the chain of a neural network as a flow of column vectors, so all low-level operations (operating on a single datapoint) expect both the input and the output as a column vector.
"""
module Nn

#import Base.Threads.@spawn

using Random, Zygote, ProgressMeter, Reexport, DocStringExtensions
import Distributions: Uniform

using ForceImport
@force using ..Api
@force using ..Utils

import Base.size
import Base: +, -, *, /, sum, sqrt

import Base.show

# module own functions
export AbstractLayer, forward, backward, get_params, get_nparams, get_gradient, set_params!, size, NN,
       buildNetwork, predict, loss, train!, getindex, init_optalg!, single_update!,
       DenseLayer, DenseNoBiasLayer, VectorFunctionLayer, ScalarFunctionLayer,
       Learnable,
       show, fitting_info

export NeuralNetworkEstimator
export NNHyperParametersSet, NeuralNetworkEstimatorOptionsSet

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
       @inbounds  @simd for n in 1:N # @inbounds  @simd
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
       @inbounds @simd for n in 1:N # @simd
          values[n] -= item.data[n]
      end
  end
  return Learnable(Tuple(values))
end
function *(items::Learnable...)
  values = collect(items[1].data)
  N = length(values)
  @inbounds for item in items[2:end]
      @inbounds @simd for n in 1:N # @simd
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

abstract type AbstractLayer end
abstract type RecursiveLayer <: AbstractLayer end

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
function forward(layer::AbstractLayer,x)
 error("Not implemented for this kind of layer. Please implement `forward(layer,x)`.")
end

"""
    backward(layer,x,next_gradient)

Compute backpropagation for this layer

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `next_gradient`: Derivative of the overal loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer inputs

"""
function backward(layer::AbstractLayer,x,next_gradient)
    error("Not implemented for this kind of layer. Please implement `backward(layer,x,next_gradient)`.")
end

"""
    get_params(layer)

Get the layers current value of its trainable parameters

# Parameters:
* `layer`:  Worker layer

# Return:
* The current value of the layer's trainable parameters as tuple of matrices. It is up to you to decide how to organise this tuple, as long you are consistent with the `get_gradient()` and `set_params()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.
"""
function get_params(layer::AbstractLayer)
  error("Not implemented for this kind of layer. Please implement `get_params(layer)`.")
end

"""
    get_gradient(layer,x,next_gradient)

Compute backpropagation for this layer

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `next_gradient`: Derivative of the overaall loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer's trainable parameters as tuple of matrices. It is up to you to decide how to organise this tuple, as long you are consistent with the `get_params()` and `set_params()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.
"""
function get_gradient(layer::AbstractLayer,x,next_gradient)
    error("Not implemented for this kind of layer. Please implement `get_gradient(layer,x,next_gradient)`.")
  end

"""
    set_params!(layer,w)

Set the trainable parameters of the layer with the given values

# Parameters:
* `layer`: Worker layer
* `w`:     The new parameters to set (Learnable)

# Notes:
*  The format of the tuple wrapped by Learnable must be consistent with those of the `get_params()` and `get_gradient()` functions.
"""
function set_params!(layer::AbstractLayer,w)
    error("Not implemented for this kind of layer. Please implement `set_params!(layer,w)`.")
end


"""
    size(layer)

Get the dimensions of the layers in terms of (dimensions in input , dimensions in output)

# Notes:
* You need to use `import Base.size` before defining this function for your layer
"""
function size(layer::AbstractLayer)
    error("Not implemented for this kind of layer. Please implement `size(layer)`.")
end

"""get_nparams(layer)

Return the number of parameters of a layer.

It doesn't need to be implemented by each layer type, as it uses get_params().
"""
function get_nparams(layer::AbstractLayer)
    pars = get_params(layer)
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
    layers::Array{AbstractLayer,1}
    cf::Function
    dcf::Union{Function,Nothing}
    trained::Bool
    name::String
end

"""
   buildNetwork(layers,cf;dcf,name)

Instantiate a new Feedforward Neural Network

!!! warning
    This function is deprecated and will possibly be removed in BetaML 0.9.
    Use the model [`NeuralNetworkEstimator`](@ref) instead. 

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
    makecolvector(x)
    values = x
    for l in nn.layers
        values = forward(l,values)
    end
    return values
end
=#

function predict(nn::NN,x)
    x = makematrix(x)
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
    x = makematrix(x)
    y = makematrix(y)
    (n,d) = size(x)
    #(nn.trained || n == 1) ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `train!(nn,x,y)`"
    ϵ = 0.0
    for i in 1:n
        ŷ = predict(nn,x[i,:]')[1,:]
        ϵ += nn.cf(y[i,:],ŷ)
    end
    return ϵ/n
end

"""
   get_params(nn)

Retrieve current weigthts

# Parameters:
* `nn`: Worker network

# Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
@inline function get_params(nn::NN)
  return [get_params(l) for l in nn.layers]
end


"""
   get_gradient(nn,x,y)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network (d,1)
* `y`:   Label input (d,1)

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function get_gradient(nn::NN,x::Union{T,AbstractArray{T,1}},y::Union{T2,AbstractArray{T2,1}}) where { T <: Number, T2 <: Number}

  x = makecolvector(x)
  y = makecolvector(y)

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
    backwardStack[end] = nn.dcf(y,forwardStack[end]) # adding dϵ_dHatY
  else
    backwardStack[end] = gradient(nn.cf,y,forwardStack[end])[2] # using AD from Zygote
  end
  @inbounds for lidx in nLayers:-1:1
     l = nn.layers[lidx]
     dϵ_do = backward(l,forwardStack[lidx],backwardStack[lidx+1])
     backwardStack[lidx] = dϵ_do
  end

  # Step 3: Computing gradient of weigths
  dWs = Array{Learnable,1}(undef,nLayers)
  @inbounds for lidx in 1:nLayers
     dWs[lidx] = get_gradient(nn.layers[lidx],forwardStack[lidx],backwardStack[lidx+1])
  end

  return dWs
end

"""
   get_gradient(nn,xbatch,ybatch)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `nn`:      Worker network
* `xbatch`:  Input to the network (n,d)
* `ybatch`:  Label input (n,d)

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function get_gradient(nn,xbatch::AbstractArray{T,2},ybatch::AbstractArray{T2,2}) where {T <: Number, T2 <: Number}
    #return [get_gradient(nn,xbatch[j,:],ybatch[j,:]) for j in 1:size(xbatch,1)]
    bsize = size(xbatch,1)
    gradients = Array{Vector{Learnable},1}(undef,bsize)
    # Note: in Julia 1.6 somehow the multithreading is less efficient than in Julia 1.5
    # Using @inbounds @simd result faster than using 4 threads, so reverting to it.
    # But to keep following the evolution, as there seems to be some issues on performances
    # in Julia 1.6: https://discourse.julialang.org/t/drop-of-performances-with-julia-1-6-0-for-interpolationkernels/58085
    # Maybe when that's solved it will be again more convenient to use multi-threading
    #Threads.@threads
    @inbounds  for j in 1:bsize # @simd
       gradients[j] =  get_gradient(nn,xbatch[j,:],ybatch[j,:])
    end
    return gradients
end

"""
   set_params!(nn,w)

Update weigths of the network

# Parameters:
* `nn`: Worker network
* `w`:  The new weights to set
"""
function set_params!(nn::NN,w)
    for lidx in 1:length(nn.layers)
        set_params!(nn.layers[lidx],w[lidx])
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

"get_nparams(nn) - Return the number of trainable parameters of the neural network."
function get_nparams(nn::NN)
    nP = 0
    for l in nn.layers
        nP += get_nparams(l)
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
- `ADAM` The ADAM algorithm, an adaptive moment estimation optimiser.

See `?[Name OF THE ALGORITHM]` for their details

You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `singleUpdate(⋅)` (type `?singleUpdate` for details).

"""
abstract type OptimisationAlgorithm end

include("Nn_default_optalgs.jl")

"""
   fitting_info(nn,x,y;n,batch_size,epochs,verbosity,n_epoch,n_batch)

Default callback funtion to display information during training, depending on the verbosity level

# Parameters:
* `nn`: Worker network
* `x`:  Batch input to the network (batch_size,d)
* `y`:  Batch label input (batch_size,d)
* `n`: Size of the full training set
* `n_batches` : Number of baches per epoch
* `epochs`: Number of epochs defined for the training
* `verbosity`: Verbosity level defined for the training (NONE,LOW,STD,HIGH,FULL)
* `n_epoch`: Counter of the current epoch
* `n_batch`: Counter of the current batch

#Notes:
* Reporting of the error (loss of the network) is expensive. Use `verbosity=NONE` for better performances
"""
function fitting_info(nn,x,y;n,n_batches,epochs,verbosity,n_epoch,n_batch)
   if verbosity == NONE
       return false # doesn't stop the training
   end

   nMsgDict = Dict(LOW => 0, STD => 10,HIGH => 100, FULL => n)
   nMsgs = nMsgDict[verbosity]
   batch_size = size(x,1)

   if verbosity == FULL || ( n_batch == n_batches && ( n_epoch == 1  || n_epoch % ceil(epochs/nMsgs) == 0))

      ϵ = loss(nn,x,y)
      println("Training.. \t avg ϵ on (Epoch $n_epoch Batch $n_batch): \t $(ϵ)")
   end
   return false
end

"""
   train!(nn,x,y;epochs,batch_size,sequential,opt_alg,verbosity,cb)

Train a neural network with the given x,y data

!!! warning
    This function is deprecated and will possibly be removed in BetaML 0.9.
    Use the model [`NeuralNetworkEstimator`](@ref) instead. 

# Parameters:
* `nn`:         Worker network
* `x`:          Training input to the network (records x dimensions)
* `y`:          Label input (records x dimensions)
* `epochs`:     Number of passages over the training set [def: `100`]
* `batch_size`:  Size of each individual batch [def: `min(size(x,1),32)`]
* `sequential`: Wether to run all data sequentially instead of random [def: `false`]
* `opt_alg`:     The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]
* `verbosity`:  A verbosity parameter for the trade off information / efficiency [def: `STD`]
* `cb`:         A callback to provide information. [def: `fitting_info`]
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
- You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `single_update!(⋅)` (type `?single_update!` for details).
- You can implement your own callback function, altought the one provided by default is already pretty generic (its output depends on the `verbosity` parameter). See [`fitting_info`](@ref) for informations on the cb parameters.
- Both the callback function and the [`single_update!`](@ref) function of the optimisation algorithm can be used to stop the training algorithm, respectively returning `true` or `stop=true`.
- The verbosity can be set to any of `NONE`,`LOW`,`STD`,`HIGH`,`FULL`.
- The update is done computing the average gradient for each batch and then calling `single_update!` to let the optimisation algorithm perform the parameters update
"""
function train!(nn::NN,x,y; epochs=100, batch_size=min(size(x,1),32), sequential=false, verbosity::Verbosity=STD, cb=fitting_info, opt_alg::OptimisationAlgorithm=ADAM(),rng = Random.GLOBAL_RNG)#,   η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0opt_alg::SD=SD())
    if verbosity > STD
        @codelocation
    end
    x = makematrix(x)
    y = makematrix(y)
    (n,d)     = size(x)
    batch_size = min(size(x,1),batch_size)
    if verbosity > NONE # Note that are two "Verbosity type" objects. To compare with numbers use Int(NONE) > 1
        println("***\n*** Training $(nn.name) for $epochs epochs with algorithm $(typeof(opt_alg)).")
    end
    ϵ_epoch_l = Inf
    θ_epoch_l = get_params(nn)
    ϵ_epoch   = loss(nn,x,y)
    θ_epoch   = get_params(nn)
    ϵ_epochs  = Float64[]
    θ_epochs  = []

    init_optalg!(opt_alg::OptimisationAlgorithm;θ=get_params(nn),batch_size=batch_size,x=x,y=y)
    if verbosity == NONE
        showTime = typemax(Float64)
    elseif verbosity <= LOW
        showTime = 50
    elseif verbosity <= STD
        showTime = 1
    elseif verbosity <= HIGH
        showTime = 0.5
    else
        showTime = 0.2
    end
    
    @showprogress showTime "Training the Neural Network..."    for t in 1:epochs
       batches = batch(n,batch_size,sequential=sequential,rng=rng)
       n_batches = length(batches)
       if t == 1
           if (verbosity >= STD) push!(ϵ_epochs,ϵ_epoch); end
           if (verbosity > STD) push!(θ_epochs,θ_epoch); end
       end
       for (i,batch) in enumerate(batches)
           xbatch = x[batch, :]
           ybatch = y[batch, :]
           θ   = get_params(nn)
           # remove @spawn and fetch (on next row) to get single thread code
           # note that there is no random number issue here..
           #gradients   = @spawn get_gradient(nn,xbatch,ybatch)
           #sumGradient = sum(fetch(gradients))
           gradients   = get_gradient(nn,xbatch,ybatch)
           sumGradient = sum(gradients)

           ▽   = sumGradient / length(batch)
           #▽   = gradDiv.(gradSum([get_gradient(nn,xbatch[j,:],ybatch[j,:]) for j in 1:batch_size]), batch_size)
           res = single_update!(θ,▽;n_epoch=t,n_batch=i,n_batches=n_batches,xbatch=xbatch,ybatch=ybatch,opt_alg=opt_alg)
           set_params!(nn,res.θ)
           cbOut = cb(nn,xbatch,ybatch,n=d,n_batches=n_batches,epochs=epochs,verbosity=verbosity,n_epoch=t,n_batch=i)
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
           θ_epoch = get_params(nn)
           push!(θ_epochs,θ_epoch); end
    end

    if (verbosity > NONE)
        if verbosity > LOW
            ϵ_epoch = loss(nn,x,y)
        end
        println("Training of $epochs epoch completed. Final epoch error: $(ϵ_epoch).");
     end
    nn.trained = true
    return (epochs=epochs,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs)
end

"""
   single_update!(θ,▽;n_epoch,n_batch,batch_size,xbatch,ybatch,opt_alg)

Perform the parameters update based on the average batch gradient.

# Parameters:
- `θ`:         Current parameters
- `▽`:         Average gradient of the batch
- `n_epoch`:    Count of current epoch
- `n_batch`:    Count of current batch
- `n_batches`:  Number of batches per epoch
- `xbatch`:    Data associated to the current batch
- `ybatch`:    Labels associated to the current batch
- `opt_alg`:    The Optimisation algorithm to use for the update

# Notes:
- This function is overridden so that each optimisation algorithm implement their
own version
- Most parameters are not used by any optimisation algorithm. They are provided
to support the largest possible class of optimisation algorithms
- Some optimisation algorithms may change their internal structure in this function
"""
function single_update!(θ,▽;n_epoch,n_batch,n_batches,xbatch,ybatch,opt_alg::OptimisationAlgorithm)
   return single_update!(θ,▽,opt_alg;n_epoch=n_epoch,n_batch=n_batch,n_batches=n_batches,xbatch=xbatch,ybatch=ybatch)
end

function single_update!(θ,▽,opt_alg::OptimisationAlgorithm;n_epoch,n_batch,n_batches,xbatch,ybatch)
    error("singleUpdate() not implemented for this optimisation algorithm")
end

"""
   init_optalg!(opt_alg;θ,batch_size,x,y)

Initialize the optimisation algorithm

# Parameters:
- `opt_alg`:    The Optimisation algorithm to use
- `θ`:         Current parameters
- `batch_size`:    The size of the batch
- `x`:   The training (input) data
- `y`:   The training "labels" to match
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

# Notes:
- Only a few optimizers need this function and consequently ovverride it. By default it does nothing, so if you want write your own optimizer and don't need to initialise it, you don't have to override this method
"""
init_optalg!(opt_alg::OptimisationAlgorithm;θ,batch_size,x,y,rng = Random.GLOBAL_RNG) = nothing

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
            W  = get_params(nn)
            dW = get_gradient(nn,xᵢ,yᵢ)
            newW = gradientDescentSingleUpdate(W,dW,ηₜ)
            set_params!(nn,newW)
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

# ------------------------------------------------------------------------------
# V2 Api

#$([println(\"- $(i)\" for i in subtypes(AbstractLayer)])
# $(subtypes(AbstractLayer))
#
"""
**`$(TYPEDEF)`**

Hyperparameters for the `Feedforward` neural network model

## Parameters:
$(FIELDS)

To know the available layers type `subtypes(AbstractLayer)`) and then type `?LayerName` for information on how to use each layer.

"""
Base.@kwdef mutable struct NNHyperParametersSet <: BetaMLHyperParametersSet
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`]
    It must always assume y and ŷ as (n x d) matrices, eventually using `dropdims` inside.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and eventually change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
    tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges = Dict("epochs"=>[50,100,150],"batch_size"=>[2,4,8,16,32],"opt_alg"=>[SGD(λ=2),SGD(λ=1),SGD(λ=3),ADAM(λ=0.5),ADAM(λ=1),ADAM(λ=0.25)], "shuffle"=>[false,true]),multithreads=false)
end

""" 
NeuralNetworkEstimatorOptionsSet

A struct defining the options used by the Feedforward neural network model

## Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct NeuralNetworkEstimatorOptionsSet
   "Cache the results of the fitting stage, as to allow predict(mod) [default: `true`]. Set it to `false` to save memory for large data."
   cache::Bool = true
   "An optional title and/or description for this model"
   descr::String = "" 
   "The verbosity level to be used in training or prediction (see [`Verbosity`](@ref)) [deafult: `STD`]
   "
   verbosity::Verbosity = STD
   "A call back function to provide information during training [def: `fitting_info`"
   cb::Function=fitting_info
   "0ption for hyper-parameters autotuning [def: `false`, i.e. not autotuning performed]. If activated, autotuning is performed on the first `fit!()` call. Controll auto-tuning trough the option `tunemethod` (see the model hyper-parameters)"
   autotune::Bool = false
   "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
   "
   rng::AbstractRNG = Random.GLOBAL_RNG
end

Base.@kwdef mutable struct NeuralNetworkEstimatorLearnableParameters <: BetaMLLearnableParametersSet
    nnstruct::Union{Nothing,NN} = nothing    
end

"""

**`NeuralNetworkEstimator`**

A "feedforward" neural network (supervised).

For the parameters see [`NNHyperParametersSet`](@ref).

## Notes:
- data must be numerical
- the label can be a _n-records_ vector or a _n-records_ by _n-dimensions_ matrix, but the result is always a matrix.
  - For one-dimension regressions drop the unnecessary dimension with `dropdims(ŷ,dims=2)`
  - For classification tasks the columns should normally be interpreted as the probabilities for each categories
"""
mutable struct NeuralNetworkEstimator <: BetaMLSupervisedModel
    hpar::NNHyperParametersSet
    opt::NeuralNetworkEstimatorOptionsSet
    par::Union{Nothing,NeuralNetworkEstimatorLearnableParameters}
    cres::Union{Nothing,AbstractArray}
    fitted::Bool
    info::Dict{String,Any}
end

function NeuralNetworkEstimator(;kwargs...)
    m              = NeuralNetworkEstimator(NNHyperParametersSet(),NeuralNetworkEstimatorOptionsSet(),NeuralNetworkEstimatorLearnableParameters(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        found || error("Keyword \"$kw\" is not part of this model.")
    end
    return m
end

function fit!(m::NeuralNetworkEstimator,X,Y)
    
    (m.fitted) || autotune!(m,(X,Y))

    # Parameter alias..
    layers      = m.hpar.layers
    loss        = m.hpar.loss
    dloss       = m.hpar.dloss
    epochs      = m.hpar.epochs
    batch_size  = m.hpar.batch_size
    opt_alg     = m.hpar.opt_alg
    shuffle     = m.hpar.shuffle
    cache       = m.opt.cache
    descr       = m.opt.descr
    verbosity   = m.opt.verbosity
    cb          = m.opt.cb
    rng         = m.opt.rng
    fitted      = m.fitted

    nR,nD       = size(X)
    nRy, nDy    = size(Y,1), size(Y,2)         
    
    nR == nRy || error("X and Y have different number of records (rows)")

    if !fitted
        if layers == nothing
            innerSize = nDy < 10 ? Int(round(nD*2)) : Int(round(nD*1.3))
            l1 = DenseLayer(nD,innerSize, f=relu, df=drelu, rng=rng)
            l2 = DenseLayer(innerSize,innerSize, f=relu, df=drelu, rng=rng)
            # now let's see if y is continuous, all positives or all in [0,1] and choose the last layer according
            allPos   = all(Y .>= 0.0)
            allSum1  = all(sum(Y,dims=2) .≈ 1.0)
            allProbs = allPos && allSum1  && nDy >1
            if !allPos
                l3 = DenseLayer(innerSize,nDy, f=identity, df=didentity, rng=rng)
                layers = [l1,l2,l3]
            elseif allPos && ! allProbs
                l3 = DenseLayer(innerSize,nDy, f=relu, df=drelu, rng=rng)
                layers = [l1,l2,l3]
            else
                l3 = DenseLayer(innerSize,nDy, f=relu, df=drelu, rng=rng)
                l4 = VectorFunctionLayer(nDy,f=softmax)
                layers = [l1,l2,l3,l4]
            end
        end
        # Check that the first layer has the dimensions of X and the last layer has the output dimensions of Y
        nn_isize = size(layers[1])[1]
        nn_osize = size(layers[end])[2]

        nn_isize == nD || error("The first layer of the network must have the ndims of the input data ($nD) instead of $(nn_isize).")
        nn_osize == nDy || error("The last layer of the network must have the ndims of the output data ($nDy) instead of $(nn_osize). For classification tasks, this is normally the number of possible categories.")

        m.par = NeuralNetworkEstimatorLearnableParameters(NN(deepcopy(layers),loss,dloss,false,descr))
        m.info["epochsRan"] = 0
        m.info["lossPerEpoch"] = Float64[]
        m.info["parPerEpoch"] = []
        m.info["xndims"]   = nD
        m.info["yndims"]   = nDy
        #m.info["fitted_records"] = O
    end


    nnstruct = m.par.nnstruct


    out = train!(nnstruct,X,Y; epochs=epochs, batch_size=batch_size, sequential=!shuffle, verbosity=verbosity, cb=cb, opt_alg=opt_alg,rng = rng)

    m.info["epochsRan"]     += out.epochs
    append!(m.info["lossPerEpoch"],out.ϵ_epochs) 
    append!(m.info["parPerEpoch"],out.θ_epochs) 
    m.info["xndims"]    = nD
    m.info["fitted_records"] = nR
    m.info["nLayers"] = length(nnstruct.layers)
    m.info["nPar"] = get_nparams(m.par.nnstruct)
   
    if cache
       ŷ  = predict(nnstruct,X)
       if ndims(ŷ) > 1 && nn_osize == 1
          m.cres = dropdims(ŷ,dims=2)
       else
          m.cres = ŷ
       end 
    end

    m.fitted = true
    m.par.nnstruct.trained = true

    return cache ? m.cres : nothing
end

#=
# Need to overrite for the size check in the output specific to NN
function predict(m::NeuralNetworkEstimator)
    println("gooo")
    if m.fitted 
        ŷ        = m.cres
        nn_osize = size(m.par.nnstruct.layers[end])[2]
        if ndims(ŷ) > 1 && nn_osize == 1
            return dropdims(ŷ,dims=2)
        else
            return ŷ
        end 
    else
       if m.opt.verbosity > NONE
          @warn "Trying to predict an unfitted model. Run `fit!(model,X,[Y])` before!"
       end
       return nothing
    end
end
=#



function predict(m::NeuralNetworkEstimator,X)
    ŷ        = predict(m.par.nnstruct,X)
    nn_osize = size(m.par.nnstruct.layers[end])[2]
    if ndims(ŷ) > 1 && nn_osize == 1
        return dropdims(ŷ,dims=2)
    else
        return ŷ
    end    
end

function show(io::IO, ::MIME"text/plain", m::NeuralNetworkEstimator)
    if m.fitted == false
        print(io,"NeuralNetworkEstimator - A Feed-forward neural network (unfitted)")
    else
        print(io,"NeuralNetworkEstimator - A Feed-forward neural network (fitted on $(m.info["fitted_records"]) records)")
    end
end

function show(io::IO, m::NeuralNetworkEstimator)
    m.opt.descr != "" && println(io,m.opt.descr)
    if m.fitted == false
        println(io,"NeuralNetworkEstimator - A $(length(m.hpar.layers))-layers feedfordward neural network (unfitted)")
        println(io,"Loss function:")
        println(io,m.hpar.loss)
        println(io,"Optimisation algorithm:")
        println(io,m.hpar.opt_alg)
        println("Layers:")
        println("#\t # In \t\t # Out \t\t Type")
        for (i,l) in enumerate(m.hpar.layers)
          shapes = size(l)
          println("$i \t $(shapes[1]) \t\t $(shapes[2]) \t\t $(typeof(l)) ")
        end
    else
        println(io,"NeuralNetworkEstimator - A $(m.info["xndims"])-dimensions $(m.info["nLayers"])-layers feedfordward neural network (fitted on $(m.info["fitted_records"]) records)")
        println(io,"Cost function:")
        println(io,m.hpar.loss)
        println(io,"Optimisation algorithm:")
        println(io,m.hpar.opt_alg)
        println(io, "Layers:")
        println(io, "#\t # In \t\t  # Out \t\t  Type")
        for (i,l) in enumerate(m.par.nnstruct.layers)
          shapes = size(l)
          println(io, "$i \t $(shapes[1]) \t\t $(shapes[2]) \t\t $(typeof(l)) ")
        end
        println(io,"Output of `info(model)`:")
        for (k,v) in info(m)
            print(io,"- ")
            print(io,k)
            print(io,":\t")
            println(io,v)
        end
    end
end


# MLJ interface
include("Nn_MLJ.jl")

end # end module
