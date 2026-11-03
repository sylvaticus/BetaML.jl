"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
    BetaML.Nn module

Implement the functionality required to define an artificial Neural Network, train it with data, forecast data and assess its performance.

Common types of layers and optimisation algorithms are already provided, but you can define your own by subclassing respectively the `AbstractLayer` and `OptimisationAlgorithm` abstract types.

The module provides the following types or functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition:

- [`DenseLayer`](@ref): Classical feed-forward layer with user-defined activation function
- [`DenseNoBiasLayer`](@ref): Classical layer without the bias parameter
- [`VectorFunctionLayer`](@ref): Layer whose activation function runs over the ensemble of its nodes rather than on each one individually. No learnable weights on input, optional learnable weights as parameters of the activation function.
- [`ScalarFunctionLayer`](@ref): Layer whose activation function runs over each node individually, like a classic `DenseLayer`, but with no learnable weights on input and optional learnable weights as parameters of the activation function.
- [`ReplicatorLayer`](@ref): Alias for a `ScalarFunctionLayer` with no learnable parameters and identity as activation function
- [`ReshaperLayer`](@ref): Reshape the output of a layer (or the input data) to the shape needed for the next one
- [`PoolingLayer`](@ref): In the middle between `VectorFunctionLayer` and `ScalarFunctionLayer`, it applies a function to the set of nodes defined in a sliding kernel. Weightless.
- [`ConvLayer`](@ref): A generic N+1 (channels) dimensional convolutional layer 
- [`GroupedLayer`](@ref): To stack several layers into a single layer, e.g. for multi-branches networks
- [`NeuralNetworkEstimator`](@ref): Build the chained network and define a cost function

Each layer can use a default activation function, one of the functions provided in the `Utils` module (`relu`, `tanh`, `softmax`,...) or one provided by you.
BetaML will try to recognise if it is a "known" function for which it sets the exact derivatives, otherwise you can normally provide the layer with it.
If the derivative of the activation function is not provided (either manually or automatically), AD will be used and training may be slower, although this difference tends to vanish with bigger datasets.

You can alternatively implement your own layer by defining a new type as subtype of the abstract type `AbstractLayer`. Each user-implemented layer must define the following methods:

- A suitable constructor
- `forward(layer,x)`
- `backward(layer,x,next_gradient)`
- `get_params(layer)`
- `get_gradient(layer,x,next_gradient)`
- `set_params!(layer,w)`
- `size(layer)`

The first three functions work on a single record. Training and prediction internally work on whole batches of records, calling the batched versions `forward_batch(layer,X)`, `backward_batch(layer,X,next_gradient)` and `get_gradient_batch(layer,X,next_gradient)`, whose default implementations simply loop over the records of the batch. Implement the batched versions instead of (or in addition to) the single-record ones to speed up your layer, as all the default layers do.

# Running on GPUs

Neural networks can be trained and used on GPUs of any vendor (NVIDIA, AMD, Apple, Intel). Just load the package of your GPU vendor (e.g. `using CUDA`, `using AMDGPU`, `using Metal` or `using oneAPI`) and either:
- pass your data already on the GPU (e.g. `fit!(m,cu(X),cu(Y))`): with the default option `device=:auto` the network is trained where the data is, and the results are returned there;
- or keep your data on the CPU and use the option `device=:gpu` of [`NeuralNetworkEstimator`](@ref): each batch is moved to the GPU for the computation, while the results are returned on the CPU.

On GPUs, unless the data is already there with a different floating point type, the computations are performed in `Float32`. To run your own layers on GPUs, their `forward_batch`, `backward_batch` and `get_gradient_batch` functions must use array-level operations (and not loops over the individual elements of the arrays) and BetaML must be able to rebuild the layer with its arrays moved to the GPU. This is automatic if the layer has a constructor taking all its fields in order (like the default one), otherwise define `Adapt.adapt_structure(to,layer::MyLayer)` (see the Adapt.jl package).

Most default layers run entirely on GPUs. The exceptions, that still work with data on GPUs but compute record by record on the CPU, are:
- `VectorFunctionLayer` with functions other than `softmax`, unless the function can work on a whole batch and the layer is created with `batched=true`;
- `PoolingLayer` with functions other than `maximum` and `mean`.
In particular, for pooling on GPUs use `PoolingLayer` with `f=maximum` or `f=mean` (e.g. `PoolingLayer(nₗ,k,1,stride=1,padding=0,f=mean)` is the equivalent of `pool1d(x,k,f=mean)`), or a `VectorFunctionLayer` with `batched=true` and a pooling function working on whole batches with array slices (see the documentation of `VectorFunctionLayer`).

# Model fitting:

- `fit!(nn,X,Y)`:  fitting function
- `fitting_info(nn)`: Default callback function during fitting
- `SGD`:  The classical optimisation algorithm
- `ADAM`: A faster moment-based optimisation algorithm 

To define your own optimisation algorithm define a subtype of `OptimisationAlgorithm` and implement the function `single_update!(θ,▽;opt_alg)` and optionally `init_optalg!(⋅)` specific for it.

# Model predictions and assessment:

- `predict(nn)` or `predict(nn,X)`: Return the output given the data


While high-level functions operating on the dataset expect it to be in the standard format (n_records × n_dimensions matrices) it is customary to represent the chain of a neural network as a flow of column vectors, so all low-level operations (operating on a single datapoint) expect both the input and the output as a column vector.
"""
module Nn

#import Base.Threads.@spawn

using Random, LinearAlgebra, StaticArrays, LoopVectorization, Zygote, ProgressMeter, Reexport, DocStringExtensions
import Distributions: Uniform
import Statistics: mean
import ForwardDiff, Adapt, Functors, MLDataDevices
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const

using ForceImport
@force using ..Api
@force using ..Utils

import Base.size
import Base: +, -, *, /, sum, sqrt

import Base.show

# module own functions
export AbstractLayer, forward, backward, get_params, get_gradient, set_params!, size, preprocess! # layer API
export forward_batch, backward_batch, get_gradient_batch # batched layer API
#export forward_old, backward_old, get_gradient_old
export DenseLayer, DenseNoBiasLayer, VectorFunctionLayer, ScalarFunctionLayer, ReplicatorLayer # Available layers
export GroupedLayer
export ConvLayer, ReshaperLayer, PoolingLayer
export init_optalg!, single_update! # Optimizers API
export SGD,ADAM, DebugOptAlg # Available optimizers
export Learnable, fitting_info, NeuralNetworkEstimator, NeuralNetworkE_hp, NeuralNetworkE_options # NN API

# export get_nparams, NN, buildNetwork, predict, loss, train!, getindex, show # old

# for working on gradient as e.g [([1.0 2.0; 3.0 4.0], [1.0,2.0,3.0]),([1.0,2.0,3.0],1.0)]
"""
Learnable(data)

Structure representing the learnable parameters of a layer or its gradient.

The learnable parameters of a layer are given in the form of a N-tuple of arrays of any dimension (e.g. we can have a layer with the first parameter being a matrix, and the second one being a vector). The arrays can be stored on any device (CPU or GPU).
We wrap the tuple in its own structure a bit for some efficiency gain, but above all to define standard mathematical operations on the gradients without doing "type piracy" with respect to Base tuples.

Operations with a scalar convert the scalar to the element type of the arrays, so that e.g. `Float32` parameters stay `Float32`.
"""
struct Learnable{T<:Tuple}
    data::T
end
Learnable(data::AbstractVector) = Learnable(Tuple(data))

# Convert a scalar to the (floating point) eltype of an array, to avoid promoting the whole array (e.g. from Float32 to Float64 on a GPU)
_eltyped(x::AbstractArray{T},sc::Number) where {T<:AbstractFloat} = convert(T,sc)
_eltyped(x,sc::Number) = sc

+(item::Learnable) = item
+(item1::Learnable,item2::Learnable) = Learnable(map((x,y) -> x .+ y, item1.data, item2.data))
-(item1::Learnable,item2::Learnable) = Learnable(map((x,y) -> x .- y, item1.data, item2.data))
-(item1::Learnable,item2::Learnable,items::Learnable...) = -(-(item1,item2),items...)
*(item1::Learnable,item2::Learnable) = Learnable(map((x,y) -> x .* y, item1.data, item2.data))
/(item1::Learnable,item2::Learnable) = Learnable(map((x,y) -> x ./ y, item1.data, item2.data))
sum(items::Learnable...)  = +(items...)
+(item::Learnable,sc::Number) = Learnable(map(x -> x .+ _eltyped(x,sc), item.data))
+(sc::Number, item::Learnable) = +(item,sc)
-(item::Learnable,sc::Number) = Learnable(map(x -> x .- _eltyped(x,sc), item.data))
-(sc::Number, item::Learnable) = Learnable(map(x -> _eltyped(x,sc) .- x, item.data))
*(item::Learnable,sc::Number) = Learnable(map(x -> x .* _eltyped(x,sc), item.data))
*(sc::Number, item::Learnable) = *(item,sc)
/(item::Learnable,sc::Number) = Learnable(map(x -> x ./ _eltyped(x,sc), item.data))
/(sc::Number,item::Learnable) = Learnable(map(x -> _eltyped(x,sc) ./ x, item.data))
sqrt(item::Learnable) = Learnable(map(x -> sqrt.(x), item.data))

## Skeleton for the layer functionality.
# See nn_default_layers.jl for actual implementations

abstract type AbstractLayer end
abstract type RecursiveLayer <: AbstractLayer end

include("Nn_devices.jl")


include("default_layers/DenseLayer.jl")
include("default_layers/DenseNoBiasLayer.jl")
include("default_layers/VectorFunctionLayer.jl")
include("default_layers/ScalarFunctionLayer.jl")
include("default_layers/GroupedLayer.jl")
include("default_layers/ConvLayer.jl")
include("default_layers/PoolingLayer.jl")
include("default_layers/ReshaperLayer.jl")
include("default_layers/RNNLayer.jl")

# Single-record API. Layers implementing the batched API get these functions for free, as they
# are computed by calling the batched functions on a batch with a single record.

# true if `f` has a method specific for the layer type, i.e. not the generic `AbstractLayer` fallback
function _implements(f,layer,nargs)
    m = which(f,Tuple{typeof(layer),ntuple(_ -> Any,nargs-1)...})
    return Base.unwrap_unionall(m.sig).parameters[2] !== AbstractLayer
end
_asbatch(x,shape) = reshape(x,shape...,1)

"""
    forward(layer,x)

Predict the output of the layer given the input

# Parameters:
* `layer`:  Worker layer
* `x`:      Input to the layer

# Return:
- An Array{T,1} of the prediction (even for a scalar)

# Notes:
- For layers implementing [`forward_batch`](@ref) this is computed by calling it on a batch with a single record
"""
function forward(layer::AbstractLayer,x)
    _implements(forward_batch,layer,2) || error("Not implemented for this kind of layer. Please implement `forward(layer,x)` (or `forward_batch(layer,X)`).")
    (insize,outsize) = size(layer)
    return reshape(forward_batch(layer,_asbatch(x,insize)),outsize...)
end

"""
    backward(layer,x,next_gradient)

Compute backpropagation for this layer with respect to its inputs

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `next_gradient`: Derivative of the overall loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer's inputs

# Notes:
- For layers implementing [`backward_batch`](@ref) this is computed by calling it on a batch with a single record
"""
function backward(layer::AbstractLayer,x,next_gradient)
    _implements(backward_batch,layer,3) || error("Not implemented for this kind of layer. Please implement `backward(layer,x,next_gradient)` (or `backward_batch(layer,X,next_gradient)`).")
    (insize,outsize) = size(layer)
    return reshape(backward_batch(layer,_asbatch(x,insize),_asbatch(next_gradient,outsize)),insize...)
end

"""
    get_params(layer)

Get the current value of the layer's trainable parameters

# Parameters:
* `layer`:  Worker layer

# Return:
* The current value of the layer's trainable parameters as a tuple of matrices. It is up to you to decide how to organise this tuple, as long as you are consistent with the `get_gradient()` and `set_params()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.
"""
function get_params(layer::AbstractLayer)
  error("Not implemented for this kind of layer. Please implement `get_params(layer)`.")
end

"""
    get_gradient(layer,x,next_gradient)

Compute backpropagation for this layer with respect to the layer weights

# Parameters:
* `layer`:        Worker layer
* `x`:            Input to the layer
* `next_gradient`: Derivative of the overall loss with respect to the input of the next layer (output of this layer)

# Return:
* The evaluated gradient of the loss with respect to this layer's trainable parameters as a tuple of matrices. It is up to you to decide how to organise this tuple, as long as you are consistent with the `get_params()` and `set_params()` functions. Note that starting from BetaML 0.2.2 this tuple needs to be wrapped in its `Learnable` type.

# Notes:
- For layers implementing [`get_gradient_batch`](@ref) this is computed by calling it on a batch with a single record
"""
function get_gradient(layer::AbstractLayer,x,next_gradient)
    _implements(get_gradient_batch,layer,3) || error("Not implemented for this kind of layer. Please implement `get_gradient(layer,x,next_gradient)` (or `get_gradient_batch(layer,X,next_gradient)`).")
    (insize,outsize) = size(layer)
    return get_gradient_batch(layer,_asbatch(x,insize),_asbatch(next_gradient,outsize))
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

Get the size of the layer in terms of (size in input, size in output) - both as tuples

# Notes:
* You need to use `import Base.size` before defining this function for your layer
"""
function size(layer::AbstractLayer)
    error("Not implemented for this kind of layer. Please implement `size(layer)`.")
end

# Batched API. Training and predictions use these functions. The default implementations loop
# over the records calling the single-record functions. Layers can implement them with
# array-level (vectorised) operations, which are much faster, especially on GPUs.

"""
    forward_batch(layer,X)

Predict the output of the layer for a whole batch of records

# Parameters:
* `layer`: Worker layer
* `X`:     Input to the layer, with the records along the last dimension (i.e. an array whose size is the input size of the layer augmented by the number of records)

# Return:
* The output of the layer, again with the records along the last dimension

# Notes:
* The default implementation calls `forward(layer,x)` for each record. Implement this function directly with array-level operations to speed up your layer.
"""
function forward_batch(layer::AbstractLayer,X)
    return stack(forward(layer,x) for x in eachslice(X,dims=ndims(X)))
end

"""
    backward_batch(layer,X,next_gradient)

Compute backpropagation for this layer with respect to its inputs for a whole batch of records

# Parameters:
* `layer`:         Worker layer
* `X`:             Input to the layer, with the records along the last dimension
* `next_gradient`: Derivative of the overall loss with respect to the input of the next layer (output of this layer), with the records along the last dimension

# Return:
* The evaluated gradient of the loss with respect to this layer's inputs, with the records along the last dimension

# Notes:
* The default implementation calls `backward(layer,x,next_gradient)` for each record. Implement this function directly with array-level operations to speed up your layer.
"""
function backward_batch(layer::AbstractLayer,X,next_gradient)
    return stack(backward(layer,x,g) for (x,g) in zip(eachslice(X,dims=ndims(X)),eachslice(next_gradient,dims=ndims(next_gradient))))
end

"""
    get_gradient_batch(layer,X,next_gradient)

Compute backpropagation for this layer with respect to the layer weights for a whole batch of records

# Parameters:
* `layer`:         Worker layer
* `X`:             Input to the layer, with the records along the last dimension
* `next_gradient`: Derivative of the overall loss with respect to the input of the next layer (output of this layer), with the records along the last dimension

# Return:
* The **sum** over the records of the batch of the gradients of the loss with respect to this layer's trainable parameters, wrapped in a `Learnable` object

# Notes:
* The default implementation sums `get_gradient(layer,x,next_gradient)` over the records. Implement this function directly with array-level operations to speed up your layer.
"""
function get_gradient_batch(layer::AbstractLayer,X,next_gradient)
    return sum(get_gradient(layer,x,g) for (x,g) in zip(eachslice(X,dims=ndims(X)),eachslice(next_gradient,dims=ndims(next_gradient))))
end

"""
    backward_and_gradient_batch(layer,X,next_gradient)

Return both `backward_batch(layer,X,next_gradient)` and `get_gradient_batch(layer,X,next_gradient)`.

Layers can implement this function to share the intermediate computations of the two (the default implementation just calls them both).
"""
function backward_and_gradient_batch(layer::AbstractLayer,X,next_gradient)
    return (backward_batch(layer,X,next_gradient),get_gradient_batch(layer,X,next_gradient))
end

"""
    get_nparams(layer)

Return the number of parameters of a layer.

It doesn't need to be implemented by each layer type, as it uses get_params().
"""
function get_nparams(layer::AbstractLayer)
    return sum(length,get_params(layer).data;init=0)
end

"""
    random_init!(layer;rng)

"Reset" the layer parameters (weights) to random values

It doesn't need to be implemented by each layer type 
"""
function random_init!(layer::AbstractLayer;rng=Random.GLOBAL_RNG)
    pars = get_params(layer)
    for p in pars.data
        sizes = size(p)
        nP = *(sizes...)
        T = eltype(p)
        copyto!(p,xavier_init(nP,nP,sizes;rng=rng,eltype=T)) # copyto! works also for parameters on GPU
    end
end

#=
"""
$(TYPEDSIGNATURES)

Perform a Xavier initialisation of the weights

# Parameters:
- `previous_npar`: number of parameters of the previous layer
- `this_npar`: number of parameters of this layer
- `outsize`: tuple with the size of the weights [def: `(this_npar,previous_npar)`]
- `rng` : random number generator [def: `Random.GLOBAL_RNG`]
- `eltype`: eltype of the weight array [def: `Float64`]

"""
function xavier_init(previous_npar,this_npar,outsize=(this_npar,previous_npar);rng=Random.GLOBAL_RNG,eltype=Float64)
=#



"""
$(TYPEDSIGNATURES)

Preprocess the layer with information known at layer creation (i.e. no data info used)

This function is used for some layers to cache some computation that doesn't require the data and it is called at the beginning of `fit!`.

# Notes:
- as it doesn't depend on data, it is not reset by `reset!`
"""
function preprocess!(layer::AbstractLayer)
    return nothing
end

# ------------------------------------------------------------------------------
# NN-related functions
"""
   NN

Low-level representation of a Neural Network. Use the model `NeuralNetworkEstimator` instead.

# Fields:
* `layers`:  Array of layer objects
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
    This function has been de-exported in BetaML 0.9.
    Use the model [`NeuralNetworkEstimator`](@ref) instead. 

Parameters:
* `layers`: Array of layer objects
* `cf`:     Cost function
* `dcf`:    Derivative of the cost function [def: `nothing`]
* `name`:   Name of the network [def: "Neural Network"]

# Notes:
* Even if the network ends with a single output node, the cost function and its derivative should always expect y and ŷ as column vectors.
"""
function buildNetwork(layers,cf;dcf=match_known_derivatives(cf),name="Neural Network")
    return NN(layers,cf,dcf,false,name)
end


# Internally the network works on batches of data with the records along the last dimension
# ("features × records"), on the device where the network parameters are, while the user-facing
# functions take and return data with the records along the first dimension ("records × features").

"First learnable parameter array of the network (`nothing` for networks without parameters)"
function _first_param(nn::NN)
    for l in nn.layers
        pars = get_params(l).data
        isempty(pars) || return pars[1]
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Prototype array (see `Nn_devices.jl`) of the device and eltype where the network computations take place, i.e. where its parameters are (or where the data `x` is for networks without parameters).
"""
function _compute_proto(nn::NN,x)
    p = _first_param(nn)
    isnothing(p) && return similar(x,float(eltype(x)),0)
    return similar(p,eltype(p),0)
end

"Number of records processed at once when computing predictions or losses over a whole dataset"
const _CHUNK_SIZE = 1024
_chunks(n) = [i:min(i+_CHUNK_SIZE-1,n) for i in 1:_CHUNK_SIZE:n]

"""
Return the data `x` as a (records × features) matrix: a vector is considered a single record, unless the network has a single input (in which case each element is a record)
"""
_asrecords(nn::NN,x::AbstractMatrix) = x
_asrecords(nn::NN,x::AbstractVector) = prod(size(nn.layers[1])[1]) == 1 ? makematrix(x) : reshape(x,1,:)

"Forward pass of a batch `X` (features × records) through all the layers"
function _forward(nn::NN,X)
    for l in nn.layers
        X = forward_batch(l,X)
    end
    return X
end

"""
   predict(nn::NN,x)

Low-level network predictions. Use `predict(m::NeuralNetworkEstimator)` instead

# Parameters:
* `nn`:  Worker network
* `x`:   Input to the network (n × d), or a single record (d)

# Notes:
* The computation happens on the device where the network parameters are, while the predictions are returned on the device where `x` is.
"""
function predict(nn::NN,x)
    lastlayer_size = size(nn.layers[end])[2]
    length(lastlayer_size) == 1 || error("The last NN layer should always be a single dimension vector. Eventually use `ReshaperLayer` to reshape its output as a vector.")
    x     = _asrecords(nn,x)
    proto = _compute_proto(nn,x)
    outs  = [permutedims(_forward(nn,_to_device(proto,permutedims(x[c,:])))) for c in _chunks(size(x,1))]
    out   = length(outs) == 1 ? outs[1] : reduce(vcat,outs)
    return _to_device(similar(x,eltype(out),0),out)
end

"""
   loss(fnn,x,y)

Low level function that computes the avg. network loss on a test set (or a single (1 × d) data point)

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network (n) or (n x d)
* `y`:   Label input (n) or (n x d)
"""
function loss(nn::NN,x,y)
    x     = _asrecords(nn,x)
    y     = size(x,1) == 1 ? reshape(y,1,:) : makematrix(y)
    n     = size(x,1)
    proto = _compute_proto(nn,x)
    ϵ     = 0.0
    for c in _chunks(n)
        ŷ  = permutedims(_forward(nn,_to_device(proto,permutedims(x[c,:]))))
        ϵ += nn.cf(_to_device(proto,y[c,:]),ŷ)
    end
    return ϵ/n
end

"""
   get_params(nn)

Retrieve current weights

# Parameters:
* `nn`: Worker network

# Notes:
* The output is a vector of tuples of each layer's input weights and bias weights
"""
@inline function get_params(nn::NN)
  return [get_params(l) for l in nn.layers]
end

"""
   _batch_gradient(nn,X,Y)

Compute the gradient of the loss with respect to the weights, **summed** over the records of the batch

# Parameters:
* `nn`: Worker network
* `X`:  Input to the network (d × n), on the device of the network parameters
* `Y`:  Label input (dy × n), on the device of the network parameters

# Notes:
* The output is a vector of `Learnable` objects, one for each layer
* The loss function and its derivative are computed with the records along the first dimension (n × dy), as documented in [`NeuralNetworkE_hp`](@ref)
"""
function _batch_gradient(nn::NN,X,Y)
    nLayers = length(nn.layers)

    # Step 1: Forward pass
    forwardStack    = Vector{Any}(undef,nLayers+1)
    forwardStack[1] = X
    for (i,l) in enumerate(nn.layers)
        forwardStack[i+1] = forward_batch(l,forwardStack[i])
    end

    # Step 2: Derivative of the loss with respect to the network output
    y = permutedims(Y)
    ŷ = permutedims(forwardStack[end])
    if !isnothing(nn.dcf)
        dϵ_dŷ = nn.dcf(y,ŷ)
    else
        dϵ_dŷ = gradient(nn.cf,y,ŷ)[2] # using AD from Zygote
    end
    next_gradient = permutedims(dϵ_dŷ)

    # Step 3: Backpropagation pass, computing the gradient of the weights of each layer
    dWs = Vector{Learnable}(undef,nLayers)
    for lidx in nLayers:-1:1
        l = nn.layers[lidx]
        if lidx > 1
            next_gradient, dWs[lidx] = backward_and_gradient_batch(l,forwardStack[lidx],next_gradient)
        else # no need to backpropagate further than the first layer
            dWs[lidx] = get_gradient_batch(l,forwardStack[lidx],next_gradient)
        end
    end
    return dWs
end

"""
   get_gradient(nn,x,y)

Low level function that retrieves the current gradient of the weights (i.e. derivative of the cost with respect to the weights) for a single record. Unexported in BetaML >= v0.9

# Parameters:
* `nn`: Worker network
* `x`:   Input to the network (d,1)
* `y`:   Label input (d,1)

#Notes:
* The output is a vector of tuples of each layer's input weights and bias weights
"""
function get_gradient(nn::NN,x::Union{T,AbstractArray{T,N1}},y::Union{T2,AbstractArray{T2,N2}}) where { T <: Number, T2 <: Number, N1, N2}
    x     = x isa Number ? [x] : x
    y     = y isa Number ? [y] : y
    proto = _compute_proto(nn,x)
    return _batch_gradient(nn,_to_device(proto,reshape(x,:,1)),_to_device(proto,reshape(y,:,1)))
end

"""
   set_params!(nn,w)

Update weights of the network

# Parameters:
* `nn`: Worker network
* `w`:  The new weights to set
"""
function set_params!(nn::NN,w)
    for lidx in 1:length(nn.layers)
        set_params!(nn.layers[lidx],w[lidx])
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

function preprocess!(nn::NN)
    for l in nn.layers
        preprocess!(l)
    end
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

You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `single_update!(⋅)` (type `?single_update!` for details).

"""
abstract type OptimisationAlgorithm end

include("Nn_default_optalgs.jl")

"""
   fitting_info(nn,xbatch,ybatch,x,y;n,n_batches,epochs,epochs_ran,verbosity,n_epoch,n_batch)

Default callback function to display information during training, depending on the verbosity level

# Parameters:
* `nn`: Worker network
* `xbatch`:  Batch input to the network (batch_size,din)
* `ybatch`:  Batch label input (batch_size,dout)
* `x`:  Full input to the network (n_records,din)
* `y`:  Full label input (n_records,dout)
* `n`: Size of the full training set
* `n_batches` : Number of batches per epoch
* `epochs`: Number of epochs defined for the training
* `epochs_ran`: Number of epochs already run in previous training sessions
* `verbosity`: Verbosity level defined for the training (NONE,LOW,STD,HIGH,FULL)
* `n_epoch`: Counter of the current epoch
* `n_batch`: Counter of the current batch

#Notes:
* Reporting of the error (loss of the network) is expensive. Use `verbosity=NONE` for better performance
"""
function fitting_info(nn,xbatch,ybatch,x,y;n,n_batches,epochs,epochs_ran,verbosity,n_epoch,n_batch)
   if verbosity == NONE
       return false # doesn't stop the training
   end

   nMsgDict = Dict(LOW => 0, STD => 10,HIGH => 100, FULL => n)
   nMsgs = nMsgDict[verbosity]

   if verbosity == FULL || ( n_batch == n_batches && ( n_epoch == 1  || n_epoch % ceil(epochs/nMsgs) == 0))

      ϵ = loss(nn,x,y)
      println("Training.. \t avg loss on epoch $n_epoch ($(n_epoch+epochs_ran)): \t $(ϵ)")
   end
   return false
end

"Return a copy on the CPU of the network parameters (e.g. to store them for each epoch)"
_params_cpu_copy(nn::NN) = [Learnable(map(Array,p.data)) for p in get_params(nn)]

"""
   train!(nn,x,y;epochs,batch_size,sequential,opt_alg,verbosity,cb)

Low level function that trains a neural network with the given x,y data.

!!! warning
    This function is deprecated and has been unexported in BetaML v0.9.
    Use the model [`NeuralNetworkEstimator`](@ref) instead. 

# Parameters:
* `nn`:         Worker network
* `x`:          Training input to the network (records x dimensions)
* `y`:          Label input (records x dimensions)
* `epochs`:     Number of passages over the training set [def: `100`]
* `batch_size`:  Size of each individual batch [def: `min(size(x,1),32)`]
* `sequential`: Whether to run all data sequentially instead of randomly [def: `false`]
* `opt_alg`:     The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]
* `verbosity`:  A verbosity parameter for the trade off information / efficiency [def: `STD`]
* `cb`:         A callback to provide information. [def: `fitting_info`]
* `rng`:        Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]

# Return:
- A named tuple with the following information
  - `epochs`: Number of epochs actually run
  - `ϵ_epochs`: The average error on each epoch (if `verbosity > LOW`)
  - `θ_epochs`: The parameters at each epoch (if `verbosity > STD`)

# Notes:
- Currently supported algorithms:
    - `SGD`, the classical (Stochastic) Gradient Descent optimiser
    - `ADAM`,  an adaptive moment estimation optimiser
- Look at the individual optimisation algorithm (`?[Name OF THE ALGORITHM]`) for info on its parameters, e.g. [`?SGD`](@ref SGD) for the Stochastic Gradient Descent.
- You can implement your own optimisation algorithm using a subtype of `OptimisationAlgorithm` and implementing its constructor and the update function `single_update!(⋅)` (type `?single_update!` for details).
- You can implement your own callback function, although the one provided by default is already pretty generic (its output depends on the `verbosity` parameter). See [`fitting_info`](@ref) for information on the cb parameters.
- Both the callback function and the [`single_update!`](@ref) function of the optimisation algorithm can be used to stop the training algorithm, respectively returning `true` or `stop=true`.
- The verbosity can be set to any of `NONE`,`LOW`,`STD`,`HIGH`,`FULL`.
- The update is done computing the average gradient for each batch and then calling `single_update!` to let the optimisation algorithm perform the parameters update
- The training takes place on the device (CPU or GPU) where the network parameters are, moving there each batch of data if needed
"""
function train!(nn::NN,x,y; epochs=100, batch_size=min(size(x,1),32), sequential=false, nepochs_ran=0,verbosity::Verbosity=STD, cb=fitting_info, onfail="tryagain", fail_epoch_ref=2,fail_epoch=5, fail_threshold=0.01,opt_alg::OptimisationAlgorithm=ADAM(),rng = Random.GLOBAL_RNG)#,   η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0opt_alg::SD=SD())
    if verbosity > STD
        @codelocation
    end
    x = makematrix(x)
    y = makematrix(y)

    fail_epoch > fail_epoch_ref || error("fail_epoch ($fail_epoch) must be higher than fail_epoch_ref ($fail_epoch_ref) !")

    preprocess!(nn)

    (n,d)     = size(x)
    batch_size = min(size(x,1),batch_size)
    if verbosity > NONE # Note that these are two "Verbosity type" objects. To compare with numbers use Int(NONE) > 1
        println("***\n*** Training $(nn.name) for $epochs epochs with algorithm $(typeof(opt_alg)).")
    end
    # The computation happens where the network parameters are, with the records along the columns
    proto     = _compute_proto(nn,x)
    xt        = permutedims(x)
    yt        = permutedims(y)

    ϵ_epoch_l = Inf
    θ_epoch_l = _params_cpu_copy(nn)
    ϵ_epoch   = loss(nn,x,y)
    θ_epoch   = _params_cpu_copy(nn)
    ϵ_epochs  = Float64[]
    θ_epochs  = []

    ϵ_refepoch    = 0.0
    ϵ_epoch_fail  = 0.0


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
    
    @showprogress dt=showTime desc="Training the Neural Network..." for t in 1:epochs
       batches = batch(n,batch_size,sequential=sequential,rng=rng)
       n_batches = length(batches)
       #if t == 1 # removed otherwise the array of losses/pars would be nepochs+1
       #    if (verbosity >= STD) push!(ϵ_epochs,ϵ_epoch); end
       #    if (verbosity > STD) push!(θ_epochs,θ_epoch); end
       #end
       for (i,batch) in enumerate(batches)
           xbatch = view(x,batch,:)
           ybatch = view(y,batch,:)
           θ   = get_params(nn)
           sumGradient = _batch_gradient(nn,_to_device(proto,xt[:,batch]),_to_device(proto,yt[:,batch]))
           ▽   = sumGradient / length(batch)

           res = single_update!(θ,▽;n_epoch=t+nepochs_ran,n_batch=i,n_batches=n_batches,xbatch=xbatch,ybatch=ybatch,opt_alg=opt_alg)
           set_params!(nn,res.θ)
           cbOut = cb(nn,xbatch,ybatch,x,y,n=d,n_batches=n_batches,epochs=epochs,epochs_ran=nepochs_ran,verbosity=verbosity,n_epoch=t,n_batch=i)
           if(res.stop==true || cbOut==true)
               nn.trained = true
               return (epochs=t,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs,success=true)
           end
       end # batch end
       t == fail_epoch_ref && (ϵ_refepoch = loss(nn,x,y))
       if t == fail_epoch
           ϵ_epoch_fail = loss(nn,x,y)
           if  (ϵ_refepoch - ϵ_epoch_fail) / ϵ_refepoch < fail_threshold
              if onfail == "error"
                error("Fitting the model on the data failed. Loss is not decreasing.")
              elseif onfail == "stop"
                verbosity >= STD && @info "Fitting the model on the data failed. Loss is not decreasing."
                nn.trained = false
                return (epochs=t,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs,success=false)
              elseif onfail == "tryagain"
                verbosity >= STD && @info "Fitting the model on the data failed. Loss is not decreasing. Trying again (unless this is the last attempt)."
                nn.trained = false
                return (epochs=t,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs,success=false)
              end
            end
       end

       if (verbosity >= STD)
           ϵ_epoch_l = ϵ_epoch
           ϵ_epoch = loss(nn,x,y)
           push!(ϵ_epochs,ϵ_epoch)
           if isnan(ϵ_epoch)
                println("Loss is NaN. Stopping training.")
                nn.trained = false
                return (epochs=t,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs,success=false)
           end
       end
       if (verbosity > STD)
           θ_epoch_l = θ_epoch
           θ_epoch = _params_cpu_copy(nn)
           push!(θ_epochs,θ_epoch);
        end
    end # end of epoch

    if (verbosity > NONE)
        if verbosity > LOW
            ϵ_epoch = loss(nn,x,y)
        end
        println("Training of $epochs epochs completed. Final epoch error: $(ϵ_epoch).");
    end
    nn.trained = true
    return (epochs=epochs,ϵ_epochs=ϵ_epochs,θ_epochs=θ_epochs,success=true)
end

"""
   single_update!(θ,▽;n_epoch,n_batch,n_batches,xbatch,ybatch,opt_alg)

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
- This function is overridden so that each optimisation algorithm implements its
own version
- Most parameters are not used by any optimisation algorithm. They are provided
to support the largest possible class of optimisation algorithms
- Some optimisation algorithms may change their internal structure in this function
"""
function single_update!(θ,▽;n_epoch,n_batch,n_batches,xbatch,ybatch,opt_alg::OptimisationAlgorithm)
   return single_update!(θ,▽,opt_alg;n_epoch=n_epoch,n_batch=n_batch,n_batches=n_batches,xbatch=xbatch,ybatch=ybatch)
end

function single_update!(θ,▽,opt_alg::OptimisationAlgorithm;n_epoch,n_batch,n_batches,xbatch,ybatch)
    error("single_update!() not implemented for this optimisation algorithm")
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
* `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]

# Notes:
- Only a few optimizers need this function and consequently override it. By default it does nothing, so if you want to write your own optimizer and don't need to initialise it, you don't have to override this method
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
        if nMsgs != 0 && (t % ceil(epochs/nMsgs) == 0 || t == 1 || t == epochs)
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
        println("*** Avg. error after epoch $epochs : $(ϵ/size(x)[1]) (convergence not reached)")
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

To know the available layers, type `subtypes(AbstractLayer)` and then type `?LayerName` for information on how to use each layer.

"""
Base.@kwdef mutable struct NeuralNetworkE_hp <: BetaMLHyperParametersSet
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`]
    It must always assume y and ŷ as (n x d) matrices, possibly using `dropdims` inside, and return the loss summed over the n records. Write it with array-level operations (e.g. broadcasting) for it to work also on GPUs.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost` if `loss==squared_cost`, `nothing` otherwise, i.e. use the derivative of the squared cost or autodiff]"
    dloss::Union{Function,Nothing}  = nothing
    "Number of epochs, i.e. passages through the whole training sample [def: `200`]"
    epochs::Int64 = 200
    "Size of each individual batch [def: `16`]"
    batch_size::Int64 = 16
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and optionally change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
    tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges = Dict("epochs"=>[50,100,150],"batch_size"=>[2,4,8,16,32],"opt_alg"=>[SGD(λ=2),SGD(λ=1),SGD(λ=3),ADAM(λ=0.5),ADAM(λ=1),ADAM(λ=0.25)], "shuffle"=>[false,true]),multithreads=false)
end

""" 
NeuralNetworkE_options

A struct defining the options used by the Feedforward neural network model

## Parameters:
$(FIELDS)
"""
Base.@kwdef mutable struct NeuralNetworkE_options
   "Cache the results of the fitting stage, so as to allow predict(mod) [default: `true`]. Set it to `false` to save memory for large data."
   cache::Bool = true
   "An optional title and/or description for this model"
   descr::String = "" 
   "The verbosity level to be used in training or prediction (see [`Verbosity`](@ref)) [default: `STD`]"
   verbosity::Verbosity = STD
   "A callback function to provide information during training [def: `fitting_info`]"
   cb::Function=fitting_info
   "Option for hyper-parameters autotuning [def: `false`, i.e. no autotuning performed]. If activated, autotuning is performed on the first `fit!()` call. Control auto-tuning through the option `tunemethod` (see the model hyper-parameters)"
   autotune::Bool = false
   """
   The strategy to apply in case of failure during the fitting (loss not decreasing) [def: "tryagain"].
   The default is to try again with a new random initialization of the weights and the batch ("tryagain"). Other valid values are "continue", "stop" and "error". The first ignores the issue, the second one stops the training (but without generating an error) and the last one stops the fitting and generates an error.
   """
   onfail::String = "tryagain"

   """
   The epoch to use as initial comparison to check for the onfail strategy [def: `2`]
   """
   fail_epoch_ref::Int64 = 2
   """
   The epoch at which to try the onfail strategy [def: `5`]
   """
   fail_epoch::Int64 = 5
   """
   The (relative) threshold to use for the onfail strategy [def: `0.01`]. If the loss doesn't decrease by at least this value, the onfail strategy is applied. Note that this is computed in relation to the loss on the _second_ epoch, so it is not an absolute value. 
   """
   fail_threshold::Float64 = 0.01
   """
   The maximum number of attempts to use for the tryagain strategy [def: `10`]
   """
   fail_attempts::Int64 = 10
   "Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG = Random.GLOBAL_RNG
   """
   The device where to train the network and compute the predictions [def: `:auto`]. Options are:
   - `:auto`: use the device where the training data is (e.g. the GPU if `X` is a GPU array);
   - `:cpu`: use the CPU;
   - `:gpu`: use the GPU of the backend package loaded (e.g. `using CUDA`, `using AMDGPU`, `using Metal` or `using oneAPI`), moving each batch of data there if it is on the CPU;
   - a device object of the MLDataDevices.jl package (e.g. `CUDADevice()`), to choose a specific GPU backend.
   Predictions are always returned on the device of the input data. On GPUs the network parameters are converted to `Float32` (unless the data is already on the GPU with a different floating point type).
   Models saved with [`model_save`](@ref) while on a GPU have their parameters saved on the CPU. With `device=:auto`, they are moved back to the same kind of GPU when used for predictions, if its package is loaded (otherwise they stay on the CPU).
   """
   device::Union{Symbol,String,MLDataDevices.AbstractDevice} = :auto
end

Base.@kwdef mutable struct NeuralNetworkEstimator_lp <: BetaMLLearnableParametersSet
    nnstruct::Union{Nothing,NN} = nothing    
end

"""

**`NeuralNetworkEstimator`**

A "feedforward" (but also multi-branch) neural network (supervised).

For the parameters see [`NeuralNetworkE_hp`](@ref) and for the training options [`NeuralNetworkE_options`](@ref) (we have a few more options for this specific estimator).

# Notes:
- data must be numerical
- the label can be a _n-records_ vector or a _n-records_ by _n-dimensions_ matrix, but the result is always a matrix.
  - For one-dimension regressions drop the unnecessary dimension with `dropdims(ŷ,dims=2)`
  - For classification tasks the columns should normally be interpreted as the probabilities for each category

# Examples:

- Classification...

```julia
julia> using BetaML

julia> X = [1.8 2.5; 0.5 20.5; 0.6 18; 0.7 22.8; 0.4 31; 1.7 3.7];

julia> y = ["a","b","b","b","b","a"];

julia> ohmod = OneHotEncoder()
A OneHotEncoder BetaMLModel (unfitted)

julia> y_oh  = fit!(ohmod,y)
6×2 Matrix{Bool}:
 1  0
 0  1
 0  1
 0  1
 0  1
 1  0

julia> layers = [DenseLayer(2,6),DenseLayer(6,2),VectorFunctionLayer(2,f=softmax)];

julia> m      = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=300,verbosity=LOW)
NeuralNetworkEstimator - A Feed-forward neural network (unfitted)

julia> ŷ_prob = fit!(m,X,y_oh)
***
*** Training  for 300 epochs with algorithm ADAM.
Training..       avg ϵ on (Epoch 1 Batch 1):     0.4116936481380642
Training of 300 epochs completed. Final epoch error: 0.44308719831108734.
6×2 Matrix{Float64}:
 0.853198    0.146802
 0.0513715   0.948629
 0.0894273   0.910573
 0.0367079   0.963292
 0.00548038  0.99452
 0.808334    0.191666

julia> ŷ      = inverse_predict(ohmod,ŷ_prob)
6-element Vector{String}:
 "a"
 "b"
 "b"
 "b"
 "b"
 "a"
```
- Regression...

```julia
julia> using BetaML

julia> X = [1.8 2.5; 0.5 20.5; 0.6 18; 0.7 22.8; 0.4 31; 1.7 3.7];

julia> y = 2 .* X[:,1] .- X[:,2] .+ 3;

julia> layers = [DenseLayer(2,6),DenseLayer(6,6),DenseLayer(6,1)];

julia> m      = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=3000,verbosity=LOW)
NeuralNetworkEstimator - A Feed-forward neural network (unfitted)

julia> ŷ      = fit!(m,X,y);
***
*** Training  for 3000 epochs with algorithm ADAM.
Training..       avg ϵ on (Epoch 1 Batch 1):     33.30063874270561
Training of 3000 epochs completed. Final epoch error: 34.61265465430473.

julia> hcat(y,ŷ)
6×2 Matrix{Float64}:
   4.1    4.11015
 -16.5  -16.5329
 -13.8  -13.8381
 -18.4  -18.3876
 -27.2  -27.1667
   2.7    2.70542
```  
"""
mutable struct NeuralNetworkEstimator <: BetaMLSupervisedModel
    hpar::NeuralNetworkE_hp
    opt::NeuralNetworkE_options
    par::Union{Nothing,NeuralNetworkEstimator_lp}
    cres::Union{Nothing,AbstractArray}
    fitted::Bool
    info::Dict{String,Any}
end

function NeuralNetworkEstimator(;kwargs...)
    m              = NeuralNetworkEstimator(NeuralNetworkE_hp(),NeuralNetworkE_options(),NeuralNetworkEstimator_lp(),nothing,false,Dict{Symbol,Any}())
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
    # Special correction for NeuralNetworkE_hp
    kwkeys = keys(kwargs) #in(2,[1,2,3])
    #if !in(:dloss,kwkeys) # if dloss is not explicitly provided
    #    if   (in(:loss,kwkeys) && kwargs[:loss] == squared_cost  ) || # loss is explicitly provided and it is equal to squared_loss
    #        (!in(:loss,kwkeys) )                               # (or) loss is not explicitly provided
    #        m.hpar.dloss = dsquared_cost
    #    end
    #end
    if !in(:dloss,kwkeys) # if dloss is not explicitly provided
        m.hpar.dloss = match_known_derivatives(m.hpar.loss)
    end
    return m
end

"""
$(TYPEDSIGNATURES)

Return the prototype array (see `Nn_devices.jl`) of the device and eltype where to train the network `nn`, given the `device` option and the training data `X`.
"""
function _target_proto(nn::NN,device,X;verbosity=STD)
    device = _device_spec(device)
    p      = _first_param(nn)
    tref   = if device === :cpu || device isa MLDataDevices.CPUDevice
        nothing
    elseif device === :auto
        _iscpu(X) ? nothing : X
    elseif device === :gpu
        _device_ref(_gpu_device())
    else
        _device_ref(device)
    end
    if isnothing(tref) || _iscpu(tref)
        T = isnothing(p) ? float(eltype(X)) : eltype(p)
        return Vector{T}(undef,0)
    end
    if !isnothing(p) && _samedevice(p,tref)
        T = eltype(p)                 # the network is already there
    elseif _samedevice(X,tref) && eltype(X) <: AbstractFloat
        T = eltype(X)                 # the user put the data on the GPU with this float type
    else
        T = Float32
        if !isnothing(p) && eltype(p) != T && verbosity >= STD
            @info "Converting the neural network parameters to Float32 to run on the GPU." maxlog=1
        end
    end
    return similar(tref,T,0)
end

"""
$(TYPEDSIGNATURES)

Move (if needed) the network parameters to the device where to train the network, given the `device` option and the training data `X`.
"""
function _place_model!(nn::NN,device,X;verbosity=STD)
    proto = _target_proto(nn,device,X;verbosity=verbosity)
    p     = _first_param(nn)
    if !isnothing(p) && !(_samedevice(p,proto) && eltype(p) == eltype(proto))
        verbosity > STD && println("Moving the neural network parameters to the device of a `$(typeof(proto))` array.")
        mover     = _DeviceMover(proto)
        nn.layers = [_layer_to_device(mover,l) for l in nn.layers]
    end
    return nothing
end

# Before saving, move the parameters (and the optimiser state) on the CPU, as GPU arrays can't be saved
_optalg_to_cpu(opt_alg::OptimisationAlgorithm) = opt_alg
function _optalg_to_cpu(opt_alg::ADAM)
    a2   = ADAM(η=opt_alg.η, λ=opt_alg.λ, β₁=opt_alg.β₁, β₂=opt_alg.β₂, ϵ=opt_alg.ϵ)
    a2.m = [Learnable(map(_tocpu,l.data)) for l in opt_alg.m]
    a2.v = [Learnable(map(_tocpu,l.data)) for l in opt_alg.v]
    return a2
end
function Api._model_to_save(m::NeuralNetworkEstimator)
    (isnothing(m.par) || isnothing(m.par.nnstruct)) && return m
    nn = m.par.nnstruct
    p  = _first_param(nn)
    (isnothing(p) || _iscpu(p)) && return m
    mover = _DeviceMover(Vector{eltype(p)}(undef,0))
    nncpu = NN([_layer_to_device(mover,l) for l in nn.layers],nn.cf,nn.dcf,nn.trained,nn.name)
    hpar  = NeuralNetworkE_hp(;(f => getfield(m.hpar,f) for f in fieldnames(NeuralNetworkE_hp))...)
    hpar.opt_alg = _optalg_to_cpu(m.hpar.opt_alg)
    cres  = isnothing(m.cres) ? nothing : _tocpu(m.cres)
    info  = copy(m.info)
    dname = _device_name(p)
    isnothing(dname) || (info["device_at_save"] = dname) # to move the model back there when used
    return NeuralNetworkEstimator(hpar,m.opt,NeuralNetworkEstimator_lp(nncpu),cres,m.fitted,info)
end

"""
$(TYPEDSIGNATURES)

Move back to the GPU a model saved (on the CPU) with [`model_save`](@ref) while it was on a GPU, if that kind of GPU is available. Otherwise the model stays on the CPU.
"""
function _restore_saved_device!(m::NeuralNetworkEstimator)
    dname = pop!(m.info,"device_at_save",nothing)
    isnothing(dname) && return nothing
    p = _first_param(m.par.nnstruct)
    (isnothing(p) || !_iscpu(p)) && return nothing
    D = _device_type(dname)
    if !isnothing(D) && _device_available(D)
        mover = _DeviceMover(similar(_device_ref(D()),eltype(p),0))
        m.par.nnstruct.layers = [_layer_to_device(mover,l) for l in m.par.nnstruct.layers]
        m.opt.verbosity > STD && println("Moving the neural network parameters back to the $(dname.type) GPU where the model was saved from.")
    elseif m.opt.verbosity >= STD
        @info "The model was saved from a $(dname.type) GPU, but this is not available (load the package of the GPU vendor, e.g. `using CUDA`, to use it). Using the CPU."
    end
    return nothing
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
    onfail         = m.opt.onfail
    fail_epoch_ref = m.opt.fail_epoch_ref
    fail_epoch     = m.opt.fail_epoch
    fail_threshold = m.opt.fail_threshold
    fail_attempts  = m.opt.fail_attempts
    fitted         = m.fitted

    nR,nD       = size(X)
    nRy, nDy    = size(Y,1), size(Y,2)         
    
    nR == nRy || error("X and Y have a different number of records (rows)")

    if !fitted
        if layers == nothing
            # let's see if y is continuous, all positives or all in [0,1] in order to choose the last layer accordingly
            Ycpu     = _tocpu(Y)
            allPos   = all(Ycpu .>= 0.0)
            allSum1  = all(sum(Ycpu,dims=2) .≈ 1.0)
            allProbs = allPos && allSum1  && nDy >1
            if nDy == 1 || allProbs
                innerSize = (nD <= 4) ? Int(round(nD*2)) : Int(round(nD*1.5))
            elseif nDy < 5
                innerSize = (nD <= 4) ? Int(round(nD*2*nDy/1.5)) : Int(round(nD*1.5*nDy/1.5))  
            elseif nDy < 10   
                innerSize = Int(round(nD*1.3*nDy/3)) 
            else
                innerSize = Int(round(nD*1.3*log(2,nDy))) 
            end
            #innerSize = nDy < 10 ? Int(round(nD*2)) : Int(round(nD*1.3))
            #println("nD: $nD \t nDy: $nDy")
            #println(innerSize)
            l1 = DenseLayer(nD,innerSize, f=relu, df=drelu, rng=rng)
            l2 = DenseLayer(innerSize,innerSize, f=relu, df=drelu, rng=rng)


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
            m.hpar.layers = layers
        end
        # Check that the first layer has the dimensions of X and the last layer has the output dimensions of Y
        nn_isize_tuple = size(layers[1])[1]
        #println(nn_isize_tuple)
        nn_osize_tuple = size(layers[end])[2]
        #length(nn_isize_tuple) == 1 || error("The input of a neural network should always be a single dimensional vector. Use eventually `ReshaperLayer` to reshape it to a vector.")
        length(nn_osize_tuple) == 1 || error("The last neural network layer should always return a single dimensional vector. Use eventually `ReshaperLayer` to reshape it to a vector.")
        nn_isize = nn_isize_tuple[1]
        nn_osize = nn_osize_tuple[1]

        nn_isize == nD || error("The first layer of the network must have the ndims of the input data ($nD) instead of $(nn_isize).")
        nn_osize == nDy || error("The last layer of the network must have the ndims of the output data ($nDy) instead of $(nn_osize). For classification tasks, this is normally the number of possible categories.")

        m.par = NeuralNetworkEstimator_lp(NN(deepcopy(layers),loss,dloss,false,descr))
        m.info["nepochs_ran"] = 0
        m.info["loss_per_epoch"] = Float64[]
        m.info["par_per_epoch"] = []
        m.info["xndims"]   = nD
        m.info["yndims"]   = nDy
        #m.info["fitted_records"] = O
    end


    nnstruct = m.par.nnstruct
    delete!(m.info,"device_at_save") # where to train is decided by the device option and the data
    _place_model!(nnstruct,m.opt.device,X;verbosity=verbosity)
    out = ()
    for a in 1:fail_attempts
        out = train!(nnstruct,X,Y; epochs=epochs, batch_size=batch_size, sequential=!shuffle, verbosity=verbosity, cb=cb, opt_alg=opt_alg,nepochs_ran=m.info["nepochs_ran"],onfail=onfail,fail_epoch_ref=fail_epoch_ref,fail_epoch=fail_epoch,fail_threshold=fail_threshold,rng = rng)

        out.success && break
        (onfail == "continue") && break # break as one complete training done, whatever the result
        (onfail == "stop") && break # break as one training done, whatever the result
        if a == fail_attempts
            @error "Fitting the model on the data failed. Loss is not decreasing even after $(fail_attempts) attempts."
        elseif ! m.fitted # random reset only if the model has not been fitted before
            verbosity >= STD && @info "Training attempt failed, resetting the model and trying again"
            layers = m.par.nnstruct.layers
            random_init!.(layers;rng=rng) 
        end    
    end

    m.info["nepochs_ran"]     += out.epochs
    append!(m.info["loss_per_epoch"],out.ϵ_epochs) 
    append!(m.info["par_per_epoch"],out.θ_epochs) 
    m.info["xndims"]    = nD
    m.info["fitted_records"] = nR
    m.info["nLayers"] = length(nnstruct.layers)
    m.info["nPar"] = get_nparams(m.par.nnstruct)
   
    if cache
       ŷ  = predict(nnstruct,X)
       if ndims(ŷ) > 1 && size(layers[end])[2][1] == 1
          m.cres = dropdims(ŷ,dims=2)
       else
          m.cres = ŷ
       end 
    end

    m.fitted = true
    m.par.nnstruct.trained = true

    return cache ? m.cres : nothing
end

function predict(m::NeuralNetworkEstimator,X)
    # With `device=:auto` predictions are computed where the network has been trained (moving
    # back to the GPU a model saved from there), otherwise on the device requested
    if _device_spec(m.opt.device) === :auto
        _restore_saved_device!(m)
    else
        delete!(m.info,"device_at_save")
        _place_model!(m.par.nnstruct,m.opt.device,X;verbosity=m.opt.verbosity)
    end
    ŷ        = predict(m.par.nnstruct,X)
    nn_osize = size(m.par.nnstruct.layers[end])[2][1]
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
        println(io,"NeuralNetworkEstimator - A $(length(m.hpar.layers))-layers feedforward neural network (unfitted)")
        println(io,"Loss function:")
        println(io,m.hpar.loss)
        println(io,"Optimisation algorithm:")
        println(io,m.hpar.opt_alg)
        println(io,"Layers:")
        println(io,"#\t # In \t\t # Out \t\t Type")
        for (i,l) in enumerate(m.hpar.layers)
          shapes = size(l)
          println(io,"$i \t $(shapes[1]) \t\t $(shapes[2]) \t\t $(typeof(l)) ")
        end
    else
        println(io,"NeuralNetworkEstimator - A $(m.info["xndims"])-dimensions $(m.info["nLayers"])-layers feedforward neural network (fitted on $(m.info["fitted_records"]) records)")
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
end # end module
