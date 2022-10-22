"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
# Provide implementation of default layers

Provided layers
- DenseLayer
- DenseNoBiasLayer
- VectorFunctionLayer
"""

using LoopVectorization

#using Random, Zygote
#import ..Utils
#import Base.size

# ------------------------------------------------------------------------------
# DenseLayer layer

"""
   DenseLayer

Representation of a layer in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct DenseLayer <: AbstractLayer
     w::Array{Float64,2}
     wb::Array{Float64,1}
     f::Function
     df::Union{Function,Nothing}
     """
        DenseLayer(f,n,nₗ;w,wb,df)

     Instantiate a new DenseLayer

     # Positional arguments:
     * `nₗ`: Number of nodes of the previous layer
     * `n`:  Number of nodes
     # Keyword arguments:
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `wb`:  Initial weigths with respect to bias [default: Xavier initialisation, dims = (n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`

     """
     function DenseLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),wb=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n),f=identity,df=nothing)
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         wb = reshape(wb,n)
         return new(w,wb,f,df)
     end
end

function _zComp(layer::DenseLayer,x)
    w  = layer.w
    wb = layer.wb
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @turbo for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        zn   += wb[n]
        z[n]  = zn
    end
    return z
end


function forward(layer::DenseLayer,x)
  z =  _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) #layer.w * x + layer.wb # _zComp(layer,x) #   layer.w * x + layer.wb # testd @avx
  return layer.f.(z)
end

function backward(layer::DenseLayer,x,next_gradient)
   z = _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) # layer.w * x + layer.wb # _zComp(layer,x) # @avx layer.w * x + layer.wb               # tested @avx
   if layer.df != nothing
       dϵ_dz = @turbo layer.df.(z) .* next_gradient # tested @avx
    else
       dϵ_dz = @turbo layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dI =  @turbo layer.w' * dϵ_dz # @avx
end

function get_params(layer::DenseLayer)
  return Learnable((layer.w,layer.wb))
end

function get_gradient(layer::DenseLayer,x,next_gradient)
   z      =  _zComp(layer,x) #@avx layer.w * x + layer.wb #  _zComp(layer,x) #layer.w * x + layer.wb # @avx
   if layer.df != nothing
       dϵ_dz = @turbo layer.df.(z) .* next_gradient
    else
       dϵ_dz = @turbo layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dw  = @turbo dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function set_params!(layer::DenseLayer,w)
   layer.w  = w.data[1]
   layer.wb = w.data[2]
end
function size(layer::DenseLayer)
    return size(layer.w')
end


# ------------------------------------------------------------------------------
# DenseNoBiasLayer layer

"""
   DenseNoBiasLayer

Representation of a layer without bias in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n x n pr. layer)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct DenseNoBiasLayer <: AbstractLayer
     w::Array{Float64,2}
     f::Function
     df::Union{Function,Nothing}
     """
        DenseNoBiasLayer(f,nₗ,n;w,df)

     Instantiate a new DenseNoBiasLayer

        DenseNoBiasLayer(f,nₗ,n;w,df)

     Instantiate a new DenseNoBiasLayer

     # Positional arguments:
     * `nₗ`:  Number of nodes of the previous layer
     * `n`:   Number of nodes
     # Keyword arguments:
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:   Activation function [def: `identity`]
     * `df`:  Derivative of the activation function [def: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ,n))`
     """
     function DenseNoBiasLayer(nₗ,n;rng = Random.GLOBAL_RNG,w=rand(rng,Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),f=identity,df=nothing)
         # To be sure w is a matrix and wb a column vector..
         w  = reshape(w,n,nₗ)
         return new(w,f,df)
     end
end

function _zComp(layer::DenseNoBiasLayer,x)
    w  = layer.w
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @turbo for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        z[n]  = zn
    end
    return z
end

function forward(layer::DenseNoBiasLayer,x)
  z = _zComp(layer,x)
  return layer.f.(z)
end

function backward(layer::DenseNoBiasLayer,x,next_gradient)
   z = _zComp(layer,x)
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* next_gradient
    else
       dϵ_dz = layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dI = layer.w' * dϵ_dz
end

function get_params(layer::DenseNoBiasLayer)
  return Learnable((layer.w,))
end

function get_gradient(layer::DenseNoBiasLayer,x,next_gradient)
   z      = _zComp(layer,x)
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* next_gradient
    else
       dϵ_dz = layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dw  =  dϵ_dz * x'
   return Learnable((dϵ_dw,))
end

function set_params!(layer::DenseNoBiasLayer,w)
   layer.w = w.data[1]
end

function size(layer::DenseNoBiasLayer)
    return size(layer.w')
end

# ------------------------------------------------------------------------------
# VectorFunctionLayer layer

"""
   VectorFunctionLayer

Representation of a VectorFunction layer in the network. Vector
function layer expects a vector activation function, i.e. a function taking the
whole output of the previous layer an input rather than working on a single node
as "normal" activation functions would do.
Useful for example with the SoftMax function in classification or with the
`pool1D` function to implement a "pool" layer in 1 dimensions.
By default it is weightless, i.e. it doesn't apply any transformation to the
output coming from the previous layer except the activation function. However,
by passing the parameter `wsize` (a touple or array - tested only 1D) you can
pass the learnable parameter to the activation function too. It is your
responsability to be sure the activation function accept only X or also this 
learnable array (as second argument).   
The number of nodes in input must be set to the same as in the previous layer
(and if you are using this for classification, to the number of classes, i.e.
the _previous_ layer must be set equal to the number of classes in the
predictions).

# Fields:
* `w`:   Weigths (parameter) array passes as second argument to the activation
         function (if not empty)
* `nₗ`:  Number of nodes in input (i.e. length of previous layer)
* `n`:   Number of nodes in output (automatically inferred in the constructor)
* `f`:   Activation function (vector)
* `dfx`: Derivative of the (vector) activation function with respect to the
         layer inputs (x)
* `dfw`: Derivative of the (vector) activation function with respect to the
         optional learnable weigths (w)         

# Notes:
* The output `size` of this layer is given by the size of the output function,
that not necessarily is the same as the previous layers.
"""
mutable struct VectorFunctionLayer{N} <: AbstractLayer
     w::Array{Float64,N}
     nₗ::Int64
     n::Int64
     f::Function
     dfx::Union{Function,Nothing}
     dfw::Union{Function,Nothing}
     """
        VectorFunctionLayer(n;rng,wsize,w,f,dfx,dfw,dummyDataToTestOutputSize)

     Instantiate a new VectorFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softMax`]
     * `dfx`: Derivative of the activation function with respect to the data
              [default: `nothing` (i.e. use AD)]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `dummyDataToTestOutputSize`: Dummy data to test the output size [def:
     `ones(nₗ)`]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - To avoid recomputing the activation function just to determine its output size,
       we compute the output size once here in the layer constructor by calling the
       activation function with `dummyDataToTestOutputSize`. Feel free to change
       it if it doesn't match with the activation function you are setting
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function VectorFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],w=rand(rng,Uniform(-sqrt(6)/sqrt(sum(wsize)),sqrt(6)/sqrt(sum(wsize))),Tuple(wsize)), f=softMax,dfx=nothing,dfw=nothing,dummyDataToTestOutputSize=ones(nₗ))
        nw = length(wsize) 
        if nw ==0 
          n  = length(f(dummyDataToTestOutputSize))
        else
         n  = length(f(dummyDataToTestOutputSize,w))
        end
        return new{nw}(w,nₗ,n,f,dfx,dfw)
     end
end

function forward(layer::VectorFunctionLayer{N},x) where {N}
  return N == 0 ? layer.f(x) : layer.f(x,layer.w)
end

function backward(layer::VectorFunctionLayer{N},x,next_gradient) where N
   if N == 0
      if layer.dfx != nothing
         dϵ_dI = layer.dfx(x)' * next_gradient
      else  # using AD
         j = autojacobian(layer.f,x; nY=layer.n)
         dϵ_dI = j' * next_gradient
      end
      return dϵ_dI
   else
      if layer.dfx != nothing
         dϵ_dI = layer.dfx(x,layer.w)' * next_gradient
      else  # using AD
        tempfunction(x) = layer.f(x,layer.w)
        nYl = layer.n
        j = autojacobian(tempfunction,x; nY=nYl)
        dϵ_dI = j' * next_gradient
      end
     return dϵ_dI
   end
end

function get_params(layer::VectorFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient(layer::VectorFunctionLayer{N},x,next_gradient) where {N}
   if N == 0
     return Learnable(()) # parameterless layer
   else
      if layer.dfw != nothing
         dϵ_dw = layer.dfw(x,layer.w)' * next_gradient
      else  # using AD
        j = autojacobian(wt -> layer.f(x,wt),layer.w; nY=layer.n)
        dϵ_dw = j' * next_gradient
      end
     return Learnable((dϵ_dw,))
   end
end

function set_params!(layer::VectorFunctionLayer{N},w) where {N}
   if N > 0
      layer.w = w.data[1]
   end
   return nothing
end
function size(layer::VectorFunctionLayer{N}) where {N}
    # Output size for the VectorFunctionLayer is given by its activation function
    # We test its length with dummy values
    return (layer.nₗ,layer.n)
end


# ------------------------------------------------------------------------------
# ScalarFunctionLayer layer

"""
   ScalarFunctionLayer

Representation of a ScalarFunction layer in the network.
ScalarFunctionLayer applies the activation function directly to the output of
the previous layer (i.e., without passing for a weigth matrix), but using an 
optional learnable parameter (an array) used as second argument, similarly to
[`VectorFunctionLayer`(@ref).
Differently from `VectorFunctionLayer`, the function is applied scalarwise to
each node. 
 
The number of nodes in input must be set to the same as in the previous layer


# Fields:
* `w`:   Weigths (parameter) array passes as second argument to the activation
         function (if not empty)
* `n`:   Number of nodes in output (≡ number of nodes in input )
* `f`:   Activation function (vector)
* `dfx`: Derivative of the (vector) activation function with respect to the
         layer inputs (x)
* `dfw`: Derivative of the (vector) activation function with respect to the
         optional learnable weigths (w)         

# Notes:
* The output `size` of this layer is the same as those of the previous layers.
"""
mutable struct ScalarFunctionLayer{N} <: AbstractLayer
     w::Array{Float64,N}
     n::Int64
     f::Function
     dfx::Union{Function,Nothing}
     dfw::Union{Function,Nothing}
     """
        ScalarFunctionLayer(n;rng,wsize,w,f,dfx,dfw)

     Instantiate a new ScalarFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softMax`]
     * `dfx`: Derivative of the activation function with respect to the data
              [default: `nothing` (i.e. use AD)]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function ScalarFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],w=rand(rng,Uniform(-sqrt(6)/sqrt(sum(wsize)),sqrt(6)/sqrt(sum(wsize))),Tuple(wsize)), f=softMax,dfx=nothing,dfw=nothing)
        nw = length(wsize) 
        return new{nw}(w,nₗ,f,dfx,dfw)
     end
end

function forward(layer::ScalarFunctionLayer{N},x) where {N}
  return N == 0 ? layer.f.(x) : layer.f.(x,layer.w)
end

function backward(layer::ScalarFunctionLayer{N},x,next_gradient) where N
   if N == 0
      if layer.dfx != nothing
         dϵ_dI = layer.dfx.(x) .* next_gradient
      else  # using AD
         dϵ_dI = layer.f'.(x) .* next_gradient
      end
      return dϵ_dI
   else
      if layer.dfx != nothing
         dϵ_dI = layer.dfx.(x,Ref(layer.w)) .* next_gradient
      else  # using AD
        #tempfunction(x) = layer.f.(x,Ref(layer.w))
        df_dx = [gradient(xt -> layer.f(xt,layer.w),xi)[1] for xi in x]  
        dϵ_dI = df_dx .* next_gradient 
      end
     return dϵ_dI
   end
end

function get_params(layer::ScalarFunctionLayer{N}) where {N}
   return N == 0 ? Learnable(()) : Learnable((layer.w,))
end

function get_gradient(layer::ScalarFunctionLayer{N},x,next_gradient) where {N}
   if N == 0
     return Learnable(()) # parameterless layer
   else
      if layer.dfw != nothing
         dϵ_dw = [layer.dfw(xi,wj) for wj in layer.w, xi in x] * next_gradient
      else  # using AD
        tempfunction(w) = [layer.f(xi,w) for xi in x]
        j = autojacobian(tempfunction,layer.w; nY=layer.n)
        dϵ_dw = j' * next_gradient
      end
     return Learnable((dϵ_dw,))
   end
end

function set_params!(layer::ScalarFunctionLayer{N},w) where {N}
   if N > 0
      layer.w = w.data[1]
   end
   return nothing
end
function size(layer::ScalarFunctionLayer{N}) where {N}
    # Output size for the ScalarFunctionLayer is given by its activation function
    # We test its length with dummy values
    return (layer.n,layer.n)
end



# ------------------------------------------------------------------------------
# RNN layer

"""
   RNNLayer

Representation of a layer in the network

# Fields:
* `wx`: Weigths matrix with respect to the input from data (n by n_input)
* `ws`: Weigths matrix with respect to the layer state (n x n )
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
* `s` : State
"""
mutable struct RNNLayer <: RecursiveLayer
     wx::Array{Float64,2}
     ws::Array{Float64,2}
     wb::Array{Float64,1}
     s::Array{Float64,1}
     f::Function
     df::Union{Function,Nothing}
     """
        RNNLayer(nₗ,n;f,wx,ws,wb,df)

     Instantiate a new RNNLayer

     # Positional arguments:
     * `nₗ`: Number of nodes of the input
     * `n`:  Number of nodes of the state (and the output)
     # Keyword arguments:
     * `wx`:  Initial weigths with respect to input [default: Xavier initialisation, dims = (n,nₗ)]
     * `ws`:  Initial weigths with respect to input [default: Xavier initialisation, dims = (n,n)]
     * `wb`:  Initial weigths with respect to bias [default: Xavier initialisation, dims = (n)]
     * `s`:   Initial states [def: zeros(n)]
     * `f`:   Activation function [def: `relu`]
     * `df`:  Derivative of the activation function [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

     # Notes:
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n))`

     """
     function RNNLayer(nₗ,n;rng = Random.GLOBAL_RNG,wx=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n,nₗ),ws=rand(rng, Uniform(-sqrt(6)/sqrt(n+n),sqrt(6)/sqrt(n+n)),n,n),wb=rand(rng, Uniform(-sqrt(6)/sqrt(nₗ+n),sqrt(6)/sqrt(nₗ+n)),n),s=zeros(n),f=relu,df=nothing)
         # To be sure w is a matrix and wb a column vector..
         wx  = reshape(w,n,nₗ)
         ws  = reshape(w,n,n)
         wb  = reshape(wb,n)
         s   = reshape(s,n)
         return new(w,wb,s,f,df)
     end
end
#=
function _zComp(layer::RNNLayer,x)
    w  = layer.w
    wb = layer.wb
    z  = zeros(eltype(x),size(w,1))
    @inbounds for n in axes(w,1)
        zn = zero(eltype(x))
        @simd for nl in axes(x,1)
            zn += w[n,nl] * x[nl]
        end
        zn   += wb[n]
        z[n]  = zn
    end
    return z
end
=#

function forward(layer::RNNLayer,x)
  #z =  _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) #layer.w * x + layer.wb # _zComp(layer,x) #   layer.w * x + layer.wb # testd @avx
  z =  layer.wb + layer.wx * x + layer.ws * s 
  return layer.f.(z)
end

function backward(layer::RNNLayer,x,next_gradient) #TODO
   z = _zComp(layer,x) #@avx layer.w * x + layer.wb #_zComp(layer,x) # layer.w * x + layer.wb # _zComp(layer,x) # @avx layer.w * x + layer.wb               # tested @avx
   if layer.df != nothing
       dϵ_dz =  layer.df.(z) .* next_gradient # tested @avx
    else
       dϵ_dz = layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dI =  layer.w' * dϵ_dz # @avx
end

function get_params(layer::RNNLayer)
  return Learnable((layer.wb,layer.wx,layer.ws))
end

function get_gradient(layer::RNNLayer,x,next_gradient) #TODO
   z      =  _zComp(layer,x) #@avx layer.w * x + layer.wb #  _zComp(layer,x) #layer.w * x + layer.wb # @avx
   if layer.df != nothing
       dϵ_dz = layer.df.(z) .* next_gradient
    else
       dϵ_dz = layer.f'.(z) .* next_gradient # using AD
    end
   dϵ_dw  = dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
end

function set_params!(layer::RNNLayer,w)
   layer.wb = w.data[1]
   layer.wx = w.data[2]
   layer.ws = w.data[3]
end
function size(layer::RNNLayer)
    return size(layer.w')
end