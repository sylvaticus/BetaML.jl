"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

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
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new VectorFunctionLayer

     # Positional arguments:
     * `nₗ`: Number of nodes (must be same as in the previous layer)
     # Keyword arguments:
     * `wsize`: A tuple or array specifying the size (number of elements) of the
       learnable parameter [def: empty array]
     * `w`:   Initial weigths with respect to input [default: Xavier initialisation, dims = (nₗ,n)]
     * `f`:  Activation function [def: `softmax`]
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
     function VectorFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],w=rand(rng,Uniform(-sqrt(6)/sqrt(sum(wsize)),sqrt(6)/sqrt(sum(wsize))),Tuple(wsize)), f=softmax,dfx=match_known_derivatives(f),dfw=nothing,dummyDataToTestOutputSize=ones(nₗ))
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
         dfz = layer.dfx(x)'
      else  # using AD
         dfz = (autojacobian(layer.f,x; nY=layer.n))'
      end
      dϵ_dI =  @turbo dfz * next_gradient
      return dϵ_dI
   else
      if layer.dfx != nothing
         dfz = layer.dfx(x,layer.w)'
      else  # using AD
        tempfunction(x) = layer.f(x,layer.w)
        nYl = layer.n
        dfz = (autojacobian(tempfunction,x; nY=nYl))'
      end
      dϵ_dI = @turbo dfz * next_gradient
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
         dfz = layer.dfw(x,layer.w)'
      else  # using AD
        dfz = (autojacobian(wt -> layer.f(x,wt),layer.w; nY=layer.n))'
      end
      dϵ_dw = @turbo dfz * next_gradient
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
    return ((layer.nₗ,),(layer.n,))
end
