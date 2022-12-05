"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


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
     * `f`:  Activation function [def: `softmax`]
     * `dfx`: Derivative of the activation function with respect to the data
              [default: `nothing` (i.e. use AD)]
     * `dfw`: Derivative of the activation function with respect to the
              learnable parameter [default: `nothing` (i.e. use AD)]
     * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
     # Notes:
     - If the derivative is provided, it should return the gradient as a (n,n) matrix (i.e. the Jacobian)
     - Xavier initialization = `rand(Uniform(-sqrt(6)/sqrt(sum(wsize...)),sqrt(6)/sqrt(sum(wsize...))))`
     """
     function ScalarFunctionLayer(nₗ;rng = Random.GLOBAL_RNG,wsize=Int64[],w=rand(rng,Uniform(-sqrt(6)/sqrt(sum(wsize)),sqrt(6)/sqrt(sum(wsize))),Tuple(wsize)), f=softmax,dfx=match_known_derivatives(f),dfw=nothing)
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
         dfz = layer.dfx.(x)
      else  # using AD
         dfz = layer.f'.(x)
      end
      dϵ_dI = @turbo dfz  .* next_gradient
      return dϵ_dI
   else
      if layer.dfx != nothing
         df_dx = layer.dfx.(x,Ref(layer.w))
      else  # using AD
        #tempfunction(x) = layer.f.(x,Ref(layer.w))
        df_dx = [gradient(xt -> layer.f(xt,layer.w),xi)[1] for xi in x]  
      end
      dϵ_dI = @turbo df_dx .* next_gradient
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
         dfz = [layer.dfw(xi,wj) for wj in layer.w, xi in x] 
         
      else  # using AD
        tempfunction(w) = [layer.f(xi,w) for xi in x]
        dfz = (autojacobian(tempfunction,layer.w; nY=layer.n))'
      end
      dϵ_dw = @turbo dfz * next_gradient
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
    return ((layer.n,),(layer.n,))
end
