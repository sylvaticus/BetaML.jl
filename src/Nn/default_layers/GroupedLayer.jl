"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a "group" of layers, each of which operates on different inputs (features) and acting as a single layer in the network.

# Fields:
- `layers`: The individual layers that compose this grouped layer

"""
mutable struct GroupedLayer <: AbstractLayer
     layers::Array{AbstractLayer,1}
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new GroupedLayer, a layer made up of several other layers stacked together in order to cover all the data dimensions but without connect all the inputs to all the outputs like a single `DenseLayer` would do.

     # Positional arguments:
     - `layers`: The individual layers that compose this grouped layer

     # Notes:
     - can be used to create composable neural networks with multiple branches
     - tested only with 1 dimensional layers. For convolutional networks use ReshaperLayers before and/or after.
     """
     function GroupedLayer(layers)
         return new(layers)
     end
end

function _get_n_layers_weights(layer::GroupedLayer)
      return [length(get_params(l).data)  for l in layer.layers ]
end

function forward(layer::GroupedLayer,x)
   nL = length(layer.layers)
   isizes = [size(l)[1][1] for l in layer.layers] 
   isizes_swapped = vcat(0,isizes)
   return  vcat([forward(layer.layers[i],selectdim(x,1,sum(isizes_swapped[1:i])+1: sum(isizes_swapped[1:i+1]))) for i in 1:nL]...)
end

function backward(layer::GroupedLayer,x,next_gradient)
   nL = length(layer.layers)
   isizes = [size(l)[1][1] for l in layer.layers] 
   isizes_swapped = vcat(0,isizes)
   osizes = [size(l)[2][1] for l in layer.layers]
   osizes_swapped = vcat(0,osizes)
   return  vcat([backward(layer.layers[i], # todo: attention here if first layer has zero paraemters !
        selectdim(x,1,sum(isizes_swapped[1:i])+1 : sum(isizes_swapped[1:i+1])),
        selectdim(next_gradient,1,sum(osizes_swapped[1:i])+1: sum(osizes_swapped[1:i+1]))
      ) for i in 1:nL]...) # dϵ_dI   
end

function get_params(layer::GroupedLayer)
  return Learnable((vcat([ [get_params(l).data...] for l in layer.layers]...)...,))
end

function get_gradient(layer::GroupedLayer,x,next_gradient)
   nL = length(layer.layers)
   isizes = [size(l)[1][1] for l in layer.layers] 
   isizes_swapped = vcat(0,isizes)
   osizes = [size(l)[2][1] for l in layer.layers]
   osizes_swapped = vcat(0,osizes)
   return  Learnable((vcat([ [get_gradient(layer.layers[i],
         selectdim(x,1,sum(isizes_swapped[1:i])+1: sum(isizes_swapped[1:i+1])),
         selectdim(next_gradient,1,sum(osizes_swapped[1:i])+1: sum(osizes_swapped[1:i+1]))
      ).data...] for i in 1:nL]...)...,)) # [dϵ_dw]
end

function set_params!(layer::GroupedLayer,w)
   nWs            = _get_n_layers_weights(layer)
   nWs_swapped    = vcat(0,nWs)
   nL             = length(layer.layers)
   for i in 1:length(layer.layers)
      set_params!(layer.layers[i],Learnable(w.data[sum(nWs_swapped[1:i])+1:sum(nWs_swapped[1:i+1])]))
   end
end

function size(layer::GroupedLayer)
   isize = sum([size(l)[1][1] for l in layer.layers]) 
   osize = sum([size(l)[2][1] for l in layer.layers])
   return ((isize,),(osize,))
end