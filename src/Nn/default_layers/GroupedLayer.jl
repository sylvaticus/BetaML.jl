"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
   GroupedLayer

Representation of a "group" of layers, each of which operates on different inputs (features) and acting as a single layer in the network.

# Fields:
- `layers`: The individual layers that compose this grouped layer

"""
mutable struct GroupedLayer <: AbstractLayer
     layers::Array{AbstractLayer,1}
     """
        GroupedLayer(layers)

     Instantiate a new GroupedLayer

     # Positional arguments:
     - `layers`: The individual layers that compose this grouped layer


     # Notes:
     - used to create composable neural networks with multiple branches
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
   #isizes_swapped = vcat(1,isizes)
   #return  vcat([forward(layer.layers[i],selectdim(x,1,sum(isizes_swapped[1:i]): sum(isizes_swapped[1:i+1])-1)) for i in 1:nL]...)
   isizes_swapped = vcat(0,isizes)
   return  vcat([forward(layer.layers[i],selectdim(x,1,sum(isizes_swapped[1:i])+1: sum(isizes_swapped[1:i+1]))) for i in 1:nL]...)


end

function backward(layer::GroupedLayer,x,next_gradient)
   nL = length(layer.layers)
   #isizes = [size(l)[1][1] for l in layer.layers] 
   #isizes_swapped = vcat(1,isizes)
   #osizes = [size(l)[2][1] for l in layer.layers]
   #osizes_swapped = vcat(1,osizes)
   #return  vcat([backward(layer.layers[i], # todo: attention here if first layer has zero paraemters !
   #     selectdim(x,1,sum(isizes_swapped[1:i]): sum(isizes_swapped[1:i+1])-1),
   #     selectdim(next_gradient,1,sum(osizes_swapped[1:i]): sum(osizes_swapped[1:i+1])-1)
   #   ) for i in 1:nL]) # dϵ_dI

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
   #isizes = [size(l)[1][1] for l in layer.layers] 
   #isizes_swapped = vcat(1,isizes)
   #osizes = [size(l)[2][1] for l in layer.layers]
   #osizes_swapped = vcat(1,osizes)
   #return  [get_gradient(layer.layers[i],
   #     selectdim(x,1,sum(isizes_swapped[1:i]): sum(isizes_swapped[1:i+1])-1),
   #     selectdim(next_gradient,1,sum(osizes_swapped[1:i]): sum(osizes_swapped[1:i+1])-1)
   #   ) for i in 1:nL] # [dϵ_dw]
   isizes = [size(l)[1][1] for l in layer.layers] 
   isizes_swapped = vcat(0,isizes)
   osizes = [size(l)[2][1] for l in layer.layers]
   osizes_swapped = vcat(0,osizes)
   return  Learnable((vcat([ [get_gradient(layer.layers[i],
         selectdim(x,1,sum(isizes_swapped[1:i])+1: sum(isizes_swapped[1:i+1])),
         selectdim(next_gradient,1,sum(osizes_swapped[1:i])+1: sum(osizes_swapped[1:i+1]))
      ).data...] for i in 1:nL]...)...,)) # [dϵ_dw]
   #return  [get_gradient(layer.layers[i],
   #   selectdim(x,1,sum(isizes_swapped[1:i])+1: sum(isizes_swapped[1:i+1])),
   #   selectdim(next_gradient,1,sum(osizes_swapped[1:i])+1: sum(osizes_swapped[1:i+1]))
   #).data for i in 1:nL] # [dϵ_dw]

end

function set_params!(layer::GroupedLayer,w)
   nWs            = _get_n_layers_weights(layer)
   nWs_swapped    = vcat(0,nWs)
   nL             = length(layer.layers)
   for i in 1:length(layer.layers)
      #set_params!(layer.layers[i],Learnable((w.data[sum(nWs_swapped[1:i]):sum(nWs_swapped[1:i+1])-1)...,))
      #println(w.data[sum(nWs_swapped[1:i])+1:sum(nWs_swapped[1:i+1])])
      set_params!(layer.layers[i],Learnable(w.data[sum(nWs_swapped[1:i])+1:sum(nWs_swapped[1:i+1])]))
   end
   #[set_params!(layer.layers[i],Learnable(w.data[i])) for i in 1:length(layer.layers)]
end

function size(layer::GroupedLayer)
   isize = sum([size(l)[1][1] for l in layer.layers]) 
   osize = sum([size(l)[2][1] for l in layer.layers])
   return ((isize,),(osize,))
end