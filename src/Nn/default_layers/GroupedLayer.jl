"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."


"""
$(TYPEDEF)

Representation of a "group" of layers, each of which operates on different inputs (features), acting as a single layer in the network.

# Fields:
- `layers`: The individual layers that compose this grouped layer

"""
struct GroupedLayer <: AbstractLayer
     layers::Array{AbstractLayer,1}
     @doc """
     $(TYPEDSIGNATURES)

     Instantiate a new GroupedLayer, a layer made up of several other layers stacked together in order to cover all the data dimensions but without connecting all the inputs to all the outputs like a single `DenseLayer` would do.

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

# The batch X has the records along the columns. Each inner layer receives the rows (features)
# of its own input and the outputs of the inner layers are stacked along the rows.

"Ranges of the rows of the input (`dim=1`) or output (`dim=2`) of each inner layer"
function _grouped_ranges(layer::GroupedLayer,dim)
   sizes = [size(l)[dim][1] for l in layer.layers]
   ends  = cumsum(sizes)
   return [e-s+1:e for (s,e) in zip(sizes,ends)]
end

function forward_batch(layer::GroupedLayer,X)
   X = _asmatrix(X)
   return vcat([forward_batch(l,selectdim(X,1,r)) for (l,r) in zip(layer.layers,_grouped_ranges(layer,1))]...)
end

function backward_batch(layer::GroupedLayer,X,next_gradient)
   X  = _asmatrix(X)
   next_gradient = _asmatrix(next_gradient)
   return vcat([backward_batch(l,selectdim(X,1,ri),selectdim(next_gradient,1,ro))
                for (l,ri,ro) in zip(layer.layers,_grouped_ranges(layer,1),_grouped_ranges(layer,2))]...) # dϵ_dI
end

function get_params(layer::GroupedLayer)
  return Learnable((vcat([ [get_params(l).data...] for l in layer.layers]...)...,))
end

function get_gradient_batch(layer::GroupedLayer,X,next_gradient)
   X  = _asmatrix(X)
   next_gradient = _asmatrix(next_gradient)
   return Learnable((vcat([ [get_gradient_batch(l,selectdim(X,1,ri),selectdim(next_gradient,1,ro)).data...]
                for (l,ri,ro) in zip(layer.layers,_grouped_ranges(layer,1),_grouped_ranges(layer,2))]...)...,)) # [dϵ_dw]
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