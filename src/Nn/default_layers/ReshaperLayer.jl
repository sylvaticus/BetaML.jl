"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."
# Experimental

"""
$(TYPEDEF)


Representation of a "reshaper" (weightless) layer in the network

Reshape the output of a layer (or the input data) to the shape needed for the next one.

# Fields:
$(TYPEDFIELDS)
"""
struct ReshaperLayer{NDIN,NDOUT} <: AbstractLayer
    "Input size"
    input_size::SVector{NDIN,Int64}
    "Output size"
    output_size::SVector{NDOUT,Int64}
    
    @doc """
    $(TYPEDSIGNATURES)

    Instantiate a new ReshaperLayer

    # Positional arguments:
    * `input_size`:    Shape of the input layer (tuple).
    * `output_size`:   Shape of the output layer (tuple) [def: `prod([input_size...])`, i.e. reshape to a vector of appropriate length].
    """
    function ReshaperLayer(input_size, output_size=prod([input_size...]))
        NDIN = length(input_size)
        if typeof(output_size) <: Integer
            output_size = (output_size,)
        end
        NDOUT = length(output_size)
        return new{NDIN,NDOUT}(input_size,output_size)
     end
end


# The batch X has the records along the last dimension

function forward_batch(layer::ReshaperLayer,X)
  return reshape(X,layer.output_size...,_nbatch(X))
end

function backward_batch(layer::ReshaperLayer,X,next_gradient)
   return reshape(next_gradient,layer.input_size...,_nbatch(next_gradient))
end

function get_params(layer::ReshaperLayer)
  return Learnable(())
end

function get_gradient_batch(layer::ReshaperLayer,X,next_gradient)
   return Learnable(())
end

function set_params!(layer::ReshaperLayer,w)
    return nothing
end
function size(layer::ReshaperLayer)
    return (layer.input_size,layer.output_size)
end