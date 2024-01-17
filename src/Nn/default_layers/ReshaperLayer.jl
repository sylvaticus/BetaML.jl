"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."
# Experimental

"""
$(TYPEDEF)


Representation of a "reshaper" (weigthless) layer in the network

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
    * `output_size`:   Shape of the input layer (tuple) [def: `prod([input_size...]))`, i.e. reshape to a vector of appropriate lenght].
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


function forward(layer::ReshaperLayer,x)
  return reshape(x,layer.output_size...)
end

function backward(layer::ReshaperLayer,x,next_gradient)
   return reshape(next_gradient,layer.input_size...)
end

function get_params(layer::ReshaperLayer)
  return Learnable(())
end

function get_gradient(layer::ReshaperLayer,x,next_gradient)
   return Learnable(())
end

function set_params!(layer::ReshaperLayer,w)
    return nothing
end
function size(layer::ReshaperLayer)
    return (layer.input_size,layer.output_size)
end