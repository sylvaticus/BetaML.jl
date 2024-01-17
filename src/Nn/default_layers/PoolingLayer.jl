"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Experimental. Works well, but still too slow for any practical application.

"""
$(TYPEDEF)

Representation of a pooling layer in the network

# Fields:
$(TYPEDFIELDS)
"""
struct PoolingLayer{ND,NDPLUS1,NDPLUS2,TF <: Function, TDF <: Union{Nothing,Function}} <: AbstractLayer
   "Input size (including nchannel_in as last dimension)"
   input_size::SVector{NDPLUS1,Int64}
   "Output size (including nchannel_out as last dimension)"
   output_size::SVector{NDPLUS1,Int64}
   "kernel_size augmented by the nchannels_in and nchannels_out dimensions"
   kernel_size::SVector{NDPLUS2,Int64}
   "Padding (initial)"
   padding_start::SVector{ND,Int64}
   "Padding (ending)"
   padding_end::SVector{ND,Int64}
   "Stride"
   stride::SVector{ND,Int64}
   "Number of dimensions (excluding input and output channels)"
   ndims::Int64
   "Activation function"
   f::TF
   "Derivative of the activation function"
   df::TDF #Union{Function,Nothing}

   "x ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   #x_ids::Array{NTuple{NDPLUS1,Int32},1}
   "y ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   #y_ids::Array{NTuple{NDPLUS1,Int32},1}
   "w ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   #w_ids::Array{NTuple{NDPLUS2,Int32},1}

   "A x-dims array of vectors of ids of y reached by the given x"
   #x_to_y_ids::Array{Vector{NTuple{NDPLUS1,Int32}},NDPLUS1} # not needed
   "A y-dims array of vectors of ids of x(s) contributing to the giving y"
   y_to_x_ids::Array{Vector{NTuple{NDPLUS1,Int32}},NDPLUS1}
   


   @doc """
   $(TYPEDSIGNATURES)

   Instantiate a new nD-dimensional, possibly multichannel PoolingLayer

   The input data is either a column vector (in which case is reshaped) or an array of `input_size` augmented by the `n_channels` dimension, the output size depends on the `input_size`, `kernel_size`, `padding` and `striding` but has always `nchannels_out` as its last dimention.  

   # Positional arguments:
   * `input_size`:    Shape of the input layer (integer for 1D convolution, tuple otherwise). Do not consider the channels number here.
   * `kernel_size`:   Size of the kernel (aka filter) (integer for 1D or hypercube kernels or nD-sized tuple for assymmetric kernels). Do not consider the channels number here.
   * `nchannels_in`:  Number of channels in input
   * `nchannels_out`: Number of channels in output
   
   # Keyword arguments:
   * `stride`: "Steps" to move the convolution with across the various tensor dimensions [def: `kernel_size`, i.e. each X contributes to a single y]
   * `padding`: Integer or 2-elements tuple of tuples of the starting end ending padding across the various dimensions [def: `nothing`, i.e. set the padding required to keep out_side = in_side / stride ]
   * `f`:   Activation function. It should have a vector as input and produce a scalar as output[def: `maximum`]
   * `df`:  Derivative (gradient) of the activation function for the various inputs. [default: `nothing` (i.e. use AD)]


   # Notes:
   - to retrieve the output size of the layer, use `size(PoolLayer[2])`. The output size on each dimension _d_ (except the last one that is given by `nchannels_out`) is given by the following formula (ceiled): `output_size[d] = 1 + (input_size[d]+2*padding[d]-kernel_size[d])/stride[d]`
   - differently from a ConvLayer, the pooling applies always on a single channel level, so that the output has always the same number of channels of the input. If you want to reduce the channels number either use a `ConvLayer` with the desired number of channels in output or use a `ReghaperLayer` to add a 1-element further dimension that will be treated as "channel" and choose the desided stride for the last pooling dimension (the one that was originally the channel dimension) 
   """
   function PoolingLayer(input_size,kernel_size,nchannels_in;
            stride  = kernel_size,
            padding = nothing, # (zeros(Int64,length(input_size)),zeros(Int64,length(input_size))),
            f       = maximum,
            df      = match_known_derivatives(f))
      
      nchannels_out = nchannels_in
      # be sure all are tuples of right dimension...
      if typeof(input_size) <: Integer
         input_size = (input_size,)
      end
      nD = length(input_size)
      if typeof(kernel_size) <: Integer
         kernel_size = ([kernel_size for d in 1:nD]...,)
      end
      length(input_size) == length(kernel_size) || error("Number of dimensions of the kernel must equate number of dimensions of input data")
      if typeof(stride) <: Integer
         stride = ([stride for d in 1:nD]...,)
      end
      if typeof(padding) <: Integer
         padding_start = ([padding for d in 1:nD]...,)
         padding_end   = ([padding for d in 1:nD]...,)
      elseif isnothing(padding) # compute padding to keep same size/stride if not provided
         target_out_size = [Int(ceil(input_size[d]/stride[d])) for d in 1:length(input_size)]
         padding_total   = [(target_out_size[d]-1)*stride[d] - input_size[d]+kernel_size[d] for d in 1:length(input_size)]
         padding_start   =  Int.(ceil.(padding_total ./ 2))
         padding_end     =  padding_total .- padding_start  
      else
         padding_start = padding[1]
         padding_end   = padding[2]
      end
      nD == length(stride) || error("`stride` must be either a scalar or a tuple that equates the number of dimensions of input data")
      nD == length(padding_start) == length(padding_end) || error("`padding` must be: (a) the value `nothing` for automatic computation to keep same size, (b) a scalar for same padding on all dimensions or (c) a 2-elements tuple where each elements are tuples that equate the number of dimensions of input data for indicating the padding to set in front of the data and the padding to set at the ending of the data")

      #println(typeof(weight_init))
      #println(weight_init)

      #new{nD,nD+1}(weight_init,usebias,bias_init,padding,stride,nD,f,df)
      input_size_with_nchin  = (input_size...,nchannels_in)
      kernel_size_with_nchin_nchout = (kernel_size...,nchannels_in, nchannels_out)

      # Computation of out_size. Code from size() that we can't yet use
      output_size_with_nchout = ([1 + Int(floor((input_size[d]+padding_start[d]+padding_end[d]-kernel_size[d])/stride[d])) for d in 1:nD]...,nchannels_out)
      
      #x_to_y_ids = [Vector{NTuple{nD+1,Int32}}() for i in CartesianIndices(input_size_with_nchin)] # not needed
      y_to_x_ids = [Vector{NTuple{nD+1,Int32}}() for i in CartesianIndices(output_size_with_nchout)]

      new{nD,nD+1,nD+2,typeof(f),typeof(df)}(input_size_with_nchin,output_size_with_nchout,kernel_size_with_nchin_nchout,padding_start,padding_end,stride,nD,f,df,y_to_x_ids)
   end
end

"""
$(TYPEDSIGNATURES)

Alternative constructor for a `PoolingLayer` where the number of channels in input is specified as a further dimension in the input size instead of as a separate parameter, so to use `size(previous_layer)[2]` if one wish.

For arguments and default values see the documentation of the main constructor.
"""
function PoolingLayer(input_size_with_channel,kernel_size;
     stride  = kernel_size,
     padding = nothing, # (zeros(Int64,length(input_size)),zeros(Int64,length(input_size))),
     f       = maximum,
     df      = match_known_derivatives(f))
     return PoolingLayer(input_size_with_channel[1:end-1],kernel_size,input_size_with_channel[end]; stride=stride,padding=padding,f=f,df=df)

end


function preprocess!(layer::PoolingLayer{ND,NDPLUS1,NDPLUS2}) where {ND,NDPLUS1,NDPLUS2}
   if layer.y_to_x_ids !=  [Vector{NTuple{NDPLUS1,Int32}}() for i in CartesianIndices((layer.output_size...,))]
      return # layer already prepocessed
   end

   input_size, output_size = size(layer)
   nchannels_in  = input_size[end]
   convsize      = input_size[1:end-1]
   ndims_conv    = ND

   ksize = layer.kernel_size
   ysize = output_size
   #println(ysize)
   
   # preallocating temp variables
   k_idx               = Array{Int32,1}(undef,NDPLUS2)
   y_idx               = Array{Int32,1}(undef,NDPLUS1)
   k_idx_conv          = Array{Int32,1}(undef,ND)
   y_idx_conv          = Array{Int32,1}(undef,ND)
   idx_x_source_padded = Array{Int32,1}(undef,ND)
   checkstart          = Array{Bool,1}(undef,ND)
   checkend            = Array{Bool,1}(undef,ND)
   x_idx               = Array{Int32,1}(undef,NDPLUS1)

   @inbounds for nch_in in 1:nchannels_in
      #println("Processing in layer :", nch_in)
      #@inbounds for nch_out in 1:nchannels_out
         #println("- processing out layer :", nch_out)
         @inbounds for k_idx_conv in CartesianIndices( ((ksize[1:end-2]...),) ) 
            k_idx_conv = Tuple(k_idx_conv)
            k_idx = (k_idx_conv...,nch_in,nch_in)
            @inbounds for y_idx_conv in CartesianIndices( ((ysize[1:end-1]...),) )
               y_idx_conv = Tuple(y_idx_conv)
               y_idx      = (y_idx_conv...,nch_in)
               #println("y_idx: ",y_idx)
               #println("k_idx: ",k_idx)
               #println("layer.stride: ",layer.stride)
               #quit(1)
               check = true
               @inbounds for d in 1:ndims_conv
                  idx_x_source_padded[d] = k_idx_conv[d] + (y_idx_conv[d] - 1 ) * layer.stride[d]
                  checkstart[d] =  idx_x_source_padded[d] > layer.padding_start[d]
                  checkend[d]   =  idx_x_source_padded[d] <=  layer.padding_start[d] .+ convsize[d]
                  checkstart[d] && checkend[d] || begin check = false; break; end
               end
               check || continue
               
               @inbounds @simd  for d in 1:ndims_conv
                  x_idx[d] = idx_x_source_padded[d] - layer.padding_start[d]
               end
               x_idx[ndims_conv+1] = nch_in
               #println()
               push!(layer.y_to_x_ids[y_idx...],((x_idx...,)))
               #println("---")
               #println("x_idx: ", x_idx)
               #println("w_idx: ", w_idx)
               #println("y_idx: ", y_idx)
               #push!(layer.x_ids,((x_idx...,)))
               #push!(layer.w_ids,w_idx)
               #push!(layer.y_ids,y_idx)
               #de_dx_ch_in[idx_x_source...] += dϵ_dz_ch_out[dy_idx...] * w_ch_in_out[w_idx...]
            
            end
         end
      #end
   end
end


"""
$(TYPEDSIGNATURES)

Compute forward pass of a ConvLayer

"""
function forward(layer::PoolingLayer,x)
   
   _, output_size = size(layer)
   y    = zeros(output_size)
   for y_idx in CartesianIndices(y)
      y_idx       = Tuple(y_idx)
      x_ids       = layer.y_to_x_ids[y_idx...]
      x_vals      = [x[idx...] for idx in x_ids]
      #println(x_vals)
      #println(layer.f(x_vals))
      y[y_idx...] = layer.f(x_vals)
   end
   return y
end

function _zComp!(z,layer::PoolingLayer{ND,NDPLUS1,NDPLUS2},x) where {ND,NDPLUS1,NDPLUS2}
   for y_idx in CartesianIndices(z)
      y_idx       = Tuple(y_idx)
      x_ids       = layer.y_to_x_ids[y_idx...]
      x_vals      = [x[idx...] for idx in x_ids]
      #println(x_vals)
      #println(layer.f(x_vals))
      y[y_idx...] = layer.f(x_vals)
   end

end

function backward(layer::PoolingLayer{ND,NDPLUS1,NDPLUS2},x, next_gradient) where {ND,NDPLUS1,NDPLUS2}
   de_dx     = zeros(layer.input_size...)
   for y_idx in CartesianIndices(next_gradient)
      #println("----")
      x_ids       = layer.y_to_x_ids[y_idx]
      x_vals      = [x[idx...] for idx in x_ids]
      df_val      = layer.df(x_vals)
      #println("y_idx: ",y_idx)
      #println("x_idx: ",x_ids)
      #println("x_vals: ",x_vals)
      #println("df_val: ",df_val) 
      #println("next_gradient[y_idx]: ",next_gradient[y_idx]) 
      for (i,x_idx) in enumerate(x_ids)
         #println("- x_idx: ", x_idx)
         de_dx[x_idx...] += next_gradient[y_idx] .* df_val[i]
      end
   end
   return de_dx
end


function get_params(layer::PoolingLayer)
    return Learnable(())
end


function get_gradient(layer::PoolingLayer{ND,NDPLUS1,NDPLUS2},x, next_gradient) where {ND,NDPLUS1,NDPLUS2}
   return Learnable(()) 
end

function set_params!(layer::PoolingLayer,w)

end

"""
$(TYPEDSIGNATURES)

Get the dimensions of the layers in terms of (dimensions in input, dimensions in output) including channels as last dimension
"""
function size(layer::PoolingLayer{ND,NDPLUS1,NDPLUS2}) where {ND,NDPLUS1,NDPLUS2}
   return ((layer.input_size...,),(layer.output_size...,))
end