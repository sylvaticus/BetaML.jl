"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Experimental

"""
$(TYPEDEF)

Representation of a convolutional layer in the network

# Fields:
$(TYPEDFIELDS)
"""
mutable struct ConvLayer{ND,NDPLUS1,NDPLUS2} <: AbstractLayer
   "Input size (including nchannel_in as last dimension)"
   input_size::SVector{NDPLUS1,Int64}
   "Weight tensor (aka \"filter\" or \"kernel\") with respect to the input from previous layer or data (kernel_size array augmented by the nchannels_in and nchannels_out dimensions)"
   weight::Array{Float64,NDPLUS2}
   "Wether to use (and learn) a bias weigth [def: true]"
   usebias::Bool
   "Bias (nchannels_out array)"
   bias::Array{Float64,1}
   "Padding (initial)"
   padding_start::SVector{ND,Int64}
   "Padding (ending)"
   padding_end::SVector{ND,Int64}
   "Stride"
   stride::SVector{ND,Int64}
   "Number of dimensions (excluding input and output channels)"
   ndims::Int64
   "Activation function"
   f::Function
   "Derivative of the activation function"
   df::Union{Function,Nothing}

   "x ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   x_ids::Array{NTuple{NDPLUS1,Int32},1}
   "y ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   y_ids::Array{NTuple{NDPLUS1,Int32},1}
   "w ids of the convolution (computed in `preprocessing`` - itself at the beginning of `train`"
   w_ids::Array{NTuple{NDPLUS2,Int32},1}


   @doc """
   $(TYPEDSIGNATURES)

   Instantiate a new nD-dimensional, possibly multichannel ConvolutionalLayer

   The input data is either a column vector (in which case is reshaped) or an array of `input_size` augmented by the `n_channels` dimension, the output size depends on the `input_size`, `kernel_size`, `padding` and `striding` but has always `nchannels_out` as its last dimention.  

   # Positional arguments:
   * `input_size`:    Shape of the input layer (integer for 1D convolution, tuple otherwise). Do not consider the channels number here.
   * `kernel_size`:   Size of the kernel (aka filter or learnable weights) (integer for 1D or hypercube kernels or nD-sized tuple for assymmetric kernels). Do not consider the channels number here.
   * `nchannels_in`:  Number of channels in input
   * `nchannels_out`: Number of channels in output
   # Keyword arguments:
   * `stride`: "Steps" to move the convolution with across the various tensor dimensions [def: `ones`]
   * `padding`: Integer or 2-elements tuple of tuples of the starting end ending padding across the various dimensions [def: `nothing`, i.e. set the padding required to keep the same dimensions in output (with stride==1)]
   * `f`:   Activation function [def: `relu`]
   * `df`:  Derivative of the activation function [default: try to match a known funcion, AD otherwise. Use `nothing` to force AD]
   * `kernel_init`:   Initial weigths with respect to the input [default: Xavier initialisation]. If given, it should be a multidimensional array of `kernel_size` augmented by `nchannels_in` and `nchannels_out` dimensions
   * `bias_init`:     Initial weigths with respect to the bias [default: Xavier initialisation]. If given it should be a `nchannels_out` vector of scalars.
   * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

   # Notes:
   - Xavier initialization is sampled from a `Uniform` distribution between `⨦ sqrt(6/(prod(input_size)*nchannels_in))`
   - to retrieve the output size of the layer, use `size(ConvLayer[2])`. The output size on each dimension _d_ (except the last one that is given by `nchannels_out`) is given by the following formula (ceiled): `output_size[d] = 1 + (input_size[d]+2*padding[d]-kernel_size[d])/stride[d]`
   - with strides higher than 1, the automatic padding is set to keep out_size = in_side/stride
   """
   function ConvLayer(input_size,kernel_size,nchannels_in,nchannels_out;
            stride  = (ones(Int64,length(input_size))...,),
            rng     = Random.GLOBAL_RNG,
            padding = nothing, # zeros(Int64,length(input_size)),
            kernel_init  = rand(rng, Uniform(-sqrt(6/(prod(input_size)*nchannels_in)),sqrt(6/(prod(input_size)*nchannels_in))),(kernel_size...,nchannels_in,nchannels_out)...),
            usebias = true,
            bias_init    = usebias ? rand(rng, Uniform(-sqrt(6/(prod(input_size)*nchannels_in)),sqrt(6/(prod(input_size)*nchannels_in))),nchannels_out) : zeros(Float64,nchannels_out),
            f       = identity,
            df      = match_known_derivatives(f))
      
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
         #target_out_size = [input_size[d]/stride[d] for d in 1:length(input_size)]
         #println(target_out_size)
         padding_total   = [(target_out_size[d]-1)*stride[d] - input_size[d]+kernel_size[d] for d in 1:length(input_size)]
         #println(padding_total)
         padding_start   =  Int.(ceil.(padding_total ./ 2))
         padding_end     =  padding_total .- padding_start  
      else
         padding_start = padding[1]
         padding_end   = padding[2]
      end
      nD == length(stride) || error("`stride` must be either a scalar or a tuple that equates the number of dimensions of input data")
      nD == length(padding_start) == length(padding_end) || error("`padding` must be: (a) the value `nothing` for automatic computation, (b) a scalar for same padding on all dimensions or (c) a 2-elements tuple where each elements are tuples that equate the number of dimensions of input data for indicating the padding to set in front of the data and the padding to set at the ending of the data")

      #println(typeof(weight_init))
      #println(weight_init)

      #new{nD,nD+1}(weight_init,usebias,bias_init,padding,stride,nD,f,df)
      new{nD,nD+1,nD+2}((input_size...,nchannels_in),kernel_init,usebias,bias_init,padding_start,padding_end,stride,nD,f,df,[],[],[])
   end
end

"""
$(TYPEDSIGNATURES)

Alternative constructor for a `ConvLayer` where the number of channels in input is specified as a further dimension in the input size instead of as a separate parameter, so to use `size(previous_layer)[2]` if one wish.

For arguments and default values see the documentation of the main constructor.
"""
function ConvLayer(input_size_with_channel,kernel_size,nchannels_out;
     stride  = (ones(Int64,length(input_size_with_channel)-1)...,),
     rng     = Random.GLOBAL_RNG,
     padding = nothing, # zeros(Int64,length(input_size)),
     kernel_init  = rand(rng, Uniform(-sqrt(6/prod(input_size_with_channel)),sqrt(6/prod(input_size_with_channel))),(kernel_size...,input_size_with_channel[end],nchannels_out)...),
     usebias = true,
     bias_init    = usebias ? rand(rng, Uniform(-sqrt(6/prod(input_size_with_channel)),sqrt(6/prod(input_size_with_channel))),nchannels_out) : zeros(Float64,nchannels_out),
     f       = identity,
     df      = nothing)

     return ConvLayer(input_size_with_channel[1:end-1],kernel_size,input_size_with_channel[end],nchannels_out; stride=stride,rng=rng,padding=padding,kernel_init=kernel_init,usebias=usebias,bias_init=bias_init,f=f,df=df)

end


function preprocess!(layer::ConvLayer{ND,NDPLUS1,NDPLUS2}) where {ND,NDPLUS1,NDPLUS2}

   if length(layer.x_ids) > 0
      return # layer already prepocessed
   end

   input_size, output_size = size(layer)
   nchannels_out = output_size[end]
   nchannels_in  = input_size[end]
   convsize      = input_size[1:end-1]
   ndims_conv    = ND

   wsize = size(layer.weight)
   ysize = output_size
   #println(ysize)
   
   # preallocating temp variables
   w_idx               = Array{Int32,1}(undef,NDPLUS2)
   y_idx               = Array{Int32,1}(undef,NDPLUS1)
   w_idx_conv          = Array{Int32,1}(undef,ND)
   y_idx_conv          = Array{Int32,1}(undef,ND)
   idx_x_source_padded = Array{Int32,1}(undef,ND)
   checkstart          = Array{Bool,1}(undef,ND)
   checkend            = Array{Bool,1}(undef,ND)
   x_idx               = Array{Int32,1}(undef,NDPLUS1)

   @inbounds for nch_in in 1:nchannels_in
      #println("Processing in layer :", nch_in)
      @inbounds for nch_out in 1:nchannels_out
         #println("- processing out layer :", nch_out)
         @inbounds for w_idx_conv in CartesianIndices( ((wsize[1:end-2]...),) ) 
            w_idx_conv = Tuple(w_idx_conv)
            w_idx = (w_idx_conv...,nch_in,nch_out)
            @inbounds for y_idx_conv in CartesianIndices( ((ysize[1:end-1]...),) )
               y_idx_conv = Tuple(y_idx_conv)
               y_idx      = (y_idx_conv...,nch_out)
               #println("y_idx: ",y_idx)
               #println("w_idx: ",w_idx)
               #println("layer.stride: ",layer.stride)
               #quit(1)
               check = true
               @inbounds for d in 1:ndims_conv
                  idx_x_source_padded[d] = w_idx_conv[d] + (y_idx_conv[d] - 1 ) * layer.stride[d]
                  checkstart[d] =  idx_x_source_padded[d] > layer.padding_start[d]
                  checkend[d]   =  idx_x_source_padded[d] <=  layer.padding_start[d] .+ convsize[d]
                  checkstart[d] && checkend[d] || begin check = false; break; end
               end
               check || continue
               
               @inbounds @simd  for d in 1:ndims_conv
                  x_idx[d] = idx_x_source_padded[d] - layer.padding_start[d]
               end
               x_idx[ndims_conv+1] = nch_in

               #println("---")
               #println("x_idx: ", x_idx)
               #println("w_idx: ", w_idx)
               #println("y_idx: ", y_idx)
               push!(layer.x_ids,((x_idx...,)))
               push!(layer.w_ids,w_idx)
               push!(layer.y_ids,y_idx)
               #de_dx_ch_in[idx_x_source...] += dϵ_dz_ch_out[dy_idx...] * w_ch_in_out[w_idx...]
            
            end
         end
      end
   end
end




function _zComp(layer::ConvLayer{ND,NDPLUS1,NDPLUS2},x) where {ND,NDPLUS1,NDPLUS2}
   input_size, output_size = size(layer)
   nchannels_out = output_size[end]

   if ndims(x) == 1
      reshape(x,size(layer)[1]) 
   end

   y = zeros(output_size)
   lx_ids  = layer.x_ids
   ly_ids  = layer.y_ids
   lw_ids  = layer.w_ids
   lweight = layer.weight

   for idx in 1:length(layer.y_ids)
     y[ly_ids[idx]...] += x[lx_ids[idx]...] * lweight[lw_ids[idx]...] 
   end

   for ch_out in 1:nchannels_out
      y_ch_out = selectdim(y,NDPLUS1,ch_out)
      y_ch_out .+= layer.bias[ch_out]
   end

   return y
end


"""
$(TYPEDSIGNATURES)

Compute forward pass of a ConvLayer

"""
function forward(layer::ConvLayer,x)
   z =  _zComp(layer,x) 
   return layer.f.(z)
 end

function backward(layer::ConvLayer{ND,NDPLUS1,NDPLUS2},x, next_gradient) where {ND,NDPLUS1,NDPLUS2}
   z      =  _zComp(layer,x)
  
   if layer.df != nothing
      dfz = layer.df.(z)  
   else
      dfz =  layer.f'.(z) # using AD
   end
   dϵ_dz  = @turbo  dfz .* next_gradient
   de_dx  = zeros(layer.input_size...)

   #lx_ids  = layer.x_ids
   #ly_ids  = layer.y_ids
   #lw_ids  = layer.w_ids
   #lweight = layer.weight

    for idx in 1:length(layer.y_ids)
      @inbounds de_dx[layer.x_ids[idx]...] += dϵ_dz[layer.y_ids[idx]...] * layer.weight[layer.w_ids[idx]...] 
   end
   return de_dx
end


function get_params(layer::ConvLayer)
  if layer.usebias
    return Learnable((layer.weight,layer.bias))
  else 
    return Learnable((layer.weight,))
  end
end


function get_gradient(layer::ConvLayer{ND,NDPLUS1,NDPLUS2},x, next_gradient) where {ND,NDPLUS1,NDPLUS2}
   z      =  _zComp(layer,x)
  
   if layer.df != nothing
      dfz = layer.df.(z)  
   else
      dfz =  layer.f'.(z) # using AD
   end

   dϵ_dz  = @turbo  dfz .* next_gradient
   de_dw  = zeros(size(layer.weight))

   lx_ids  = layer.x_ids
   ly_ids  = layer.y_ids
   lw_ids  = layer.w_ids

   for idx in 1:length(layer.y_ids)
      de_dw[lw_ids[idx]...] += dϵ_dz[ly_ids[idx]...] * x[lx_ids[idx]...] 
   end

   if layer.usebias
      dbias = zeros(length(layer.bias))
      for bias_idx in 1:length(layer.bias)
         nchannel_out = bias_idx
         dϵ_dz_nchannelOut = selectdim(dϵ_dz,layer.ndims+1,nchannel_out)
         dbias[bias_idx] = sum(dϵ_dz_nchannelOut)
      end
      return Learnable((de_dw,dbias))
   else
      return Learnable((de_dw,))
   end
end

function set_params!(layer::ConvLayer,w)
   layer.weight                 = w.data[1]
   layer.usebias && (layer.bias = w.data[2])
end

"""
$(TYPEDSIGNATURES)

Get the dimensions of the layers in terms of (dimensions in input, dimensions in output) including channels as last dimension
"""
function size(layer::ConvLayer)
   nchannels_in  = layer.input_size[end]
   nchannels_out = size(layer.weight)[end]
   in_size  = (layer.input_size...,)
   out_size = ([1 + Int(floor((layer.input_size[d]+layer.padding_start[d]+layer.padding_end[d]-size(layer.weight,d))/layer.stride[d])) for d in 1:layer.ndims]...,nchannels_out)
   #println(size(layer.weight[1],2))
   return (in_size,out_size)
end