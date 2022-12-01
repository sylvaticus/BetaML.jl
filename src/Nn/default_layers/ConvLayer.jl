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

   """
   $(TYPEDSIGNATURES)

   Instantiate a new nD-dimensional, possibly multichannel ConvolutionalLayer

   The input data is either a column vector (in which case is reshaped) or an array of `input_size` augmented by the `n_channels` dimension, the output size depends on the `input_size`, `kernel_size`, `padding` and `striding` but has always `nchannels_out` as its last dimention.  

   # Positional arguments:
   * `input_size`:    Shape of the input layer (integer for 1D convolution, tuple otherwise). Do not consider the channels number here.
   * `kernel_size`:   Size of the kernel (aka filter or learnable weights) (integer for 1D or hypercube kernels or nD-sized tuple for assymmetric kernels). Do not consider the channels number here.
   * `nchannels_in`:  Number of channels in input
   * `nchannels_out`: Number of channels in output
   # Keyword arguments:
   * `kernel_init`:   Initial weigths with respect to the input [default: Xavier initialisation]. If given, it should be a multidimensional array of `kernel_size` augmented by `nchannels_in` and `nchannels_out` dimensions
   * `bias_init`:     Initial weigths with respect to the bias [default: Xavier initialisation]. If given it should be a `nchannels_out` vector of scalars.
   * `f`:   Activation function [def: `relu`]
   * `df`:  Derivative of the activation function [default: `nothing` (i.e. use AD)]
   * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]

   # Notes:
   - Xavier initialization is sampled from a `Uniform` distribution between `⨦ sqrt(6/(prod(input_size)*nchannels_in))`
   - to retrieve the output size of the layer, use `size(ConvLayer[2])`. The output size on each dimension _d_ (except the last one that is given by `nchannels_out`) is given by the following formula (ceiled): `output_size[d] = 1 + (input_size[d]+2*padding[d]-kernel_size[d])/stride[d]`
   """
   function ConvLayer(input_size,kernel_size,nchannels_in,nchannels_out;
            stride  = (ones(Int64,length(input_size))...,),
            rng     = Random.GLOBAL_RNG,
            padding = nothing, # zeros(Int64,length(input_size)),
            kernel_init  = rand(rng, Uniform(-sqrt(6/(prod(input_size)*nchannels_in)),sqrt(6/(prod(input_size)*nchannels_in))),(kernel_size...,nchannels_in,nchannels_out)...),
            usebias = true,
            bias_init    = usebias ? rand(rng, Uniform(-sqrt(6/(prod(input_size)*nchannels_in)),sqrt(6/(prod(input_size)*nchannels_in))),nchannels_out) : zeros(Float64,nchannels_out),
            f       = identity,
            df      = nothing)
      
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
         target_out_size = [Int(round(input_size[d]/stride[d])) for d in 1:length(input_size)]
         padding_total   = [(target_out_size[d]-1)*stride[d] - input_size[d]+kernel_size[d] for d in 1:length(input_size)]
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
      new{nD,nD+1,nD+2}((input_size...,nchannels_in),kernel_init,usebias,bias_init,padding_start,padding_end,stride,nD,f,df)
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

function _xpadComp(layer::ConvLayer,x)
   input_size, output_size = size(layer)
   # input padding
   padding_start = SVector{layer.ndims+1,Int64}([layer.padding_start...,0])
   padding_end   = SVector{layer.ndims+1,Int64}([layer.padding_end...,0])
   padded_size   = input_size .+  padding_start .+ padding_end
   xstart        = padding_start .+ 1
   xends         = padding_start .+ input_size
   xpadded       = zeros(eltype(x), padded_size...)
   xpadded[[range(s,e,step=1) for (s,e) in zip(xstart,xends)]...] = x
   return xpadded
end

function _zComp(layer::ConvLayer,x)
   if ndims(x) == 1
      reshape(x,size(layer)[1]) 
    end
    input_size, output_size = size(layer)
  
    # input padding
    xpadded = _xpadComp(layer,x)

    y = zeros(output_size)

    for yi in CartesianIndices(y)
     yiarr         = convert(SVector{layer.ndims+1,Int64},yi)
     nchannel_out  = yiarr[end]
     starti        = yiarr[1:end-1] .* layer.stride .- layer.stride .+ 1
     starti        = vcat(starti,1)
     endi          = starti .+  convert(SVector{layer.ndims+1,Int64},size(layer.weight)[1:end-1]) .- 1
     weight        = selectdim(layer.weight,layer.ndims+2,nchannel_out)
     y[yi]         = layer.bias[nchannel_out] .+ dot(weight, xpadded[[range(s,e,step=1) for (s,e) in zip(starti,endi)]...])
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


function backward(layer::ConvLayer,x,next_gradient) # with respect to inputs: derror/dx
   # To look at for faster conv versions:
   #     https://arxiv.org/pdf/1312.5851.pdf 
   #     https://discourse.julialang.org/t/computing-linear-convolution-efficiently/66496

   z      =  _zComp(layer,x)
  
   if layer.df != nothing
      dfz = layer.df.(z)  
   else
      dfz =  layer.f'.(z) # using AD
   end
   dϵ_dz  = @turbo  dfz .* next_gradient
   #xpadded = _xpadComp(layer,x)

   #println("dϵ_dz: ", dϵ_dz)
   nchannels_out = size(layer.weight)[end]
   nchannels_in  = size(layer.weight)[end-1]
   input_size, output_size = size(layer)
   de_dx         = zeros(layer.input_size...)

   #=
   for idx in CartesianIndices(de_dx)
      idx = convert(SVector{layer.ndims+1,Int64},idx)
      nch_in = idx[end]
      w_ch_in = selectdim(layer.weight,layer.ndims+1,nch_in)
      convdims = idx[1:end-1]
      for nch_out in 1:nchannels_out
         w_ch_in_out  = selectdim(w_ch_in,layer.ndims+1,nch_out)
         dϵ_dz_ch_out = selectdim(dϵ_dz,layer.ndims+1,nch_out)
         de_dx_ch_in  = selectdim(de_dx,layer.ndims+1,nch_in)

      end
   end
   =#

   for nch_in in 1:nchannels_in
      w_ch_in = selectdim(layer.weight,layer.ndims+1,nch_in)
      for nch_out in 1:nchannels_out
         w_ch_in_out  = selectdim(w_ch_in,layer.ndims+1,nch_out)
         dϵ_dz_ch_out = selectdim(dϵ_dz,layer.ndims+1,nch_out)
         de_dx_ch_in  = selectdim(de_dx,layer.ndims+1,nch_in)
         for w_idx in CartesianIndices(w_ch_in_out)
            w_idx = convert(SVector{layer.ndims,Int64},w_idx)
            for dey_idx in CartesianIndices(dϵ_dz_ch_out)
               dey_idx = convert(SVector{layer.ndims,Int64},dey_idx)
               idx_x_source_padded = w_idx .+ (dey_idx .- 1 ) .* layer.stride
               if all(idx_x_source_padded .> layer.padding_start) && all(idx_x_source_padded .<=  layer.padding_start .+ input_size[1:end-1])
                  idx_x_source = idx_x_source_padded .- layer.padding_start
                  #println("idx_x_source: ", idx_x_source)
                  #println("w_idx: ", w_idx)
                  #println("dey_idx: ", dey_idx)
                  de_dx_ch_in[idx_x_source...] += dϵ_dz_ch_out[dey_idx...] * w_ch_in_out[w_idx...]
               end
            end
         end
      end
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

function get_gradient(layer::ConvLayer,x,next_gradient) # derror/dw
   #=
   dϵ_dw  = @turbo dϵ_dz * x' # @avx
   dϵ_dwb = dϵ_dz
   return Learnable((dϵ_dw,dϵ_dwb))
   =#

   #input_size, output_size = size(layer)

   z      =  _zComp(layer,x)
   if layer.df != nothing
      dfz = layer.df.(z)  
   else
      dfz =  layer.f'.(z) # using AD
   end
   dϵ_dz  = @turbo  dfz .* next_gradient

   #println("dϵ_dz: ", dϵ_dz)
   dw    = zeros(size(layer.weight))

   xpadded = _xpadComp(layer,x)
   #println("xpadded: ", xpadded)
   # dw computation
   for idx in CartesianIndices(dw)
      idx = convert(SVector{layer.ndims+2,Int64},idx)
      nchannel_out = idx[end]
      nchannel_in  = idx[end-1]
      wdims        = idx[1:end-2]
      #println("***** widx: ", idx)
      # need to compute the Ys reached by this specific weigth and the corresponding x
      # foreach Y(oudims,co) 
      #    add to weigth(dims,ci,co) the X(dims_trans,ci) that accompy the weigth(dims,ci,co) for this Y(oudims,co)
      dϵ_dz_nchannelOut = selectdim(dϵ_dz,layer.ndims+1,nchannel_out)
      xval = zeros(size(dϵ_dz_nchannelOut))
      for yi in CartesianIndices(dϵ_dz_nchannelOut)
         idx_y = convert(SVector{layer.ndims,Int64},yi)
         idx_x_dest = idx_y
         #println("- idx_y: ", idx_y)
         #println("- idx_x_dest: ", idx_x_dest)
         idx_x_source = wdims .+ (idx_y .- 1 ) .* layer.stride # xpadded[i] = w[i] + (Y[i] -1 ) * STRIDE
         #println("- idx_x_source: ", idx_x_source)
         #println("vcat(idx_x_source,nchannel_in): ",vcat(idx_x_source,nchannel_in))
         xval[idx_x_dest] = xpadded[vcat(idx_x_source,nchannel_in)...]  
         #println(xval[idx_x_dest...])
         #println(dϵ_dz_nchannelOut[yi])
         #dw[idx...] += xpadded[vcat(idx_x_source,nchannel_in)...]  * dϵ_dz_nchannelOut[idx_y...]
      end
      #println("y: ", dϵ_dz_nchannelOut)
      #println("x: ", xval)
      dw[idx...] = dot(xval,dϵ_dz_nchannelOut)  # slighly more efficient than using += on each individual product
   end

   if layer.usebias
      dbias = zeros(length(layer.bias))
      for bias_idx in 1:length(layer.bias)
         nchannel_out = bias_idx
         dϵ_dz_nchannelOut = selectdim(dϵ_dz,layer.ndims+1,nchannel_out)
         dbias[bias_idx] = sum(dϵ_dz_nchannelOut)
      end
      return Learnable((dw,dbias))
   else
      return Learnable((dwb,))
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