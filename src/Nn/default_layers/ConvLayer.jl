"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Experimental

"""
$(TYPEDEF)

A generic N+1 (channels) dimensional convolutional layer

**EXPERIMENTAL**: Still too slow for practical applications

This convolutional layer has two constructors, one with the form `ConvLayer(input_size,kernel_size,nchannels_in,nchannels_out)`, and an alternative one as `ConvLayer(input_size_with_channel,kernel_size,nchannels_out)`.
If the input is a vector, use a [`ReshaperLayer`](@ref) in front.


# Fields:
$(TYPEDFIELDS)
"""
struct ConvLayer{ND,NDPLUS1,NDPLUS2,TF <: Function, TDF <: Union{Nothing,Function}, WET <: Number, WT <: AbstractArray{WET,NDPLUS2}, BT <: AbstractVector{WET}} <: AbstractLayer
   "Input size (including nchannel_in as last dimension)"
   input_size::SVector{NDPLUS1,Int64}
   "Output size (including nchannel_out as last dimension)"
   output_size::SVector{NDPLUS1,Int64}
   "Weight tensor (aka \"filter\" or \"kernel\") with respect to the input from previous layer or data (kernel_size array augmented by the nchannels_in and nchannels_out dimensions)"
   weight::WT
   "Whether to use (and learn) a bias weight [def: true]"
   usebias::Bool
   "Bias (nchannels_out array)"
   bias::BT
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
   df::TDF

   @doc """
   $(TYPEDSIGNATURES)

   Instantiate a new nD-dimensional, possibly multichannel ConvolutionalLayer

   The input data is either a column vector (in which case it is reshaped) or an array of `input_size` augmented by the `n_channels` dimension, the output size depends on the `input_size`, `kernel_size`, `padding` and `striding` but always has `nchannels_out` as its last dimension.  

   # Positional arguments:
   * `input_size`:    Shape of the input layer (integer for 1D convolution, tuple otherwise). Do not consider the channels number here.
   * `kernel_size`:   Size of the kernel (aka filter or learnable weights) (integer for 1D or hypercube kernels or nD-sized tuple for asymmetric kernels). Do not consider the channels number here.
   * `nchannels_in`:  Number of channels in input
   * `nchannels_out`: Number of channels in output
   # Keyword arguments:
   * `stride`: "Steps" to move the convolution with across the various tensor dimensions [def: `ones`]
   * `padding`: Integer or 2-element tuple of tuples of the starting and ending padding across the various dimensions [def: `nothing`, i.e. set the padding required to keep the same dimensions in output (with stride==1)]
   * `f`:   Activation function [def: `identity`]
   * `df`:  Derivative of the activation function [default: try to match a known function, AD otherwise. Use `nothing` to force AD]
   * `kernel_eltype`: Kernel eltype [def: `Float64`]
   * `kernel_init`:   Initial weights with respect to the input [default: Xavier initialisation]. If explicitly provided, it should be a multidimensional array of `kernel_size` augmented by `nchannels_in` and `nchannels_out` dimensions
   * `bias_init`:     Initial weights with respect to the bias [default: Xavier initialisation]. If given it should be a `nchannels_out` vector of scalars.
   * `rng`: Random Number Generator (see [`FIXEDSEED`](@ref)) [default: `Random.GLOBAL_RNG`]

   # Notes:
   - Xavier initialization is sampled from a `Uniform` distribution between `⨦ sqrt(6/(prod(input_size)*nchannels_in))`
   - to retrieve the output size of the layer, use `size(ConvLayer[2])`. The output size on each dimension _d_ (except the last one that is given by `nchannels_out`) is given by the following formula (ceiled): `output_size[d] = 1 + (input_size[d]+2*padding[d]-kernel_size[d])/stride[d]`
   - with strides higher than 1, the automatic padding is set to keep out_size = in_size/stride
   """
   function ConvLayer(input_size,kernel_size,nchannels_in,nchannels_out;
            stride  = (ones(Int64,length(input_size))...,),
            rng     = Random.GLOBAL_RNG,
            padding = nothing, # zeros(Int64,length(input_size)),
            kernel_eltype = Float64,
            kernel_init  = xavier_init(
               prod(input_size)*nchannels_in,
               prod(input_size)*nchannels_in,
               (kernel_size...,nchannels_in,nchannels_out),
               rng=rng,
               eltype=kernel_eltype),
            usebias = true,
            bias_init    = usebias ? xavier_init(
               prod(input_size)*nchannels_in,
               prod(input_size)*nchannels_in,
               nchannels_out,
               rng=rng,
               eltype=kernel_eltype) : zeros(kernel_eltype,nchannels_out),
            f       = identity,
            df      = match_known_derivatives(f))
      
      # make sure all are tuples of the right dimension...
      if typeof(input_size) <: Integer
         input_size = (input_size,)
      end
      nD = length(input_size)
      if typeof(kernel_size) <: Integer
         kernel_size = ([kernel_size for d in 1:nD]...,)
      end
      length(input_size) == length(kernel_size) || error("Number of dimensions of the kernel must equal the number of dimensions of the input data")
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
      
      # computation of the output size

      #nchannels_out = kernel_size[end]
      #in_size  = (input_size...,nchannels_in)
      out_size = ([1 + Int(floor((input_size[d]+padding_start[d]+padding_end[d]-size(kernel_init,d))/stride[d])) for d in 1:nD]...,nchannels_out)
      #println(size(layer.weight[1],2))

      weight = _dense(kernel_init,kernel_eltype)
      bias   = _dense(bias_init,kernel_eltype)
      new{nD,nD+1,nD+2,typeof(f),typeof(df),kernel_eltype,typeof(weight),typeof(bias)}((input_size...,nchannels_in),out_size,weight,usebias,bias,padding_start,padding_end,stride,nD,f,df)
   end
   # Constructor from the individual fields (used e.g. to move the layer to a GPU)
   function ConvLayer(input_size::SVector,output_size::SVector,weight::AbstractArray,usebias::Bool,bias::AbstractVector,padding_start::SVector,padding_end::SVector,stride::SVector,ndims::Integer,f,df)
      nD = length(stride)
      new{nD,nD+1,nD+2,typeof(f),typeof(df),eltype(weight),typeof(weight),typeof(bias)}(input_size,output_size,weight,usebias,bias,padding_start,padding_end,stride,ndims,f,df)
   end
end

"""
$(TYPEDSIGNATURES)

Alternative constructor for a `ConvLayer` where the number of channels in input is specified as a further dimension in the input size instead of as a separate parameter, so as to use `size(previous_layer)[2]` if one wishes.

For arguments and default values see the documentation of the main constructor.
"""
function ConvLayer(input_size_with_channel,kernel_size,nchannels_out;
     stride  = (ones(Int64,length(input_size_with_channel)-1)...,),
     rng     = Random.GLOBAL_RNG,
     padding = nothing, # zeros(Int64,length(input_size)),
     kernel_eltype = Float64,
     kernel_init  = xavier_init(
       prod(input_size_with_channel),
       prod(input_size_with_channel),
       (kernel_size...,input_size_with_channel[end],nchannels_out),
       rng=rng,
       eltype=kernel_eltype),
     usebias = true,
     bias_init    = usebias ? xavier_init(
      prod(input_size_with_channel),
      prod(input_size_with_channel),
      nchannels_out,
      rng=rng,
      eltype=kernel_eltype) : zeros(kernel_eltype,nchannels_out),
     f       = identity,
     df      = match_known_derivatives(f))
     return ConvLayer(input_size_with_channel[1:end-1],kernel_size,input_size_with_channel[end],nchannels_out; stride=stride,rng=rng,padding=padding, kernel_eltype = kernel_eltype,kernel_init=kernel_init,usebias=usebias,bias_init=bias_init,f=f,df=df)
end

# The convolution is computed by generic N-dimensional kernels (KernelAbstractions.jl), running
# on the CPU or on the GPU where the layer weights are. Each kernel instance computes a single
# element of its output, "gathering" all the contributions to it (so no atomic operations are
# needed). The batch X has the records along the last dimension (input_size... × n records).
# The valid ranges of the loop indices are computed in advance on each dimension, so that the
# inner loops contain no bound checks.

# z[o...,co,b] = bias[co] + ∑_{ci,k} w[k...,ci,co] * x[i...,ci,b], with i = (o-1)*stride+k-pad
@kernel function _conv_forward_kernel!(z, @Const(x), @Const(w), @Const(bias), stride, pad, ::Val{ND}) where {ND}
   I   = @index(Global, Cartesian)
   o   = ntuple(d -> I[d], Val(ND))
   co  = I[ND+1]
   b   = I[ND+2]
   ks  = ntuple(d -> max(1,1+pad[d]-(o[d]-1)*stride[d]):min(size(w,d),size(x,d)+pad[d]-(o[d]-1)*stride[d]), Val(ND))
   acc = zero(eltype(z)) + bias[co]
   @inbounds for ci in 1:size(w,ND+1), k in CartesianIndices(ks)
      i    = ntuple(d -> (o[d]-1)*stride[d] + k[d] - pad[d], Val(ND))
      acc += w[k,ci,co] * x[i...,ci,b]
   end
   @inbounds z[I] = acc
end

# dx[i...,ci,b] = ∑_{co,o} w[k...,ci,co] * dz[o...,co,b], with k = i+pad-(o-1)*stride
@kernel function _conv_dx_kernel!(dx, @Const(dz), @Const(w), stride, pad, ::Val{ND}) where {ND}
   I   = @index(Global, Cartesian)
   i   = ntuple(d -> I[d], Val(ND))
   ci  = I[ND+1]
   b   = I[ND+2]
   os  = ntuple(d -> max(1,cld(i[d]+pad[d]-size(w,d),stride[d])+1):min(size(dz,d),fld(i[d]+pad[d]-1,stride[d])+1), Val(ND))
   acc = zero(eltype(dx))
   @inbounds for co in 1:size(w,ND+2), o in CartesianIndices(os)
      k    = ntuple(d -> i[d] + pad[d] - (o[d]-1)*stride[d], Val(ND))
      acc += w[k...,ci,co] * dz[o,co,b]
   end
   @inbounds dx[I] = acc
end

# dw[k...,ci,co] = ∑_{b,o} dz[o...,co,b] * x[i...,ci,b], with i = (o-1)*stride+k-pad
@kernel function _conv_dw_kernel!(dw, @Const(dz), @Const(x), stride, pad, ::Val{ND}) where {ND}
   I   = @index(Global, Cartesian)
   k   = ntuple(d -> I[d], Val(ND))
   ci  = I[ND+1]
   co  = I[ND+2]
   os  = ntuple(d -> max(1,cld(1+pad[d]-k[d],stride[d])+1):min(size(dz,d),fld(size(x,d)+pad[d]-k[d],stride[d])+1), Val(ND))
   acc = zero(eltype(dw))
   @inbounds for b in 1:size(x,ND+2), o in CartesianIndices(os)
      i    = ntuple(d -> (o[d]-1)*stride[d] + k[d] - pad[d], Val(ND))
      acc += dz[o,co,b] * x[i...,ci,b]
   end
   @inbounds dw[I] = acc
end

_convinput(layer::ConvLayer,X)  = reshape(X,layer.input_size...,_nbatch(X))
_convoutput(layer::ConvLayer,X) = reshape(X,layer.output_size...,_nbatch(X))

function _zComp(layer::ConvLayer{ND},X) where {ND}
   z = similar(layer.weight,(layer.output_size...,_nbatch(X)))
   _conv_forward_kernel!(_backend(z))(z,X,layer.weight,layer.bias,Tuple(layer.stride),Tuple(layer.padding_start),Val(ND); ndrange=size(z))
   return z
end

function _dϵ_dz(layer::ConvLayer,X,next_gradient)
   df = _derivative(layer.f,layer.df)
   return df.(_zComp(layer,X)) .* next_gradient
end

function _dϵ_dx(layer::ConvLayer{ND},dϵ_dz) where {ND}
   dx = similar(layer.weight,(layer.input_size...,_nbatch(dϵ_dz)))
   _conv_dx_kernel!(_backend(dx))(dx,dϵ_dz,layer.weight,Tuple(layer.stride),Tuple(layer.padding_start),Val(ND); ndrange=size(dx))
   return dx
end

function _dϵ_dw(layer::ConvLayer{ND},dϵ_dz,X) where {ND}
   dw = similar(layer.weight)
   _conv_dw_kernel!(_backend(dw))(dw,dϵ_dz,X,Tuple(layer.stride),Tuple(layer.padding_start),Val(ND); ndrange=size(dw))
   if layer.usebias
      dbias = vec(sum(dϵ_dz,dims=(ntuple(identity,ND)...,ND+2)))
      return Learnable((dw,dbias))
   else
      return Learnable((dw,))
   end
end

"""
$(TYPEDSIGNATURES)

Compute forward pass of a ConvLayer for a batch of records (on the last dimension)

"""
function forward_batch(layer::ConvLayer,X)
   return layer.f.(_zComp(layer,_convinput(layer,X)))
end

function backward_batch(layer::ConvLayer,X,next_gradient)
   dϵ_dz = _dϵ_dz(layer,_convinput(layer,X),_convoutput(layer,next_gradient))
   return _dϵ_dx(layer,dϵ_dz)
end

function get_gradient_batch(layer::ConvLayer,X,next_gradient)
   X     = _convinput(layer,X)
   dϵ_dz = _dϵ_dz(layer,X,_convoutput(layer,next_gradient))
   return _dϵ_dw(layer,dϵ_dz,X)
end

function backward_and_gradient_batch(layer::ConvLayer,X,next_gradient)
   X     = _convinput(layer,X)
   dϵ_dz = _dϵ_dz(layer,X,_convoutput(layer,next_gradient))
   return (_dϵ_dx(layer,dϵ_dz),_dϵ_dw(layer,dϵ_dz,X))
end

function get_params(layer::ConvLayer)
  if layer.usebias
    return Learnable((layer.weight,layer.bias))
  else 
    return Learnable((layer.weight,))
  end
end


function set_params!(layer::ConvLayer,w)
   layer.weight                 .= w.data[1]
   layer.usebias && (layer.bias .= w.data[2])
end

"""
$(TYPEDSIGNATURES)

Get the dimensions of the layer in terms of (dimensions in input, dimensions in output) including channels as last dimension
"""
function size(layer::ConvLayer)
   #nchannels_in  = layer.input_size[end]
   #nchannels_out = size(layer.weight)[end]
   #in_size  = (layer.input_size...,)
   #out_size = ([1 + Int(floor((layer.input_size[d]+layer.padding_start[d]+layer.padding_end[d]-size(layer.weight,d))/layer.stride[d])) for d in 1:layer.ndims]...,nchannels_out)
   #println(size(layer.weight[1],2))
   #return (in_size,out_size)
   return ((layer.input_size...,),(layer.output_size...,))
end