"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Experimental. Works well, but still too slow for any practical application.

"""
$(TYPEDEF)

Representation of a pooling layer in the network (weightless)

**EXPERIMENTAL**: Still too slow for practical applications

In the middle between `VectorFunctionLayer` and `ScalarFunctionLayer`, it applies a function to the set of nodes defined in a sliding kernel.

# Fields:
$(TYPEDFIELDS)
"""
struct PoolingLayer{ND,NDPLUS1,NDPLUS2,TF <: Function, TDF <: Union{Nothing,Function}, WET <: Number} <: AbstractLayer
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

   @doc """
   $(TYPEDSIGNATURES)

   Instantiate a new nD-dimensional, possibly multichannel PoolingLayer

   The input data is either a column vector (in which case it is reshaped) or an array of `input_size` augmented by the `n_channels` dimension, the output size depends on the `input_size`, `kernel_size`, `padding` and `striding` but always has `nchannels_out` as its last dimension.  

   # Positional arguments:
   * `input_size`:    Shape of the input layer (integer for 1D convolution, tuple otherwise). Do not consider the channels number here.
   * `kernel_size`:   Size of the kernel (aka filter) (integer for 1D or hypercube kernels or nD-sized tuple for asymmetric kernels). Do not consider the channels number here.
   * `nchannels_in`:  Number of channels in input
   
   # Keyword arguments:
   * `kernel_eltype`: Kernel eltype [def: `Float64`]
   * `stride`: "Steps" to move the convolution with across the various tensor dimensions [def: `kernel_size`, i.e. each X contributes to a single y]
   * `padding`: Integer or 2-element tuple of tuples of the starting and ending padding across the various dimensions [def: `nothing`, i.e. set the padding required to keep out_size = in_size / stride ]
   * `f`:   Activation function. It should have a vector as input and produce a scalar as output [def: `maximum`]
   * `df`:  Derivative (gradient) of the activation function for the various inputs. [default: `nothing` (i.e. use AD)]
   * `kernel_eltype`: Eltype of the output when the input is not made of floating point numbers [def: `Float64`]


   # Notes:
   - to retrieve the output size of the layer, use `size(PoolLayer[2])`. The output size on each dimension _d_ (except the last one that is given by `nchannels_out`) is given by the following formula (ceiled): `output_size[d] = 1 + (input_size[d]+2*padding[d]-kernel_size[d])/stride[d]`
   - on GPUs only `f=maximum` and `f=mean` are supported (their derivatives are computed directly within the GPU kernel), while other functions work only on the CPU. For other pooling functions on GPUs, use a [`VectorFunctionLayer`](@ref) with `batched=true` (see its documentation)
   - a 1-dimensional pooling of `nₗ` nodes over windows of `k` nodes moving one node at a time, as done by [`pool1d`](@ref), is obtained with `PoolingLayer(nₗ,k,1,stride=1,padding=0,f=f)`. The output has a further singleton "channel" dimension (`size(layer) == ((nₗ,1),(nₗ-k+1,1))`), that the following dense layers simply ignore
   - differently from a ConvLayer, the pooling always applies on a single channel level, so that the output always has the same number of channels as the input. If you want to reduce the channels number either use a `ConvLayer` with the desired number of channels in output or use a `ReshaperLayer` to add a 1-element further dimension that will be treated as "channel" and choose the desired stride for the last pooling dimension (the one that was originally the channel dimension) 
   """
   function PoolingLayer(input_size,kernel_size,nchannels_in;
            stride  = kernel_size,
            kernel_eltype = Float64,
            padding = nothing, # (zeros(Int64,length(input_size)),zeros(Int64,length(input_size))),
            f       = maximum,
            df      = match_known_derivatives(f))
      
      nchannels_out = nchannels_in
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
      
      new{nD,nD+1,nD+2,typeof(f),typeof(df),kernel_eltype}(input_size_with_nchin,output_size_with_nchout,kernel_size_with_nchin_nchout,padding_start,padding_end,stride,nD,f,df)
   end
end

"""
$(TYPEDSIGNATURES)

Alternative constructor for a `PoolingLayer` where the number of channels in input is specified as a further dimension in the input size instead of as a separate parameter, so as to use `size(previous_layer)[2]` if one wishes.

For arguments and default values see the documentation of the main constructor.
"""
function PoolingLayer(input_size_with_channel,kernel_size;
     stride  = kernel_size,
     padding = nothing, # (zeros(Int64,length(input_size)),zeros(Int64,length(input_size))),
     f       = maximum,
     kernel_eltype = Float64,
     df      = match_known_derivatives(f))
     return PoolingLayer(input_size_with_channel[1:end-1],kernel_size,input_size_with_channel[end]; kernel_eltype = kernel_eltype,stride=stride,padding=padding,f=f,df=df)

end


# The pooling is computed element by element, with generic N-dimensional functions that run on
# the CPU or on the GPU where the data is (see `_compute_elements!`). The batch X has the records
# along the last dimension (input_size... × n records).

"Ranges (on each dimension) of the input elements in the pooling window of the output element `o`"
@inline _pool_window(o,ksz,stride,pad,isz) = ntuple(d -> max(1,(o[d]-1)*stride[d]+1-pad[d]):min(isz[d],(o[d]-1)*stride[d]+ksz[d]-pad[d]), length(o))

"Apply the pooling function `f` to the window `w` (with explicit loops for `maximum` and `mean`, that work on GPUs)"
_pool_f(f,w) = f(w)
@inline function _pool_f(::typeof(maximum),w)
   m = @inbounds w[first(CartesianIndices(w))]
   @inbounds for j in CartesianIndices(w)
      m = ifelse(w[j] > m, w[j], m)
   end
   return m
end
@inline function _pool_f(::typeof(mean),w)
   s = zero(eltype(w))
   @inbounds for j in CartesianIndices(w)
      s += w[j]
   end
   return s / length(w)
end

@inline function _pool_forward_el(I, x, f, ksz, stride, pad, ::Val{ND}) where {ND}
   o    = ntuple(d -> I[d], Val(ND))
   c    = I[ND+1]
   b    = I[ND+2]
   isz  = ntuple(d -> size(x,d), Val(ND))
   rngs = _pool_window(o,ksz,stride,pad,isz)
   return _pool_f(f,view(x,rngs...,c,b))
end

# dx[i...,c,b] = ∑_o g[o...,c,b] * ∂f(window(o))/∂x[i...,c,b] for the o whose window includes i
@inline function _pool_backward_el(I, x, g, f, df, ksz, stride, pad, ::Val{ND}) where {ND}
   i   = ntuple(d -> I[d], Val(ND))
   c   = I[ND+1]
   b   = I[ND+2]
   isz = ntuple(d -> size(x,d), Val(ND))
   osz = ntuple(d -> size(g,d), Val(ND))
   olo = ntuple(d -> max(1,cld(i[d]+pad[d]-ksz[d],stride[d])+1), Val(ND))
   ohi = ntuple(d -> min(osz[d],fld(i[d]+pad[d]-1,stride[d])+1), Val(ND))
   acc = zero(eltype(g))*zero(float(eltype(x)))
   for o in CartesianIndices(ntuple(d -> olo[d]:ohi[d], Val(ND)))
      rngs = _pool_window(Tuple(o),ksz,stride,pad,isz)
      j    = CartesianIndex(ntuple(d -> i[d]-first(rngs[d])+1, Val(ND)))
      @inbounds acc += g[o,c,b] * _pool_dfx(f,df,view(x,rngs...,c,b),j)
   end
   return acc
end

"Derivative of f(w) with respect to the element at (Cartesian) position `j` of the window `w`"
_pool_dfx(f,df,w,j) = df(w)[j]
_pool_dfx(f,::Nothing,w,j) = gradient(f,collect(w))[1][j] # using AD
@inline function _pool_dfx(::typeof(maximum),df,w,j)
   # only the (first) maximum element has a non-zero derivative, as in `dmaximum`
   jmax = first(CartesianIndices(w))
   @inbounds for jj in CartesianIndices(w)
      if w[jj] > w[jmax]
         jmax = jj
      end
   end
   return jmax == j ? one(eltype(w)) : zero(eltype(w))
end
@inline _pool_dfx(::typeof(mean),df,w,j) = one(eltype(w)) / length(w)

"Eltype of the output of the pooling layer: the eltype of the input if it is a float, `WET` otherwise"
_pool_eltype(::PoolingLayer{ND,NDPLUS1,NDPLUS2,TF,TDF,WET},X) where {ND,NDPLUS1,NDPLUS2,TF,TDF,WET} = eltype(X) <: AbstractFloat ? eltype(X) : WET

"""
$(TYPEDSIGNATURES)

Compute forward pass of a PoolingLayer for a batch of records (on the last dimension)

"""
function forward_batch(layer::PoolingLayer{ND},X) where {ND}
   X = reshape(X,layer.input_size...,_nbatch(X))
   y = similar(X,_pool_eltype(layer,X),(layer.output_size...,_nbatch(X)))
   ksz = ntuple(d -> layer.kernel_size[d], Val(ND))
   return _compute_elements!(y,_pool_forward_el,(X,layer.f,ksz,Tuple(layer.stride),Tuple(layer.padding_start),Val(ND)))
end

function backward_batch(layer::PoolingLayer{ND},X,next_gradient) where {ND}
   X  = reshape(X,layer.input_size...,_nbatch(X))
   g  = reshape(next_gradient,layer.output_size...,_nbatch(next_gradient))
   dx = similar(g,promote_type(eltype(g),_pool_eltype(layer,X)),size(X))
   ksz = ntuple(d -> layer.kernel_size[d], Val(ND))
   return _compute_elements!(dx,_pool_backward_el,(X,g,layer.f,layer.df,ksz,Tuple(layer.stride),Tuple(layer.padding_start),Val(ND)))
end

function get_params(layer::PoolingLayer)
    return Learnable(())
end


function get_gradient_batch(layer::PoolingLayer,X,next_gradient)
   return Learnable(()) 
end

function set_params!(layer::PoolingLayer,w)

end

"""
$(TYPEDSIGNATURES)

Get the dimensions of the layer in terms of (dimensions in input, dimensions in output) including channels as last dimension
"""
function size(layer::PoolingLayer{ND,NDPLUS1,NDPLUS2}) where {ND,NDPLUS1,NDPLUS2}
   return ((layer.input_size...,),(layer.output_size...,))
end