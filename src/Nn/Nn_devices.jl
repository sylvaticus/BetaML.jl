"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# ------------------------------------------------------------------------------
# Device (CPU/GPU) handling and other helpers shared by the layers
#
# Internally, the device where a computation takes place is represented by a "prototype": a
# zero-length array that lives on that device and has the element type to compute with.
# New arrays are allocated with `similar(proto,...)` and data is moved with `copyto!`, so the
# same code works for any GPU vendor (and for the JLArrays used in the tests). Which device an
# array lives on is given by the KernelAbstractions backend of the array, while MLDataDevices
# is used only to find the GPU backend loaded by the user when `device=:gpu`.

"Return the KernelAbstractions backend of an array (`CPU()` for CPU arrays unknown to KernelAbstractions, e.g. ranges)"
_backend(x::Array) = KA.CPU()
function _backend(x::AbstractArray)
    try
        return KA.get_backend(x)
    catch e
        e isa ArgumentError || rethrow()
        return KA.CPU()
    end
end
_iscpu(x::AbstractArray) = _backend(x) isa KA.CPU
_samedevice(x::AbstractArray,y::AbstractArray) = typeof(_backend(x)) == typeof(_backend(y))

"Return a copy of `x` on the CPU (or `x` itself if it is already there)"
_tocpu(x::AbstractArray) = _iscpu(x) ? x : Array(x)

"""
$(TYPEDSIGNATURES)

Move (and convert) the array `x` to the device and eltype of the prototype array `proto`.

If `x` is already there with the right eltype it is returned as it is (no copy).
"""
function _to_device(proto::AbstractArray{T},x::AbstractArray) where {T}
    if _samedevice(proto,x)
        return eltype(x) == T ? x : convert(AbstractArray{T},x)
    elseif _iscpu(proto)
        return convert(Array{T},Array(x))
    else
        return copyto!(similar(proto,T,size(x)),convert(Array{T},_tocpu(x)))
    end
end

"Ensure that user-provided initial weights are a mutable dense array (e.g. not a range) of the requested eltype"
_dense(x::DenseArray{T},::Type{T}) where {T} = x
_dense(x::AbstractArray,::Type{T}) where {T} = _iscpu(x) ? convert(Array{T},x) : convert(AbstractArray{T},x)

# Moving layers to a device

"Adaptor (see Adapt.jl) that moves the floating point arrays of a layer to the device (and eltype) of `proto`"
struct _DeviceMover{P<:AbstractArray}
    proto::P
end
Adapt.adapt_storage(to::_DeviceMover,x::AbstractArray{<:AbstractFloat}) = _to_device(to.proto,x)
Adapt.adapt_storage(to::_DeviceMover,x::StaticArray) = x

_ismoveleaf(x) = Functors.isleaf(x)
_ismoveleaf(x::AbstractArray{<:Number}) = true
_ismoveleaf(x::Function) = true
_ismoveleaf(x::AbstractLayer) = true

# Generic way to move a layer: rebuild it (with its constructor taking all fields in order) with
# its arrays moved. Layers that can't be rebuilt this way should define their own
# `Adapt.adapt_structure(to,layer::MyLayer)`.
function Adapt.adapt_structure(to::_DeviceMover,layer::AbstractLayer)
    return Functors.fmap(x -> Adapt.adapt(to,x), layer; exclude = x -> x !== layer && _ismoveleaf(x))
end

"""
$(TYPEDSIGNATURES)

Return a copy of `layer` whose learnable parameters are on the device (and have the eltype) of `mover.proto`.
"""
function _layer_to_device(mover::_DeviceMover,layer::AbstractLayer)
    isempty(get_params(layer).data) && return layer
    lname  = nameof(typeof(layer))
    advice = "Define `Adapt.adapt_structure(to,layer::$(lname))` returning a copy of the layer with its arrays `x` replaced by `Adapt.adapt(to,x)`, or give the layer a constructor taking all its fields in order and make sure that the fields can store arrays of any type (e.g. using type parameters)."
    newlayer = try
        Adapt.adapt(mover,layer)
    catch e
        error("Unable to move a layer of type `$(lname)` to the requested device. $advice Original error: $(sprint(showerror,e))")
    end
    for p in get_params(newlayer).data
        if p isa AbstractArray{<:AbstractFloat} && !(_samedevice(p,mover.proto) && eltype(p) == eltype(mover.proto))
            error("The learnable parameters of the layer of type `$(lname)` have not been moved to the requested device. $advice")
        end
    end
    return newlayer
end

# Resolution of the `device` option

_device_spec(spec::AbstractString) = _device_spec(Symbol(lowercase(spec)))
function _device_spec(spec::Symbol)
    spec in (:auto,:cpu,:gpu) || error("Unknown device `$(spec)`. Use `:auto`, `:cpu`, `:gpu` or a device object of the MLDataDevices.jl package (e.g. `CUDADevice()`).")
    return spec
end
_device_spec(spec::MLDataDevices.AbstractDevice) = spec

"""
$(TYPEDSIGNATURES)

Return the MLDataDevices device of the first functional GPU backend loaded by the user (CUDA, AMDGPU, Metal or oneAPI).
"""
function _gpu_device()
    for D in (MLDataDevices.CUDADevice, MLDataDevices.AMDGPUDevice, MLDataDevices.MetalDevice, MLDataDevices.oneAPIDevice)
        if MLDataDevices.loaded(D) && MLDataDevices.functional(D)
            return MLDataDevices.gpu_device()
        end
    end
    _cuda_without_cudnn() && return MLDataDevices.CUDADevice()
    error("`device=:gpu` has been requested, but no functional GPU has been found. Load the package of your GPU vendor (e.g. `using CUDA`, `using AMDGPU`, `using Metal` or `using oneAPI`) and check that it works.")
end

"True if CUDA.jl is loaded and functional (MLDataDevices selects CUDA only if cuDNN.jl is loaded too, but we don't need cuDNN)"
function _cuda_without_cudnn()
    cudaext = Base.get_extension(MLDataDevices,:CUDAExt)
    return !isnothing(cudaext) && cudaext.CUDA.functional()
end

"True if the GPU backend of the device type `D` (e.g. `CUDADevice`) is loaded and functional"
function _device_available(D::Type{<:MLDataDevices.AbstractDevice})
    (MLDataDevices.loaded(D) && MLDataDevices.functional(D)) && return true
    return D <: MLDataDevices.CUDADevice && _cuda_without_cudnn()
end

"Return an array on the device `dev` (an MLDataDevices device)"
_device_ref(dev::MLDataDevices.AbstractDevice) = dev(zeros(Float32,1))

# Saving a model trained on a GPU: its parameters are saved on the CPU, together with the kind of
# GPU (the MLDataDevices device type) they were on, so that the model can be moved back there when
# it is used again. The device type is stored by name, as JLD2 can't store all types.

"Name (module and type) of the MLDataDevices device type of the GPU array `x`, or `nothing` if unknown"
function _device_name(x::AbstractArray)
    D = MLDataDevices.get_device_type(x)
    (D isa Type && D <: MLDataDevices.AbstractGPUDevice) || return nothing
    return (mod=string(parentmodule(D)),type=string(nameof(D)))
end

"Device type from its name as returned by `_device_name`, or `nothing` if its module isn't loaded"
function _device_type(name)
    for m in (MLDataDevices,Main,Base.loaded_modules_array()...)
        if string(m) == name.mod && isdefined(m,Symbol(name.type))
            D = getfield(m,Symbol(name.type))
            return D isa Type && D <: MLDataDevices.AbstractDevice ? D : nothing
        end
    end
    return nothing
end

# Element-wise computation of arrays. On GPUs it is done by a KernelAbstractions kernel, on the CPU
# by plain (multi-threaded) Julia loops. This way on the CPU any function can be used (since
# KernelAbstractions 0.10 also CPU kernels are compiled like GPU ones, with the same restrictions).

@kernel function _elementwise_kernel!(out, elfun::F, args) where {F}
    I = @index(Global, Cartesian)
    @inbounds out[I] = elfun(I,args...)
end

"""
$(TYPEDSIGNATURES)

Set each element `I` of the array `out` to `elfun(I,args...)` and return `out`.

`elfun` must work on the device of `out` (on GPUs: no memory allocations, no dynamic dispatch).
"""
function _compute_elements!(out,elfun::F,args::Tuple) where {F}
    if _iscpu(out)
        if Threads.nthreads() > 1 && length(out) >= 1024 # multi-threading pays off only on large arrays
            Threads.@threads for I in CartesianIndices(out)
                @inbounds out[I] = elfun(I,args...)
            end
        else
            @inbounds for I in CartesianIndices(out)
                out[I] = elfun(I,args...)
            end
        end
    else
        _elementwise_kernel!(_backend(out))(out,elfun,args; ndrange=size(out))
    end
    return out
end

# Helpers for the batched computations of the layers

"Number of records of a batch (records are on the last dimension)"
_nbatch(X) = size(X,ndims(X))

"View a batch of 1-D records as a matrix (one record per column)"
_asmatrix(X) = ndims(X) == 2 ? X : reshape(X,:,_nbatch(X))

"Return the derivative of the scalar function `f`: `df` if known, an automatic derivative (that works also on GPUs) otherwise"
_derivative(f,df) = df
_derivative(f,::Nothing) = x -> ForwardDiff.derivative(f,x)

"""
$(TYPEDSIGNATURES)

Apply a single-record function `frecord(x)` (or `frecord(x,g)` when `G` is given) to each record of the batch `X` and stack the results along a new last dimension.

The computation happens on the CPU, while the result is returned on the device of `X`. This is used for the operations that can't be vectorised over the batch (e.g. generic vector activation functions).
"""
function _recordwise(frecord,X,G=nothing)
    Xc  = _tocpu(X)
    xs  = eachslice(Xc,dims=ndims(Xc))
    out = if isnothing(G)
        stack(frecord(x) for x in xs)
    else
        Gc = _tocpu(G)
        stack(frecord(x,g) for (x,g) in zip(xs,eachslice(Gc,dims=ndims(Gc))))
    end
    return _to_device(similar(X,eltype(out),0),out)
end

"Same as `_recordwise` but summing the (Learnable) results instead of stacking them, with the result moved to the device of `ref`"
function _recordwise_sum(frecord,X,G,ref)
    Xc  = _tocpu(X)
    Gc  = _tocpu(G)
    out = sum(frecord(x,g) for (x,g) in zip(eachslice(Xc,dims=ndims(Xc)),eachslice(Gc,dims=ndims(Gc))))
    return Learnable(map(x -> _to_device(similar(ref,eltype(x),0),x), out.data))
end
