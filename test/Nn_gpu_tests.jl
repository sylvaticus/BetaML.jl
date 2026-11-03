using Test
using Statistics, LinearAlgebra, Random, DelimitedFiles
using BetaML
const MLDataDevices = BetaML.Nn.MLDataDevices
import BetaML.Nn: ConvLayer, ReshaperLayer, buildNetwork, train!, loss, get_gradient, backward_and_gradient_batch

# The GPU support is tested with the arrays of the device chosen with the `gpu_test_device` option
# of runtests.jl (or with the environment variable BETAML_GPU_TEST_DEVICE):
# - "JLArrays" (default): JLArrays is a CPU-based reference implementation of GPU arrays. Like real
#   GPU arrays it forbids scalar indexing, so it allows testing the GPU code without a GPU (e.g. on
#   GitHub Actions), although it can't detect the errors that happen only when compiling for GPUs.
# - "CUDA" or "AMDGPU": a real GPU. Its package isn't a test dependency, so run these tests with
#   `julia test/run_gpu_tests.jl CUDA`, which creates an environment with it.
gpu_test_device = @isdefined(GPU_TEST_DEVICE) ? GPU_TEST_DEVICE : get(ENV,"BETAML_GPU_TEST_DEVICE","JLArrays")
gpu_test_device in ("JLArrays","CUDA","AMDGPU") || error("Unsupported `gpu_test_device` \"$gpu_test_device\". Use \"JLArrays\", \"CUDA\" or \"AMDGPU\".")
@eval using $(Symbol(gpu_test_device))

if gpu_test_device == "JLArrays"
    JLArrays.allowscalar(false)
    const DevArray = JLArray
    # MLDataDevices doesn't know JLArrays: this device lets us test the `device=<device object>`
    # option (the same code path of `device=:gpu`) with them...
    struct JLDevice <: MLDataDevices.AbstractGPUDevice end
    (::JLDevice)(x::AbstractArray) = JLArray(x)
    testdevice() = JLDevice()
    # ...and these are the hooks that GPU backends provide to MLDataDevices (used to restore saved models)
    const JL_AVAILABLE = Ref(true) # to simulate a machine where the GPU isn't available
    MLDataDevices.get_device_type(::JLArray) = JLDevice
    MLDataDevices.loaded(::Union{JLDevice,Type{<:JLDevice}}) = true
    MLDataDevices.functional(::Union{JLDevice,Type{<:JLDevice}}) = JL_AVAILABLE[]
elseif gpu_test_device == "CUDA"
    CUDA.functional() || error("CUDA is not functional on this machine")
    CUDA.allowscalar(false)
    const DevArray = CuArray
    testdevice() = MLDataDevices.CUDADevice()
elseif gpu_test_device == "AMDGPU"
    AMDGPU.functional() || error("AMDGPU is not functional on this machine")
    AMDGPU.allowscalar(false)
    const DevArray = ROCArray
    testdevice() = MLDataDevices.AMDGPUDevice()
end

"Move an array to the GPU (keeping its eltype)"
todev(x) = DevArray(x)

TESTRNG = FIXEDRNG

println("*** Testing Neural Network batched API and GPU support (on $gpu_test_device arrays)...")

"Move the parameters of a layer to the GPU, as `fit!` does internally"
todevlayer(l,T=Float64) = BetaML.Nn._layer_to_device(BetaML.Nn._DeviceMover(similar(todev(zeros(T,1)),T,0)),l)

"Check that the batched API of `l` is consistent with its single-record API and that it gives the same results on the GPU"
function check_layer(l,X,G)
    xs = eachslice(X,dims=ndims(X))
    gs = eachslice(G,dims=ndims(G))
    Y  = forward_batch(l,X)
    dX = backward_batch(l,X,G)
    dW = get_gradient_batch(l,X,G)
    # batched vs single-record
    @test Y  ≈ stack(forward(l,x) for x in xs)
    @test dX ≈ stack(backward(l,x,g) for (x,g) in zip(xs,gs))
    dWsingle = sum(get_gradient(l,x,g) for (x,g) in zip(xs,gs))
    @test all(dW.data .≈ dWsingle.data)
    dX2, dW2 = backward_and_gradient_batch(l,X,G)
    @test dX2 ≈ dX && all(dW2.data .≈ dW.data)
    # GPU vs CPU
    lg  = todevlayer(l)
    Xg  = todev(X); Gg = todev(G)
    Yg  = forward_batch(lg,Xg)
    dXg = backward_batch(lg,Xg,Gg)
    dWg = get_gradient_batch(lg,Xg,Gg)
    @test Yg isa DevArray && Array(Yg) ≈ Y
    @test dXg isa DevArray && Array(dXg) ≈ dX
    @test all(p isa DevArray for p in dWg.data)
    @test all(Array.(dWg.data) .≈ dW.data)
end

println("Testing Learnable...")
L32 = Learnable((Float32[1 2; 3 4], Float32[1,2]))
@test eltype((L32 * 0.5).data[1]) == Float32
@test eltype((0.1 - L32 / 3.0 + 1e-8).data[2]) == Float32
@test eltype(sqrt(L32 * L32).data[1]) == Float32
Lg = Learnable((todev(Float32[1 2; 3 4]),))
@test ((Lg + Lg) * 2.0).data[1] isa DevArray{Float32}

println("Testing the batched API of the default layers (on CPU and GPU)...")
rng = copy(TESTRNG)
B   = 5
check_layer(DenseLayer(3,4,f=tanh,rng=rng),randn(rng,3,B),randn(rng,4,B))
check_layer(DenseLayer(3,4,f=x->sin(x)+x^2,df=nothing,rng=rng),randn(rng,3,B),randn(rng,4,B)) # AD derivative
check_layer(DenseNoBiasLayer(3,4,f=relu,rng=rng),randn(rng,3,B),randn(rng,4,B))
check_layer(ReplicatorLayer(3),randn(rng,3,B),randn(rng,3,B))
check_layer(ScalarFunctionLayer(3,f=tanh),randn(rng,3,B),randn(rng,3,B))
check_layer(VectorFunctionLayer(3,f=softmax),randn(rng,3,B),randn(rng,3,B))
check_layer(VectorFunctionLayer(4,f=x->pool1d(x,2,f=mean)),randn(rng,4,B),randn(rng,3,B)) # computed record by record
check_layer(GroupedLayer([DenseLayer(2,3,rng=rng),ReplicatorLayer(1),DenseNoBiasLayer(2,1,rng=rng)]),randn(rng,5,B),randn(rng,5,B))
check_layer(ReshaperLayer((6,),(3,2)),randn(rng,6,B),randn(rng,3,2,B))
check_layer(ConvLayer((5,4),(3,2),2,3,stride=(2,1),f=relu,rng=rng),randn(rng,5,4,2,B),randn(rng,3,4,3,B))
check_layer(ConvLayer((5,4),(3,2),2,3,usebias=false,rng=rng),randn(rng,5,4,2,B),randn(rng,5,4,3,B))
check_layer(ConvLayer(7,3,1,2,stride=2,rng=rng),randn(rng,7,1,B),randn(rng,4,2,B)) # 1D
check_layer(PoolingLayer((6,4),(2,2),3),randn(rng,6,4,3,B),randn(rng,3,2,3,B))
check_layer(PoolingLayer((5,4),(3,2),3,stride=(1,2),f=mean),randn(rng,5,4,3,B),randn(rng,5,2,3,B)) # overlapping windows

println("Testing ScalarFunctionLayer and VectorFunctionLayer with learnable weights or batched functions...")
# ScalarFunctionLayer: yᵢ = f(xᵢ,wᵢ), with one weight per node or a single one shared by all nodes
fs(x,w) = w*tanh(x)
X = randn(rng,3,B); G = randn(rng,3,B)
for wsize in ([3],[1])
    w    = rand(rng,wsize[1]) .+ 0.5
    lad  = ScalarFunctionLayer(3,wsize=wsize,f=fs,w=copy(w)) # derivatives by AD
    lman = ScalarFunctionLayer(3,wsize=wsize,f=fs,w=copy(w),dfx=(x,w)->w*(1-tanh(x)^2),dfw=(x,w)->tanh(x))
    @test forward_batch(lad,X) ≈ w .* tanh.(X)
    @test backward_batch(lad,X,G) ≈ w .* (1 .- tanh.(X).^2) .* G
    dw = get_gradient_batch(lad,X,G).data[1]
    @test dw ≈ (wsize == [3] ? vec(sum(tanh.(X) .* G,dims=2)) : [sum(tanh.(X) .* G)])
    @test backward_batch(lman,X,G) ≈ backward_batch(lad,X,G)
    @test get_gradient_batch(lman,X,G).data[1] ≈ dw
    check_layer(lad,X,G)
    check_layer(lman,X,G)
end
@test_throws ErrorException ScalarFunctionLayer(3,wsize=[2],f=fs)

# VectorFunctionLayer: derivatives by AD (pullback), record by record or on the whole batch
fv(x) = x ./ sum(x,dims=1) # works on a single record as well as on a batch (column by column)
X = rand(rng,3,B) .+ 0.1; G = randn(rng,3,B)
lrec = VectorFunctionLayer(3,f=fv)
lbat = VectorFunctionLayer(3,f=fv,batched=true)
@test size(lbat) == ((3,),(3,))
@test forward_batch(lbat,X) ≈ forward_batch(lrec,X) ≈ X ./ sum(X,dims=1)
jacobian_based = stack(BetaML.autojacobian(fv,x)' * g for (x,g) in zip(eachcol(X),eachcol(G)))
@test backward_batch(lrec,X,G) ≈ jacobian_based
@test backward_batch(lbat,X,G) ≈ jacobian_based
check_layer(lrec,X,G)
check_layer(lbat,X,G)
# ...with learnable weights
fvw(x,w) = (x .* w) ./ sum(x .* w,dims=1)
w     = rand(rng,3) .+ 0.5
lrecw = VectorFunctionLayer(3,f=fvw,wsize=[3],w=copy(w))
lbatw = VectorFunctionLayer(3,f=fvw,wsize=[3],w=copy(w),batched=true)
@test forward_batch(lbatw,X) ≈ forward_batch(lrecw,X)
@test backward_batch(lbatw,X,G) ≈ backward_batch(lrecw,X,G)
dwjac = sum(BetaML.autojacobian(wt -> fvw(x,wt),w)' * g for (x,g) in zip(eachcol(X),eachcol(G)))
@test get_gradient_batch(lrecw,X,G).data[1] ≈ dwjac
@test get_gradient_batch(lbatw,X,G).data[1] ≈ dwjac
check_layer(lrecw,X,G)
check_layer(lbatw,X,G)

# Convolution vs a direct (brute force) computation
l  = ConvLayer((5,4),(3,2),2,3,stride=(2,1),padding=((1,0),(1,1)),rng=copy(TESTRNG))
X  = randn(copy(TESTRNG),5,4,2,B)
Y  = forward_batch(l,X)
(osize1,osize2) = size(l)[2][1:2]
Ybrute = zeros(osize1,osize2,3,B)
for b in 1:B, co in 1:3, o1 in 1:osize1, o2 in 1:osize2
    s = l.bias[co]
    for ci in 1:2, k1 in 1:3, k2 in 1:2
        i1 = (o1-1)*2+k1-1; i2 = (o2-1)*1+k2-0 # padding_start is (1,0)
        (1 <= i1 <= 5 && 1 <= i2 <= 4) && (s += l.weight[k1,k2,ci,co] * X[i1,i2,ci,b])
    end
    Ybrute[o1,o2,co,b] = s
end
@test Y ≈ Ybrute

# A custom layer implementing only the single-record API still works (on the CPU)
struct MyScaleLayer{T<:AbstractVector} <: AbstractLayer
    w::T
end
BetaML.Nn.forward(l::MyScaleLayer,x)            = l.w .* x
BetaML.Nn.backward(l::MyScaleLayer,x,g)         = l.w .* g
BetaML.Nn.get_params(l::MyScaleLayer)           = Learnable((l.w,))
BetaML.Nn.get_gradient(l::MyScaleLayer,x,g)     = Learnable((x .* g,))
BetaML.Nn.set_params!(l::MyScaleLayer,w)        = (l.w .= w.data[1])
Base.size(l::MyScaleLayer)                      = ((length(l.w),),(length(l.w),))
lc = MyScaleLayer([1.0,2.0,3.0])
X  = randn(copy(TESTRNG),3,B); G = randn(copy(TESTRNG),3,B)
@test forward_batch(lc,X) ≈ lc.w .* X
@test get_gradient_batch(lc,X,G).data[1] ≈ vec(sum(X .* G,dims=2))
# ...and as it is parametric and broadcasts, it can be moved to and run on GPUs
@test forward_batch(todevlayer(lc),todev(X)) isa DevArray
# A custom layer whose fields can store only CPU arrays gives an informative error when moved to a GPU
struct MyCPUScaleLayer <: AbstractLayer
    w::Vector{Float64}
end
BetaML.Nn.get_params(l::MyCPUScaleLayer) = Learnable((l.w,))
@test_throws ErrorException todevlayer(MyCPUScaleLayer([1.0,2.0]))

println("Testing NeuralNetworkEstimator on GPU...")

xtrain = rand(copy(TESTRNG),60,3)
ytrain = [2*r[1]-r[2]+r[3]^2 for r in eachrow(xtrain)]
xtest  = rand(copy(TESTRNG),10,3)
layers = [DenseLayer(3,6,f=relu,rng=copy(TESTRNG)),DenseLayer(6,1,rng=copy(TESTRNG))]
mopts  = (epochs=30,batch_size=8,verbosity=NONE,onfail="continue")

# device=:auto (default): training where the data is
mcpu   = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),mopts...)
mgpu    = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),mopts...)
ŷcpu   = fit!(mcpu,xtrain,ytrain)
ŷgpu    = fit!(mgpu,todev(xtrain),todev(ytrain))
@test ŷgpu isa DevArray{Float64}                              # results returned where the data is
@test mgpu.par.nnstruct.layers[1].w isa DevArray{Float64}      # Float64 data on the GPU => Float64 computations
@test Array(ŷgpu) ≈ ŷcpu
@test predict(mgpu,xtest) isa Vector{Float64}                 # CPU input => CPU output (computed on the GPU)
@test predict(mgpu,xtest) ≈ predict(mcpu,xtest)

# device object (same code path as device=:gpu) with CPU data: batches moved to the GPU, Float32 computations
mdev   = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),device=testdevice(),mopts...)
ŷdev   = fit!(mdev,xtrain,ytrain)
@test ŷdev isa Vector{Float32}
@test mdev.par.nnstruct.layers[1].w isa DevArray{Float32}
@test isapprox(ŷdev,ŷcpu,rtol=1e-3)
@test predict(mdev,xtest) isa Vector{Float32}
@test predict(mdev,todev(xtest)) isa DevArray{Float32}
@test mdev.hpar.layers[1].w isa Matrix{Float64}              # the user-provided layers are not modified

# device=:cpu with GPU data: computations on the CPU, results returned on the GPU
mcpu2  = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),device=:cpu,mopts...)
ŷcpu2  = fit!(mcpu2,todev(xtrain),todev(ytrain))
@test mcpu2.par.nnstruct.layers[1].w isa Matrix{Float64}
@test ŷcpu2 isa DevArray && Array(ŷcpu2) ≈ ŷcpu

# Classification with the softmax layer and cross-entropy loss
iris  = readdlm(joinpath(@__DIR__,"data","iris_shuffled.csv"),',',skipstart=1)
xiris = fit!(Scaler(),convert(Array{Float64,2}, iris[:,1:4]))
yiris = fit!(OneHotEncoder(),iris[:,5])
clayers = [DenseLayer(4,8,f=relu,rng=copy(TESTRNG)),DenseLayer(8,3,rng=copy(TESTRNG)),VectorFunctionLayer(3,f=softmax)]
copts = (loss=crossentropy,dloss=dcrossentropy,epochs=10,batch_size=16,verbosity=NONE,onfail="continue")
mccpu = NeuralNetworkEstimator(;layers=deepcopy(clayers),rng=copy(TESTRNG),copts...)
mcgpu  = NeuralNetworkEstimator(;layers=deepcopy(clayers),rng=copy(TESTRNG),copts...)
ŷccpu = fit!(mccpu,xiris,yiris)
ŷcgpu  = fit!(mcgpu,todev(xiris),todev(Float64.(yiris)))
@test Array(ŷcgpu) ≈ ŷccpu

# Convolutional network
xconv  = rand(copy(TESTRNG),40,6*6*2)
yconv  = [sum(r[1:36])-sum(r[37:72]) for r in eachrow(xconv)]
cvl    = [ReshaperLayer((72,),(6,6,2)),ConvLayer((6,6),(3,3),2,3,f=relu,rng=copy(TESTRNG)),PoolingLayer((6,6,3),(2,2)),ReshaperLayer((3,3,3)),DenseLayer(27,1,rng=copy(TESTRNG))]
mcvcpu = NeuralNetworkEstimator(;layers=deepcopy(cvl),rng=copy(TESTRNG),mopts...)
mcvgpu  = NeuralNetworkEstimator(;layers=deepcopy(cvl),rng=copy(TESTRNG),mopts...)
@test Array(fit!(mcvgpu,todev(xconv),todev(yconv))) ≈ fit!(mcvcpu,xconv,yconv)

# Parameters are saved on the CPU
model_save("test.jld2";mdevsaved=mdev)
mdevloaded = model_load("test.jld2","mdevsaved")
@test mdevloaded.par.nnstruct.layers[1].w isa Matrix{Float32}
@test mdevloaded.hpar.opt_alg.m[1].data[1] isa Matrix{Float32}
@test predict(mdevloaded,xtest) ≈ predict(mdev,xtest)
@test mdev.par.nnstruct.layers[1].w isa DevArray{Float32}     # the original model is still on the GPU

# A model trained on GPU with `device=:auto` is moved back to the GPU when used for predictions after loading it
mauto = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),mopts...)
fit!(mauto,todev(xtrain),todev(ytrain))
model_save("test.jld2";mauto)
@test !haskey(mauto.info,"device_at_save")                   # the original model is not modified
ml = model_load("test.jld2","mauto")
@test ml.par.nnstruct.layers[1].w isa Matrix{Float64}
ŷl = predict(ml,xtest)
@test ml.par.nnstruct.layers[1].w isa DevArray{Float64}       # moved back on the first use
@test ŷl isa Vector{Float64} && ŷl ≈ predict(mauto,xtest)
@test !haskey(ml.info,"device_at_save")
# ... unless the GPU isn't available (that we can simulate only with JLArrays)
if gpu_test_device == "JLArrays"
    JL_AVAILABLE[] = false
    ml = model_load("test.jld2","mauto")
    ml.opt.verbosity = STD
    @test_logs (:info,r"not available") predict(ml,xtest)
    @test ml.par.nnstruct.layers[1].w isa Matrix{Float64}
    @test predict(ml,xtest) ≈ predict(mauto,xtest)
    JL_AVAILABLE[] = true
end
# fit! follows instead the data (here on the CPU)
ml = model_load("test.jld2","mauto")
fit!(ml,xtrain,ytrain)
@test ml.par.nnstruct.layers[1].w isa Matrix{Float64}
@test !haskey(ml.info,"device_at_save")
@test predict(ml,xtest) isa Vector{Float64}
@test ml.par.nnstruct.layers[1].w isa Matrix{Float64}

# The parameters of each epoch are stored as (CPU) copies
mhigh = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),epochs=3,verbosity=HIGH,device=testdevice())
fit!(mhigh,xtrain,ytrain)
@test mhigh.info["par_per_epoch"][1][1].data[1] isa Matrix{Float32}
@test mhigh.info["par_per_epoch"][1][1].data[1] != mhigh.info["par_per_epoch"][end][1].data[1]

# Wrong or unavailable devices
@test_throws ErrorException fit!(NeuralNetworkEstimator(device=:foo,verbosity=NONE),xtrain,ytrain)
if gpu_test_device == "JLArrays" # no GPU package loaded
    @test_throws ErrorException fit!(NeuralNetworkEstimator(device=:gpu,verbosity=NONE),xtrain,ytrain)
else
    mgpu2 = NeuralNetworkEstimator(;layers=deepcopy(layers),rng=copy(TESTRNG),device=:gpu,mopts...)
    ŷgpu2 = fit!(mgpu2,xtrain,ytrain)
    @test ŷgpu2 isa Vector{Float32}
    @test mgpu2.par.nnstruct.layers[1].w isa DevArray{Float32}
    @test isapprox(ŷgpu2,ŷcpu,rtol=1e-3)
end

println("Testing AutoEncoder on GPU...")
xae    = rand(copy(TESTRNG),50,5)
aeopts = (encoded_size=2,epochs=20,verbosity=NONE,onfail="continue")
aecpu  = AutoEncoder(;rng=copy(TESTRNG),aeopts...)
aegpu  = AutoEncoder(;rng=copy(TESTRNG),aeopts...)
x2cpu  = fit!(aecpu,xae)
x2gpu  = fit!(aegpu,todev(xae))                                # device=:auto: trained where the data is
@test x2gpu isa DevArray{Float64} && Array(x2gpu) ≈ x2cpu
@test info(aegpu)["rme"] ≈ info(aecpu)["rme"]
@test predict(aegpu,xae) isa Matrix{Float64} && predict(aegpu,xae) ≈ x2cpu
@test inverse_predict(aegpu,x2gpu) isa DevArray && Array(inverse_predict(aegpu,x2gpu)) ≈ inverse_predict(aecpu,x2cpu)
aedev  = AutoEncoder(;rng=copy(TESTRNG),device=testdevice(),aeopts...)
x2dev  = fit!(aedev,xae)                                        # CPU data, trained on the GPU
@test x2dev isa Matrix{Float32} && isapprox(x2dev,x2cpu,rtol=1e-3)
@test aedev.par.fullnn.par.nnstruct.layers[1].w isa DevArray{Float32}
model_save("test.jld2";aegpu)                                   # saved on the CPU...
ael    = model_load("test.jld2","aegpu")
@test ael.par.fullnn.par.nnstruct.layers[1].w isa Matrix{Float64}
@test ael.cres isa Matrix{Float64}
@test predict(ael,xae) ≈ x2cpu
@test ael.par.fullnn.par.nnstruct.layers[1].w isa DevArray{Float64} # ...and moved back to the GPU when used

println("Testing MLJ interface with the device option...")
import MLJBase
X, y  = MLJBase.@load_boston
model = BetaML.Bmlj.NeuralNetworkRegressor(rng=copy(TESTRNG),device=testdevice(),epochs=10)
(fitresult, cache, report) = MLJBase.fit(model, 0, X, y)
yhat  = MLJBase.predict(model, fitresult, X)
@test yhat isa Vector{Float32}
@test fitresult.par.nnstruct.layers[1].w isa DevArray{Float32}
