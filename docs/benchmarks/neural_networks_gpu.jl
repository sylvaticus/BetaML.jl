# BetaML and Flux neural networks trained on the CPU and on a NVIDIA GPU (with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)), with the same network structure, data (in `Float32`), optimisation algorithm (ADAM) and batch size. The times refer to the training only, with the data already on the device and excluding the compilation, and are measured on a single run. The scores are computed on a test set.
# - Dense network: a regression of 25000 records with 50 features, with two inner layers of 256 neurons, trained for 5 epochs. The score is the relative mean error.
# - Convolutional network: the classification of the 70000 images of the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits (60000 used for training), with the network of the [Image recognition tutorial](https://github.com/sylvaticus/BetaML.jl/blob/main/docs/src/tutorials/Image%20recognition/Image_recognition.jl), trained for 2 epochs. The score is the accuracy.
using CUDA, cuDNN # cuDNN is needed by Flux for the convolutional layers on GPU
import MLDatasets
ENV["DATADEPS_ALWAYS_ACCEPT"] = true # to download MNIST without being asked
results = DataFrame(task=String[],model=String[],device=String[],time_s=Float64[],score=Float64[])

## Dense network
n, D, epochs, bs = 25_000, 50, 5, 256
x = rand(copy(TESTRNG),Float32,n,D)
y = [sum(r[1:10].^2) - sum(r[11:20]) + sum(sin.(3 .* r[21:50])) for r in eachrow(x)]
((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.8,0.2],rng=copy(TESTRNG))

function betaml_dense(x,y;epochs)
    layers = [DenseLayer(D,256,f=relu,w_eltype=Float32,rng=copy(TESTRNG)),
              DenseLayer(256,256,f=relu,w_eltype=Float32,rng=copy(TESTRNG)),
              DenseLayer(256,1,w_eltype=Float32,rng=copy(TESTRNG))]
    m = NeuralNetworkEstimator(layers=layers,epochs=epochs,batch_size=bs,opt_alg=ADAM(),cache=false,
                               onfail="continue",verbosity=NONE,rng=copy(TESTRNG))
    fit!(m,x,y) # with the default `device=:auto` the network is trained where the data is
    return m
end
function flux_dense(x,y;epochs)
    Random.seed!(123)
    dev       = CUDA.functional() && x isa CuArray ? Flux.gpu_device() : Flux.cpu_device()
    model     = Flux.Chain(Flux.Dense(D=>256,Flux.relu),Flux.Dense(256=>256,Flux.relu),Flux.Dense(256=>1)) |> dev
    loader    = Flux.DataLoader((permutedims(x),permutedims(y)),batchsize=bs,shuffle=true)
    opt_state = Flux.setup(Flux.Adam(),model)
    for e in 1:epochs
        Flux.train!((m,xb,yb) -> Flux.mse(m(xb),yb),model,loader,opt_state)
    end
    return model
end

## Convolutional network (MNIST)
function load_mnist(split)
    x, y = MLDatasets.MNIST(split=split)[:]
    x    = permutedims(x,(2,1,3))                       # 28×28 images (rows × columns) × records
    return reshape(Float32.(x),28*28,size(x,3))', y    # BetaML: records × features
end
xmtrain, ymtrain = load_mnist(:train)
xmtest,  ymtest  = load_mnist(:test)
ymtrain_oh       = Float32.(fit!(OneHotEncoder(categories=collect(0:9)),ymtrain))
mepochs, mbs     = 2, 128

function betaml_conv(x,y;epochs)
    l1 = ReshaperLayer((28*28,1),(28,28,1))
    l2 = ConvLayer(size(l1)[2],(5,5),8,stride=2,f=relu,kernel_eltype=Float32,rng=copy(TESTRNG))
    l3 = ConvLayer(size(l2)[2],(3,3),16,stride=2,f=relu,kernel_eltype=Float32,rng=copy(TESTRNG))
    l4 = ConvLayer(size(l3)[2],(3,3),32,stride=2,f=relu,kernel_eltype=Float32,rng=copy(TESTRNG))
    l5 = ConvLayer(size(l4)[2],(3,3),32,stride=2,f=relu,kernel_eltype=Float32,rng=copy(TESTRNG))
    l6 = PoolingLayer(size(l5)[2],(2,2),stride=(2,2),f=mean)
    l7 = ReshaperLayer(size(l6)[2])
    l8 = DenseLayer(size(l7)[2][1],10,w_eltype=Float32,rng=copy(TESTRNG))
    l9 = VectorFunctionLayer(10,f=softmax)
    m  = NeuralNetworkEstimator(layers=[l1,l2,l3,l4,l5,l6,l7,l8,l9],loss=crossentropy,epochs=epochs,batch_size=mbs,
                                opt_alg=ADAM(),cache=false,onfail="continue",verbosity=NONE,rng=copy(TESTRNG))
    fit!(m,x,y)
    return m
end
function flux_conv(x,y;epochs)
    Random.seed!(123)
    dev   = CUDA.functional() && x isa CuArray ? Flux.gpu_device() : Flux.cpu_device()
    model = Flux.Chain(Flux.Conv((5,5),1=>8,pad=2,stride=2,Flux.relu),Flux.Conv((3,3),8=>16,pad=1,stride=2,Flux.relu),
                       Flux.Conv((3,3),16=>32,pad=1,stride=2,Flux.relu),Flux.Conv((3,3),32=>32,pad=1,stride=2,Flux.relu),
                       Flux.GlobalMeanPool(),Flux.flatten,Flux.Dense(32=>10),Flux.softmax) |> dev
    loader    = Flux.DataLoader((reshape(permutedims(x),28,28,1,:),permutedims(y)),batchsize=mbs,shuffle=true)
    opt_state = Flux.setup(Flux.Adam(),model)
    for e in 1:epochs
        Flux.train!((m,xb,yb) -> Flux.crossentropy(m(xb),yb),model,loader,opt_state)
    end
    return model
end
flux_conv_predict(model,x) = model(reshape(permutedims(x),28,28,1,:))

## Running the benchmarks on each device
devices = CUDA.functional() ? ["CPU","GPU"] : ["CPU"]
sync()  = CUDA.functional() && CUDA.synchronize() # GPU operations are asynchronous
todevice(device,a) = device == "GPU" ? CuArray(collect(a)) : collect(a)
for device in devices
    dx(a) = todevice(device,a)
    xtr, ytr, xte   = dx(xtrain), dx(ytrain), dx(xtest)
    xmtr, ymtr, xmte = dx(xmtrain), dx(ymtrain_oh), dx(xmtest)
    ## warm up (compilation)
    betaml_dense(xtr[1:1000,:],ytr[1:1000],epochs=1); flux_dense(xtr[1:1000,:],ytr[1:1000],epochs=1)
    betaml_conv(xmtr[1:1000,:],ymtr[1:1000,:],epochs=1); flux_conv(xmtr[1:1000,:],ymtr[1:1000,:],epochs=1)
    ## dense network
    t = @elapsed (m = betaml_dense(xtr,ytr,epochs=epochs); sync())
    push!(results,["Dense","BetaML",device,t,relative_mean_error(ytest,Array(predict(m,xte)))])
    t = @elapsed (m = flux_dense(xtr,ytr,epochs=epochs); sync())
    push!(results,["Dense","Flux.jl",device,t,relative_mean_error(ytest,vec(Array(m(permutedims(xte)))))])
    ## convolutional network
    t = @elapsed (m = betaml_conv(xmtr,ymtr,epochs=mepochs); sync())
    push!(results,["Convolutional (MNIST)","BetaML",device,t,accuracy(ymtest .+ 1,mode(Array(predict(m,xmte))))])
    t = @elapsed (m = flux_conv(xmtr,ymtr,epochs=mepochs); sync())
    push!(results,["Convolutional (MNIST)","Flux.jl",device,t,accuracy(ymtest .+ 1,Flux.onecold(Array(flux_conv_predict(m,xmte))))])
end

CUDA.functional() || @warn "No functional CUDA GPU found: only the CPU benchmarks have been run."
results
