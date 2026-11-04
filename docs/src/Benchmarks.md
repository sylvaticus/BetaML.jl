# [BetaML Benchmarks](@id benchmarks)

This page reports some basic benchmarks of BetaML models, together with those of other leading Julia packages for the same tasks. It aims to check for regressions across BetaML versions and to give an idea of the relative performances of the packages, not to be a rigorous comparison.

!!! note
    This page is not computed when the documentation is built: it is written by the script `docs/benchmarks/run_benchmarks.jl` (see the instructions within it), run on the machine described below. As the times depend on the machine, compare them only within the same version of this page.

## Setup

- Date: 2026-11-04
- Julia: v1.13.1, 22 thread(s)
- CPU: Intel(R) Core(TM) Ultra 7 165H
- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU
- Packages: BetaML v0.13.1, Flux v0.16.15, DecisionTree v0.12.4, Clustering v0.15.8, GaussianMixtures v0.3.14, CUDA v6.5.1

All benchmarks share this setup. Unless otherwise noted, times, memory and allocations refer to the training of the model on the whole dataset and are medians over several runs (measured with [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)), while the scores are means and standard deviations over a 10-folds cross-validation.
We tried to match the hyperparameters of the other packages to the BetaML default ones, but differences remain, and they may explain in part the different performances and scores.

```julia
using Random, Statistics, DelimitedFiles, Logging
using DataFrames, DataStructures, BenchmarkTools, StableRNGs
using BetaML
import DecisionTree, Flux, Clustering, GaussianMixtures

TESTRNG = StableRNG(123)
DATADIR = joinpath(@__DIR__,"..","..","test","data")

"Median time (s), memory (MiB) and allocations of `f()`"
function bmark(f)
    b = @benchmark $f() samples=10 seconds=60 evals=1
    return (median(b.times)/1e9, b.memory/2^20, b.allocs)
end

"Mean and standard deviation of the (train,test) scores returned by `f(traindata,testdata,rng)` over a 10-folds cross-validation of `data`"
function cv_scores(f,data)
    sampler = KFold(nsplits=10,rng=copy(TESTRNG))
    scores  = cross_validation(f,data,sampler,return_statistics=false)
    train   = [s[1] for s in scores]
    test    = [s[2] for s in scores]
    return (mean(train),std(train),mean(test),std(test))
end
```

## Regression

A simple regression over 500 points with 4 features and `y = 2x₁ - x₂ + x₃²`. The score is the relative mean error.

```julia
n      = 500
seeds  = rand(copy(TESTRNG),n)
x      = vcat([[s*2 (s-3)^2 s/2 0.2-s] for s in seeds]...)
y      = [r[1]*2-r[2]+r[3]^2 for r in eachrow(x)]
results = DataFrame(model=String[],time_s=Float64[],memory_MiB=Float64[],allocs=Int[],
                    rme_train=Float64[],rme_train_std=Float64[],rme_test=Float64[],rme_test_std=Float64[])

## BetaML models
bml_models = OrderedDict("DT" => DecisionTreeEstimator(rng=copy(TESTRNG),verbosity=NONE),
                         "RF" => RandomForestEstimator(rng=copy(TESTRNG),verbosity=NONE),
                         "NN" => NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE))
for (name,m) in bml_models
    perf   = bmark(() -> fit!(deepcopy(m),x,y))
    scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
        m2 = deepcopy(m)
        fit!(m2,xtrain,ytrain)
        (relative_mean_error(ytrain,predict(m2,xtrain)), relative_mean_error(ytest,predict(m2,xtest)))
    end
    push!(results,[name,perf...,scores...])
end

## DecisionTree.jl models
dt_models = OrderedDict("DT (DecisionTree.jl)" => DecisionTree.DecisionTreeRegressor(rng=copy(TESTRNG)),
                        "RF (DecisionTree.jl)" => DecisionTree.RandomForestRegressor(n_trees=30,partial_sampling=1.0,rng=copy(TESTRNG)))
for (name,m) in dt_models
    perf   = bmark(() -> DecisionTree.fit!(deepcopy(m),x,y))
    scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
        m2 = deepcopy(m)
        DecisionTree.fit!(m2,xtrain,ytrain)
        (relative_mean_error(ytrain,DecisionTree.predict(m2,xtrain)), relative_mean_error(ytest,DecisionTree.predict(m2,xtest)))
    end
    push!(results,[name,perf...,scores...])
end

## Flux neural network (same structure, batch size and number of epochs as the BetaML default one)
function flux_regressor(x,y;epochs=200)
    Random.seed!(123)
    model     = Flux.Chain(Flux.Dense(4=>8,Flux.relu),Flux.Dense(8=>8,Flux.relu),Flux.Dense(8=>1))
    loader    = Flux.DataLoader((Float32.(permutedims(x)),Float32.(permutedims(y))),batchsize=16,shuffle=true)
    opt_state = Flux.setup(Flux.Adam(),model)
    for e in 1:epochs
        Flux.train!((m,xb,yb) -> Flux.mse(m(xb),yb),model,loader,opt_state)
    end
    return model
end
flux_predict(model,x) = vec(model(Float32.(permutedims(x))))

perf   = bmark(() -> flux_regressor(x,y))
scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
    model = flux_regressor(xtrain,ytrain)
    (relative_mean_error(ytrain,flux_predict(model,xtrain)), relative_mean_error(ytest,flux_predict(model,xtest)))
end
push!(results,["NN (Flux.jl)",perf...,scores...])

results
```

Results:

| model | time_s | memory_MiB | allocs | rme_train | rme_train_std | rme_test | rme_test_std |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DT | 0.004473 | 3.784 | 42066 | 0.0007237 | 2.766e-05 | 0.005161 | 0.0006102 |
| RF | 0.0284 | 68.83 | 944881 | 0.001241 | 4.599e-05 | 0.003096 | 0.0004249 |
| NN | 0.103 | 248.5 | 2345521 | 0.001173 | 5.529e-05 | 0.001195 | 0.0002459 |
| DT (DecisionTree.jl) | 0.0001026 | 0.06716 | 682 | 0.006484 | 0.0002145 | 0.008784 | 0.001354 |
| RF (DecisionTree.jl) | 0.0005902 | 2.533 | 19359 | 0.002697 | 0.0001087 | 0.00386 | 0.000828 |
| NN (Flux.jl) | 0.4456 | 268.9 | 2807148 | 0.002658 | 5.282e-05 | 0.002654 | 0.0003244 |

## Classification

A binary diagnostic classification of breast cancer (569 records, 30 features) from the [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The score is the accuracy.

```julia
bcancer = readdlm(joinpath(DATADIR,"breast_wisconsin","wdbc.data"),',')
x       = fit!(Scaler(),convert(Matrix{Float64},bcancer[:,3:end]))
y       = convert(Vector{String},bcancer[:,2])
ohm     = OneHotEncoder()
yoh     = fit!(ohm,y)
results = DataFrame(model=String[],time_s=Float64[],memory_MiB=Float64[],allocs=Int[],
                    acc_train=Float64[],acc_train_std=Float64[],acc_test=Float64[],acc_test_std=Float64[])

## BetaML models (the neural network is trained with the one-hot encoded labels)
bml_models = OrderedDict("DT"    => DecisionTreeEstimator(rng=copy(TESTRNG),verbosity=NONE),
                         "RF"    => RandomForestEstimator(rng=copy(TESTRNG),verbosity=NONE),
                         "NN"    => NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE),
                         "Perc"  => PerceptronClassifier(rng=copy(TESTRNG),verbosity=NONE),
                         "KPerc" => KernelPerceptronClassifier(rng=copy(TESTRNG),verbosity=NONE),
                         "Peg"   => PegasosClassifier(rng=copy(TESTRNG),verbosity=NONE))
for (name,m) in bml_models
    isnn   = name == "NN"
    perf   = bmark(() -> fit!(deepcopy(m),x,isnn ? yoh : y))
    scores = cv_scores([x,y,yoh]) do (xtrain,ytrain,yohtrain),(xtest,ytest,yohtest),rng
        m2 = deepcopy(m)
        fit!(m2,xtrain,isnn ? yohtrain : ytrain)
        if isnn
            (accuracy(mode(yohtrain),mode(predict(m2,xtrain))), accuracy(mode(yohtest),mode(predict(m2,xtest))))
        else
            (accuracy(ytrain,predict(m2,xtrain)), accuracy(ytest,predict(m2,xtest)))
        end
    end
    push!(results,[name,perf...,scores...])
end

## DecisionTree.jl models
dt_models = OrderedDict("DT (DecisionTree.jl)" => DecisionTree.DecisionTreeClassifier(rng=copy(TESTRNG)),
                        "RF (DecisionTree.jl)" => DecisionTree.RandomForestClassifier(n_trees=30,partial_sampling=1.0,rng=copy(TESTRNG)))
for (name,m) in dt_models
    perf   = bmark(() -> DecisionTree.fit!(deepcopy(m),x,y))
    scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
        m2 = deepcopy(m)
        DecisionTree.fit!(m2,xtrain,ytrain)
        (accuracy(ytrain,DecisionTree.predict(m2,xtrain)), accuracy(ytest,DecisionTree.predict(m2,xtest)))
    end
    push!(results,[name,perf...,scores...])
end

## Flux neural network
function flux_classifier(x,yoh;epochs=200)
    Random.seed!(123)
    model     = Flux.Chain(Flux.Dense(30=>45,Flux.relu),Flux.Dense(45=>45,Flux.relu),Flux.Dense(45=>2))
    loader    = Flux.DataLoader((Float32.(permutedims(x)),Float32.(permutedims(yoh))),batchsize=16,shuffle=true)
    opt_state = Flux.setup(Flux.Adam(),model)
    for e in 1:epochs
        Flux.train!((m,xb,yb) -> Flux.logitcrossentropy(m(xb),yb),model,loader,opt_state)
    end
    return model
end
flux_predict(model,x) = Flux.onecold(model(Float32.(permutedims(x)))) # column of the predicted class

perf   = bmark(() -> flux_classifier(x,yoh))
scores = cv_scores([x,yoh]) do (xtrain,yohtrain),(xtest,yohtest),rng
    model = flux_classifier(xtrain,yohtrain)
    (accuracy(mode(yohtrain),flux_predict(model,xtrain)), accuracy(mode(yohtest),flux_predict(model,xtest)))
end
push!(results,["NN (Flux.jl)",perf...,scores...])

results
```

Results:

| model | time_s | memory_MiB | allocs | acc_train | acc_train_std | acc_test | acc_test_std |
|:---|---:|---:|---:|---:|---:|---:|---:|
| DT | 0.4138 | 120.5 | 814248 | 0.9992 | 0.001887 | 0.9104 | 0.02664 |
| RF | 0.1861 | 443.9 | 2766313 | 0.9996 | 0.0008227 | 0.9613 | 0.02452 |
| NN | 1.173 | 3634 | 3908317 | 0.9994 | 0.0009434 | 0.9719 | 0.02059 |
| Perc | 0.05606 | 299.7 | 670927 | 0.99 | 0.005782 | 0.9579 | 0.02368 |
| KPerc | 1.005 | 1227 | 13629781 | 1 | 0 | 0.9404 | 0.03329 |
| Peg | 0.1215 | 519.2 | 2185128 | 0.9408 | 0.01851 | 0.9386 | 0.03122 |
| DT (DecisionTree.jl) | 0.004519 | 0.05252 | 308 | 1 | 0 | 0.9263 | 0.04586 |
| RF (DecisionTree.jl) | 0.002506 | 5.727 | 8771 | 1 | 0 | 0.9648 | 0.02339 |
| NN (Flux.jl) | 0.5796 | 549.6 | 4041958 | 1 | 0 | 0.9719 | 0.01695 |

## Clustering

Clustering of the [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) (150 records, 4 features) in 3 clusters. The algorithms use only the features, while the species are used only to compute the accuracy of the clusters (`acc`, with the best matching between clusters and species). We also report the silhouette score (`sil`) of the clusters. The scores are computed on the training subsets of the cross-validation.

```julia
iris     = readdlm(joinpath(DATADIR,"iris_shuffled.csv"),',',skipstart=1)
x        = convert(Array{Float64,2},iris[:,1:4])
ystrings = convert(Array{String,1},iris[:,5])
y        = fit!(OrdinalEncoder(categories=unique(ystrings)),ystrings)
results  = DataFrame(model=String[],time_s=Float64[],memory_MiB=Float64[],allocs=Int[],
                     acc=Float64[],acc_std=Float64[],sil=Float64[],sil_std=Float64[])

## The functions returning the cluster of each record, for BetaML and the other packages
cl_functions = OrderedDict(
    "KMeans"                    => x -> fit!(KMeansClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x),
    "KMedoids"                  => x -> fit!(KMedoidsClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x),
    "GMM"                       => x -> mode(fit!(GaussianMixtureClusterer(n_classes=3,rng=copy(TESTRNG),verbosity=NONE),x)),
    "KMeans (Clustering.jl)"    => x -> Clustering.kmeans(permutedims(x),3).assignments,
    "KMedoids (Clustering.jl)"  => x -> Clustering.kmedoids(pairwise(x),3).assignments,
    "HC (Clustering.jl)"        => x -> Clustering.cutree(Clustering.hclust(pairwise(x)),k=3),
    "DBSCAN (Clustering.jl)"    => x -> Clustering.dbscan(permutedims(x),0.8).assignments,
    "GMM (GaussianMixtures.jl)" => x -> mode(GaussianMixtures.gmmposterior(GaussianMixtures.GMM(3,x),x)[1]),
)
for (name,f) in cl_functions
    Random.seed!(123)
    with_logger(NullLogger()) do # GaussianMixtures.jl is quite verbose
        perf   = bmark(() -> f(x))
        scores = cv_scores([x,y]) do (xtrain,ytrain),(xtest,ytest),rng
            ŷtrain = f(xtrain)
            (accuracy(ytrain,ŷtrain,ignorelabels=true), mean(silhouette(pairwise(xtrain),ŷtrain)))
        end
        push!(results,[name,perf...,scores...])
    end
end

results
```

Results:

| model | time_s | memory_MiB | allocs | acc | acc_std | sil | sil_std |
|:---|---:|---:|---:|---:|---:|---:|---:|
| KMeans | 0.0001862 | 0.4304 | 9207 | 0.8911 | 0.008589 | 0.5523 | 0.009396 |
| KMedoids | 0.003025 | 12.28 | 268000 | 0.9022 | 0.0163 | 0.541 | 0.01404 |
| GMM | 0.00468 | 15.97 | 342850 | 0.9148 | 0.01889 | 0.5437 | 0.01411 |
| KMeans (Clustering.jl) | 2.415e-05 | 0.02069 | 58 | 0.8911 | 0.007847 | 0.5509 | 0.01001 |
| KMedoids (Clustering.jl) | 0.002078 | 3.465 | 101993 | 0.7911 | 0.1738 | 0.5316 | 0.0184 |
| HC (Clustering.jl) | 0.002183 | 3.708 | 102274 | 0.68 | 0.009106 | 0.5107 | 0.01048 |
| DBSCAN (Clustering.jl) | 9.637e-05 | 0.1847 | 1143 | 0.68 | 0.009106 | 0.5107 | 0.01048 |
| GMM (GaussianMixtures.jl) | 0.000189 | 0.3131 | 2432 | 0.9081 | 0.0153 | 0.5491 | 0.01512 |

## Neural networks on CPU and GPU

BetaML and Flux neural networks trained on the CPU and on a NVIDIA GPU (with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)), with the same network structure, data (in `Float32`), optimisation algorithm (ADAM) and batch size. The times refer to the training only, with the data already on the device and excluding the compilation, and are measured on a single run. The scores are computed on a test set.
- Dense network: a regression of 25000 records with 50 features, with two inner layers of 256 neurons, trained for 5 epochs. The score is the relative mean error.
- Convolutional network: the classification of the 70000 images of the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits (60000 used for training), with the network of the [Image recognition tutorial](https://github.com/sylvaticus/BetaML.jl/blob/main/docs/src/tutorials/Image%20recognition/Image_recognition.jl), trained for 2 epochs. The score is the accuracy.

```julia
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
```

Results:

| task | model | device | time_s | score |
|:---|:---|:---|---:|---:|
| Dense | BetaML | CPU | 1.491 | 0.07991 |
| Dense | Flux.jl | CPU | 0.9485 | 0.08164 |
| Convolutional (MNIST) | BetaML | CPU | 29.56 | 0.9462 |
| Convolutional (MNIST) | Flux.jl | CPU | 11.12 | 0.9515 |
| Dense | BetaML | GPU | 0.4544 | 0.07979 |
| Dense | Flux.jl | GPU | 0.3436 | 0.08164 |
| Convolutional (MNIST) | BetaML | GPU | 9.285 | 0.9456 |
| Convolutional (MNIST) | Flux.jl | GPU | 1.37 | 0.9524 |
