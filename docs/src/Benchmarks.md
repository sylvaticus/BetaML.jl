# BetaML Benchmarks

This benchmark allows to quickly check for regressions across versions.
As it is run and compiled using GitHub actions, and these may be powered by different computational resources, timing results are normalized using SystemBenchmark.

This page also provides a basic comparison with other leading Julia libraries for the same algorithm, USING DEFAULT VALUES.
This file is intended just for benchmarking, not much as a tutorial, and it doesn't employ a full ML workflow, just the minimum preprocessing such that the algorithms work.

## Benchmark setup
```@setup bmk
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using Test, Statistics, Random, DelimitedFiles
using  DataStructures, DataFrames, BenchmarkTools, StableRNGs, SystemBenchmark
import DecisionTree, Flux
using BetaML

TESTRNG = StableRNG(123)
```

```@example bmk
Threads.nthreads()
```

```@setup bmk
println("*** Computing System benchmarking for normalization of the outputs..")
res  = runbenchmark()
comp = comparetoref(res)
tests = ["FloatMul", "FusedMulAdd", "FloatSin", "VecMulBroad", "CPUMatMul",   
         "MatMulBroad", "3DMulBroad", "FFMPEGH264Write"]                        
avg_factor_to_ref = mean(comp[in.(comp.testname, Ref(tests)), "factor"])
```

```@example bmk
avg_factor_to_ref 
```


## Regression

A simple regression over 500 points with y = x₁²-x₂+x₃²

```@setup bmk
println("*** Benchmarking regression task..")

df_regr = DataFrame(name= String[],time=Float64[],memory=Int64[],allocs=Int64[],mre_train=Float64[],std_train=Float64[],mre_test=Float64[],std_test=Float64[])
n = 500
seeds = rand(copy(TESTRNG),n)
x = vcat([[s*2 (s-3)^2 s/2 0.2-s] for s in seeds]...)
y = [r[1]*2-r[2]+r[3]^2 for r in eachrow(x)]

bml_models = OrderedDict("DT"=>DecisionTreeEstimator(rng=copy(TESTRNG),verbosity=NONE),
                  "RF"=>RandomForestEstimator(rng=copy(TESTRNG),verbosity=NONE),
                  "NN"=>NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE),
);

for (mname,m) in bml_models
    #mname = "DT"
    #m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
    # speed measure 
    println("Processing model $mname ... ")
    bres     = @benchmark fit!(m2,$x,$y) setup=(m2 = deepcopy($m))
    m_time   = median(bres.times)
    m_memory = bres.memory
    m_allocs = bres.allocs
    sampler = KFold(nsplits=10,rng=copy(TESTRNG));
    cv_out = cross_validation([x,y],sampler,return_statistics=false) do trainData,valData,rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    m2 = deepcopy(m)
                    fit!(m2,xtrain,ytrain)
                    ŷtrain  = predict(m2,xtrain)
                    ŷval    = predict(m2,xval)
                    rme_train = relative_mean_error(ytrain,ŷtrain)
                    rme_val = relative_mean_error(yval,ŷval)
                    return (rme_train, rme_val)
    end

    mre_train = mean([r[1] for r in cv_out])
    std_train = std([r[1] for r in cv_out])
    mre_test = mean([r[2] for r in cv_out])
    std_test = std([r[2] for r in cv_out])
    push!(df_regr,[mname, m_time, m_memory, m_allocs, mre_train, std_train, mre_test, std_test])
    @test mre_test <= 0.05
end

### DecisionTree
Random.seed!(123)
dt_models = OrderedDict("DT (DecisionTrees.jl)"=>DecisionTree.DecisionTreeRegressor(),
                        "RF (DecisionTrees.jl)"=>DecisionTree.RandomForestRegressor(),
);

for (mname,m) in dt_models
    #mname = "DT"
    #m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
    # speed measure 
    bres     = @benchmark DecisionTree.fit!(m2,$x,$y) setup=(m2 = deepcopy($m))
    m_time   = median(bres.times)
    m_memory = bres.memory
    m_allocs = bres.allocs
    sampler = KFold(nsplits=10,rng=copy(TESTRNG));
    cv_out = cross_validation([x,y],sampler,return_statistics=false) do trainData,valData,rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    m2 = deepcopy(m)
                    DecisionTree.fit!(m2,xtrain,ytrain)
                    ŷtrain  = DecisionTree.predict(m2,xtrain)
                    ŷval    = DecisionTree.predict(m2,xval)
                    rme_train = relative_mean_error(ytrain,ŷtrain)
                    rme_val = relative_mean_error(yval,ŷval)
                    return (rme_train, rme_val)
    end

    mre_train = mean([r[1] for r in cv_out])
    std_train = std([r[1] for r in cv_out])
    mre_test = mean([r[2] for r in cv_out])
    std_test = std([r[2] for r in cv_out])
    push!(df_regr,[mname, m_time, m_memory, m_allocs, mre_train, std_train, mre_test, std_test])
    @test mre_test <= 0.05
end

### Flux
Random.seed!(123)
l1         = Flux.Dense(4,8,Flux.relu)
l2         = Flux.Dense(8,8,Flux.relu)
l3         = Flux.Dense(8,1,Flux.identity)
Flux_nn    = Flux.Chain(l1,l2,l3)
fluxloss(x, y) = Flux.mse(Flux_nn(x), y)
ps         = Flux.params(Flux_nn)
nndata     = Flux.Data.DataLoader((Float32.(x)', Float32.(y)'), batchsize=16,shuffle=true)

bres     = @benchmark [Flux.train!(fluxloss, ps2, $nndata, Flux.ADAM()) for i in 1:200] setup=(ps2 = deepcopy($ps))
m_time   = median(bres.times)
m_memory = bres.memory
m_allocs = bres.allocs

sampler = KFold(nsplits=10,rng=copy(TESTRNG));
cv_out = cross_validation([x,y],sampler,return_statistics=false) do trainData,valData,rng
                (xtrain,ytrain) = trainData; (xval,yval) = valData
                m2         = deepcopy(Flux_nn)
                ps2        = Flux.params(m2)
                fluxloss2(x, y) = Flux.mse(m2(x), y)
                nndata     = Flux.Data.DataLoader((Float32.(xtrain)', Float32.(ytrain)'), batchsize=16,shuffle=true)
                [Flux.train!(fluxloss2, ps2, nndata, Flux.ADAM()) for i in 1:200] 
                ŷtrain     = m2(xtrain')'
                ŷval       = m2(xval')'
                rme_train = relative_mean_error(ytrain,ŷtrain)
                rme_val = relative_mean_error(yval,ŷval)
                return (rme_train, rme_val)
end
mre_train = mean([r[1] for r in cv_out])
std_train = std([r[1] for r in cv_out])
mre_test = mean([r[2] for r in cv_out])
std_test = std([r[2] for r in cv_out])
push!(df_regr,["NN (Flux.jl)", m_time, m_memory, m_allocs, mre_train, std_train, mre_test, std_train])
@test mre_test <= 0.05

df_regr.time .= df_regr.time ./ avg_factor_to_ref
```

```@example bmk
df_regr
```

## Classification

A dicotomic diagnostic breast cancer classification based on the [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).


```@setup bmk
println("*** Benchmarking classification task..")

bcancer_file = joinpath(@__DIR__,"..","..","test","data","breast_wisconsin","wdbc.data")
bcancer  = readdlm(bcancer_file,',')
x = fit!(Scaler(),convert(Matrix{Float64},bcancer[:,3:end]))
y = convert(Vector{String},bcancer[:,2])
ohm = OneHotEncoder()
yoh = fit!(ohm,y)
df_class = DataFrame(name= String[],time=Float64[],memory=Int64[],allocs=Int64[],acc_train=Float64[],std_train=Float64[],acc_test=Float64[],std_test=Float64[])

bml_models = OrderedDict("DT"=>DecisionTreeEstimator(rng=copy(TESTRNG),verbosity=NONE),
                  "RF"=>RandomForestEstimator(rng=copy(TESTRNG),verbosity=NONE),
                  "NN"=>NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE),
                  "Perc"=>PerceptronClassifier(rng=copy(TESTRNG),verbosity=NONE),
                  "KPerc"=>KernelPerceptronClassifier(rng=copy(TESTRNG),verbosity=NONE),
                  "Peg"=>PegasosClassifier(rng=copy(TESTRNG),verbosity=NONE),
);

for (mname,m) in bml_models
    #mname = "NN"
    #m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
    # speed measure 
    println("Processing model $mname ... ")
    if mname == "NN"
        bres     = @benchmark fit!(m2,$x,$yoh) setup=(m2 = deepcopy($m))
    else
        bres     = @benchmark fit!(m2,$x,$y) setup=(m2 = deepcopy($m))
    end
    m_time   = median(bres.times)
    m_memory = bres.memory
    m_allocs = bres.allocs
    sampler = KFold(nsplits=10,rng=copy(TESTRNG));
    cv_out = cross_validation([x,y,yoh],sampler,return_statistics=false) do trainData,valData,rng
                    (xtrain,ytrain,yohtrain) = trainData; (xval,yval,yohval) = valData
                    m2 = deepcopy(m)
                    if mname == "NN"
                        fit!(m2,xtrain,yohtrain)
                    else
                        fit!(m2,xtrain,ytrain)
                    end
                    ŷtrain  = predict(m2,xtrain)
                    ŷval    = predict(m2,xval)
                    if mname == "NN"
                        acc_train = accuracy(BetaML.mode(yohtrain),BetaML.mode(ŷtrain))
                        acc_val = accuracy(BetaML.mode(yohval),BetaML.mode(ŷval))
                    else
                        acc_train = accuracy(ytrain,ŷtrain)
                        acc_val = accuracy(yval,ŷval)
                    end
                    return (acc_train, acc_val)
    end

    acc_train = mean([r[1] for r in cv_out])
    std_train = std([r[1] for r in cv_out])
    acc_test = mean([r[2] for r in cv_out])
    std_test = std([r[2] for r in cv_out])
    push!(df_class,[mname, m_time, m_memory, m_allocs, acc_train, std_train, acc_test, std_test])
    @test acc_test >= 0.6
end


Random.seed!(123)
dt_models = OrderedDict("DT (DT.jl)"=>DecisionTree.DecisionTreeClassifier(),
                 "RF (DT.jl)"=>DecisionTree.RandomForestClassifier(),
);


for (mname,m) in dt_models
    #mname = "DT"
    #m = NeuralNetworkEstimator(rng=copy(TESTRNG),verbosity=NONE)
    # speed measure 
    bres     = @benchmark DecisionTree.fit!(m2,$x,$y) setup=(m2 = deepcopy($m))
    m_time   = median(bres.times)
    m_memory = bres.memory
    m_allocs = bres.allocs
    sampler = KFold(nsplits=10,rng=copy(TESTRNG));
    cv_out = cross_validation([x,y],sampler,return_statistics=false) do trainData,valData,rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    m2 = deepcopy(m)
                    DecisionTree.fit!(m2,xtrain,ytrain)
                    ŷtrain  = DecisionTree.predict(m2,xtrain)
                    ŷval    = DecisionTree.predict(m2,xval)
                    acc_train = accuracy(ytrain,ŷtrain)
                    acc_val = accuracy(yval,ŷval)
                    return (acc_train, acc_val)
    end

    acc_train = mean([r[1] for r in cv_out])
    std_train = std([r[1] for r in cv_out])
    acc_test = mean([r[2] for r in cv_out])
    std_test = std([r[2] for r in cv_out])
    push!(df_class,[mname, m_time, m_memory, m_allocs, acc_train, std_train, acc_test, std_test])
    @test acc_test >= 0.8
end

### Flux
Random.seed!(123)
ohm = OneHotEncoder()
yoh = fit!(ohm,y)
l1         = Flux.Dense(30,45,Flux.relu)
l2         = Flux.Dense(45,45,Flux.relu)
l3         = Flux.Dense(45,2,Flux.identity)
Flux_nn    = Flux.Chain(l1,l2,l3)
fluxloss(lx, ly) = Flux.logitcrossentropy(Flux_nn(lx), ly)
ps         = Flux.params(Flux_nn)
nndata     = Flux.Data.DataLoader((Float32.(x)', Float32.(yoh)'), batchsize=15,shuffle=true)
bres       = @benchmark [Flux.train!(fluxloss, ps2, $nndata, Flux.ADAM()) for i in 1:200] setup=(ps2 = deepcopy($ps))
m_time     = median(bres.times)
m_memory   = bres.memory
m_allocs   = bres.allocs

sampler = KFold(nsplits=10,rng=copy(TESTRNG));
cv_out = cross_validation([x,y,yoh],sampler,return_statistics=false) do trainData,valData,rng
                (xtrain,ytrain,yohtrain) = trainData; (xval,yval,yohval) = valData
                m2         = deepcopy(Flux_nn)
                ps2        = Flux.params(m2)
                fluxloss2(lx, ly) = Flux.logitcrossentropy(m2(lx), ly)
                nndata     = Flux.Data.DataLoader((Float32.(xtrain)', Float32.(yohtrain)'), batchsize=16,shuffle=true)
                [Flux.train!(fluxloss2, ps2, nndata, Flux.ADAM()) for i in 1:200] 
                ŷtrain     = inverse_predict(ohm,fit!(OneHotEncoder(),mode(m2(xtrain')')))
                ŷval       = inverse_predict(ohm,fit!(OneHotEncoder(),mode(m2(xval')')))
                acc_train  = accuracy(ytrain,ŷtrain)
                acc_val  = accuracy(yval,ŷval)
                return (acc_train, acc_val)
end
acc_train = mean([r[1] for r in cv_out])
std_train = std([r[1] for r in cv_out])
acc_test = mean([r[2] for r in cv_out])
std_test = std([r[2] for r in cv_out])
push!(df_class,["NN (Flux.jl)", m_time, m_memory, m_allocs, acc_train, std_train, acc_test, std_test])
@test acc_test >= 0.8

df_class.time .= df_class.time ./ avg_factor_to_ref
```

```@example bmk
df_class
```

## Clustering

TODO :-)

## Missing imputation

TODO :-)