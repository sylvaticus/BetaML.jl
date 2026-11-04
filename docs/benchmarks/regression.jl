# A simple regression over 500 points with 4 features and `y = 2x₁ - x₂ + x₃²`. The score is the relative mean error.
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
