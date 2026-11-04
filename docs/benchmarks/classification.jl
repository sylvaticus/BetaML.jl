# A binary diagnostic classification of breast cancer (569 records, 30 features) from the [Wisconsin Breast Cancer Database](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic). The score is the accuracy.
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
