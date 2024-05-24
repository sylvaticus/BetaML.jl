# temp file 
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
ENV["PYTHON"] = "" 
using Test, Statistics, Random, DelimitedFiles, Logging
using  DataStructures, DataFrames, BenchmarkTools, StableRNGs, SystemBenchmark
import DecisionTree, Flux
import Clustering, GaussianMixtures
using BetaML
using Conda
using PyCall
pyimport_conda("sklearn", "sklearn", "conda-forge") 

TESTRNG = StableRNG(123)




println("*** Benchmarking regression task..")

bm_regression = DataFrame(name= String[],time=Float64[],memory=Int64[],allocs=Int64[],mre_train=Float64[],std_train=Float64[],mre_test=Float64[],std_test=Float64[])
n             = 500
seeds         = rand(copy(TESTRNG),n)
x             = vcat([[s*2 (s-3)^2 s/2 0.2-s] for s in seeds]...)
y             = [r[1]*2-r[2]+r[3]^2 for r in eachrow(x)]


Random.seed!(123)
dt_models = OrderedDict("DT (DecisionTrees.jl)"=>DecisionTree.DecisionTreeRegressor(rng=copy(TESTRNG)),
                        "RF (DecisionTrees.jl)"=>DecisionTree.RandomForestRegressor(n_trees=30, rng=copy(TESTRNG)),
);

# DT:
# set of regression parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)


# RF:
# set of regression build_forest() parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
#              multi-threaded forests must be seeded with an `Int`

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
    push!(bm_regression,[mname, m_time, m_memory, m_allocs, mre_train, std_train, mre_test, std_test])
    @test mre_test <= 0.05
end