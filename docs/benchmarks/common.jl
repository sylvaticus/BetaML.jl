# All benchmarks share this setup. Unless otherwise noted, times, memory and allocations refer to the training of the model on the whole dataset and are medians over several runs (measured with [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)), while the scores are means and standard deviations over a 10-folds cross-validation.
# We tried to match the hyperparameters of the other packages to the BetaML default ones, but differences remain, and they may explain in part the different performances and scores.
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
