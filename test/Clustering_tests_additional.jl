using Test
using DelimitedFiles, LinearAlgebra
import MLJBase
const Mlj = MLJBase
using StableRNGs
rng = StableRNG(123)
using BetaML.Clustering

println("*** Additional testing for the Clustering algorithms...")

#println("Testing MLJ interface for Clustering models....")
# evaluate seem not supported for unsupervised models
