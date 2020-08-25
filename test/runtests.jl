using Test

if "all" in ARGS
    println("Running ALL tests available")
else
    println("Running normal testing")
end

include("Utils_test.jl")
include("Trees_test.jl")
include("Nn_tests.jl")
include("Perceptron_test.jl")
include("Clustering_test.jl")
