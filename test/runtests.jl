using Test
using BetaML

#using Pkg # seems a Julia bug: can't find pkg `Pkg` !!
#Pkg.activate(@__DIR__)

#=
Choose what to test with `Pkg.test("BetaML", test_args=["Trees","Clustering","all"])`
or `$ julia runtests.jl Trees Clustering all`

Possible values are:
 - Utils
 - Trees
 - Nn
 - Perceptron
 - Clustering
 - GMM
 - Stats
 - Imputation
 - MLJ
 - all

The GPU support of the neural networks is tested by default with JLArrays, a CPU-based
simulation of GPU arrays. To test it on a real GPU add the option `gpu_test_device=CUDA` (or
`AMDGPU`), or set the environment variable BETAML_GPU_TEST_DEVICE, using an environment with
the GPU package. The script `run_gpu_tests.jl` takes care of all this, e.g.:
`$ julia test/run_gpu_tests.jl CUDA` (runs the Nn tests on a NVIDIA GPU)
=#

test_options = filter(a -> occursin("=",a), ARGS)  # options in the form `name=value`
test_groups  = filter(a -> !occursin("=",a), ARGS) # test groups to run
nArgs        = length(test_groups)
GPU_TEST_DEVICE = get(Dict(split.(test_options,"=",limit=2)),"gpu_test_device",get(ENV,"BETAML_GPU_TEST_DEVICE","JLArrays"))

if "all" in ARGS
    println("Running ALL tests available")
else
    println("Running normal testing")
end

# just to reset the file used to save models in several modules..
rm("test.jld2", force=true)

if "all" in ARGS || "Utils" in ARGS || nArgs == 0
    include("Utils_tests.jl")
end
if "all" in ARGS || "Trees" in ARGS || nArgs == 0
    include("Trees_tests.jl")
end
if "all" in ARGS || "Nn" in ARGS || nArgs == 0
    include("Nn_tests.jl")
    include("Nn_gpu_tests.jl") # batched API and GPU support (simulated with JLArrays)
end
if "all" in ARGS || "Perceptron" in ARGS || nArgs == 0
    include("Perceptron_tests.jl")
end
if "all" in ARGS || "Clustering" in ARGS || nArgs == 0
    include("Clustering_tests.jl")
end
if "all" in ARGS || "GMM" in ARGS || nArgs == 0
    include("GMM_tests.jl")
end

if "all" in ARGS || "Stats" in ARGS || nArgs == 0
    #include("Stats_tests.jl") strange errors on github only !!
end
if "all" in ARGS || "Imputation" in ARGS || nArgs == 0
    include("Imputation_tests.jl")
end

if "all" in ARGS || "MLJ" in ARGS # not run by default
    include("MLJ_tests_additional.jl")
end

if "all" in ARGS
    # run optional long tests
    include("Perceptron_tests_additional.jl")
    include("Trees_tests_additional.jl")
    include("Clustering_tests_additional.jl")
    include("MLJ_tests_additional.jl")
end

rm("test.jld2", force=true)
