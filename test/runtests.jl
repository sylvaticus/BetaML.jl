using Test
using BetaML

#using Pkg # seems Julia bug: can't find pkg `Pkg` !!
#Pkg.activate(@__DIR__)

# choose what to test with `Pkg.test("BetaML", test_args=["Trees","Clustering","all"])``
# or `$ julia runtests.jl Trees Clustering all`

nArgs = length(ARGS)

if "all" in ARGS
    println("Running ALL tests available")
else
    println("Running normal testing")
end

# just to reset the file used to save models in several mudules..
rm("test.jld2", force=true)

if "all" in ARGS || "Utils" in ARGS || nArgs == 0
    include("Utils_tests.jl")
end
if "all" in ARGS || "Trees" in ARGS || nArgs == 0
    include("Trees_tests.jl")
end
if "all" in ARGS || "Nn" in ARGS || nArgs == 0
    include("Nn_tests.jl")
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

if "all" in ARGS
    # run optional long tests
    include("Perceptron_tests_additional.jl")
    include("Trees_tests_additional.jl")
    include("Clustering_tests_additional.jl")
end

rm("test.jld2", force=true)
