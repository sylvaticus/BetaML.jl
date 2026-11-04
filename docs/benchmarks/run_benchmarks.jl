# Run the BetaML benchmarks and write the documentation page `docs/src/Benchmarks.md`.
#
# The page is not computed when the documentation is built (by `docs/make.jl`): run this script
# from time to time to update it, and commit the updated page.
#
# Usage, from the BetaML folder: `julia -t auto docs/benchmarks/run_benchmarks.jl [sections...]`
# - sections: some of "regression", "classification", "clustering" and "neural_networks_gpu"
#   [def: all of them]
#
# The benchmarks run in the environment of this folder (BetaML, the other packages to compare and
# CUDA.jl), that requires Julia 1.11 or newer. The GPU benchmarks run only if a functional NVIDIA
# GPU is found, otherwise only the CPU ones are reported.
# Each benchmark is defined in its own file in this folder: the initial comment lines of the file
# are its description in the page, the rest is its code, whose last value is the table of results.

VERSION >= v"1.11" || error("The benchmarks require Julia 1.11 or newer")
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Dates, Printf, DataFrames
import BetaML, Flux, DecisionTree, Clustering, GaussianMixtures, CUDA

const SECTIONS = ["regression"          => "Regression",
                  "classification"      => "Classification",
                  "clustering"          => "Clustering",
                  "neural_networks_gpu" => "Neural networks on CPU and GPU"]
const OUTFILE  = joinpath(@__DIR__,"..","src","Benchmarks.md")

selected = isempty(ARGS) ? first.(SECTIONS) : ARGS
all(in(first.(SECTIONS)),selected) || error("Unknown section(s): $(setdiff(selected,first.(SECTIONS))). Available: $(first.(SECTIONS))")

"Split a benchmark file in its description (the initial comment lines) and its code"
function split_source(file)
    lines = readlines(file)
    i     = something(findfirst(l -> !startswith(l,"#"),lines),length(lines)+1)
    desc  = [replace(l,r"^# ?" => "") for l in lines[1:i-1]]
    return join(desc,"\n"), strip(join(lines[i:end],"\n"))
end

fmt(x::AbstractFloat) = @sprintf("%.4g",x)
fmt(x) = string(x)
"Markdown table of a DataFrame"
function md_table(df)
    header = "| " * join(names(df)," | ") * " |"
    align  = "|" * join([eltype(c) <: Number ? "---:" : ":---" for c in eachcol(df)],"|") * "|"
    rows   = ["| " * join(fmt.(collect(r))," | ") * " |" for r in eachrow(df)]
    return join([header,align,rows...],"\n")
end

"Run a benchmark file in its own module, after the common setup, returning its table of results"
function run_section(name)
    mod = Module(Symbol(name))
    Base.include(mod,joinpath(@__DIR__,"common.jl"))
    return Base.include(mod,joinpath(@__DIR__,"$name.jl"))
end

results = Dict{String,DataFrame}()
for name in selected
    println(now(), " *** Running the $name benchmarks...")
    t = @elapsed results[name] = run_section(name)
    println(now(), " *** ...done in $(round(t/60,digits=1)) minutes:")
    println(results[name])
end

# Writing the page
gpu      = CUDA.functional() ? CUDA.name(CUDA.device()) : "none (the GPU benchmarks have not been run)"
packages = join(["$(nameof(p)) v$(pkgversion(p))" for p in (BetaML,Flux,DecisionTree,Clustering,GaussianMixtures,CUDA)],", ")
common_desc, common_code = split_source(joinpath(@__DIR__,"common.jl"))

page = """
# [BetaML Benchmarks](@id benchmarks)

This page reports some basic benchmarks of BetaML models, together with those of other leading Julia packages for the same tasks. It aims to check for regressions across BetaML versions and to give an idea of the relative performances of the packages, not to be a rigorous comparison.

!!! note
    This page is not computed when the documentation is built: it is written by the script `docs/benchmarks/run_benchmarks.jl` (see the instructions within it), run on the machine described below. As the times depend on the machine, compare them only within the same version of this page.

## Setup

- Date: $(Dates.format(now(),"yyyy-mm-dd"))
- Julia: v$(VERSION), $(Threads.nthreads()) thread(s)
- CPU: $(Sys.cpu_info()[1].model)
- GPU: $gpu
- Packages: $packages

$common_desc

```julia
$common_code
```
"""
for (name,title) in SECTIONS
    name in keys(results) || continue
    desc, code = split_source(joinpath(@__DIR__,"$name.jl"))
    global page *= """

    ## $title

    $desc

    ```julia
    $code
    ```

    Results:

    $(md_table(results[name]))
    """
end
write(OUTFILE,page)
println(now(), " *** Benchmark page written in $(normpath(OUTFILE))")
