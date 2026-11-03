# Run the neural network tests on a real GPU, e.g. `julia test/run_gpu_tests.jl CUDA`
#
# Usage: julia test/run_gpu_tests.jl [GPU package] [test groups...]
# - GPU package: "CUDA" (default) or "AMDGPU"
# - test groups: as in runtests.jl [default: "Nn"]
#
# GPU packages are not test dependencies (so that the normal tests, e.g. on GitHub Actions, don't
# need them). This script creates (or updates) the environment `test/gpu_env_<GPU package>`, made of
# the test dependencies plus the GPU package, and runs the tests there with the option
# `gpu_test_device=<GPU package>`.

using Pkg

gpupkg = isempty(ARGS) ? "CUDA" : ARGS[1]
groups = length(ARGS) > 1 ? ARGS[2:end] : ["Nn"]
gpupkg in ("CUDA","AMDGPU") || error("Unsupported GPU package \"$gpupkg\". Use \"CUDA\" or \"AMDGPU\".")

testdir = @__DIR__
envdir  = joinpath(testdir,"gpu_env_$gpupkg")
mkpath(envdir)
cp(joinpath(testdir,"Project.toml"),joinpath(envdir,"Project.toml"),force=true)
Pkg.activate(envdir)
Pkg.develop(path=dirname(testdir))
Pkg.add(gpupkg)
Pkg.instantiate()

cmd = `$(Base.julia_cmd()) --project=$envdir $(joinpath(testdir,"runtests.jl")) $groups gpu_test_device=$gpupkg`
println("Running: $cmd")
cd(testdir) do # as Pkg.test does
    run(cmd)
end
