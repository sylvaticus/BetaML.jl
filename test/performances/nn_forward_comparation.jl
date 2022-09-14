using PyCall
using Flux
using BetaML.Nn
using BenchmarkTools
using Distributions
import LinearAlgebra.BLAS

BLAS.set_num_threads(1)

torch = pyimport("torch")
torch.set_num_threads(1)

NN = torch.nn.Sequential(
    torch.nn.Linear(8, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 2),
    torch.nn.Tanh()
)

torch_nn(in) = NN(in)

Flux_nn = Chain(Dense(8,64,tanh),
                Dense(64,32,tanh),
                Dense(32,2,tanh))

BetaML_nn = buildNetwork([
               DenseLayer(8,64,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(8+64),sqrt(6)/sqrt(8+64)),64,8),wb=rand(Uniform(-sqrt(6)/sqrt(8+64),sqrt(6)/sqrt(8+64)),64)),
               DenseLayer(64,32,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(64+32),sqrt(6)/sqrt(64+32)),32,64),wb=rand(Uniform(-sqrt(6)/sqrt(64+32),sqrt(6)/sqrt(64+32)),32)),
               DenseLayer(32,2,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(32+2),sqrt(6)/sqrt(32+2)),2,32),wb=rand(Uniform(-sqrt(6)/sqrt(32+2),sqrt(6)/sqrt(32+2)),2))],
               squared_cost,name="Bike sharing regression model")

BetaML_nn = buildNetwork([
              DenseLayer(8,64,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(8+64),sqrt(6)/sqrt(8+64)),64,8),wb=zeros(64)),
              DenseLayer(64,32,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(64+32),sqrt(6)/sqrt(64+32)),32,64),wb=zeros(32)),
              DenseLayer(32,2,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(32+2),sqrt(6)/sqrt(32+2)),2,32),wb=zeros(2))],
              squared_cost,name="Bike sharing regression model")

BetaML_nn2 = buildNetwork([
            DenseLayer(8,64,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(8+64),sqrt(6)/sqrt(8+64)),64,8),wb=zeros(64), df=dtanh),
            DenseLayer(64,32,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(64+32),sqrt(6)/sqrt(64+32)),32,64),wb=zeros(32),df=dtanh),
            DenseLayer(32,2,f=tanh,w=rand(Uniform(-sqrt(6)/sqrt(32+2),sqrt(6)/sqrt(32+2)),2,32),wb=zeros(2),df=dtanh)],
            squared_cost,name="Bike sharing regression model",dcf=dSquaredCost)

for i in [1, 10, 100, 1000]
    println("Batch size: $i")
    torch_in = torch.rand(i,8)
    flux_in = rand(Float32,8,i)
    betaml_in = rand(Float32,i,8)
    print("pytorch:")
    @btime torch_nn($torch_in)
    print("flux   :")
    @btime Flux_nn($flux_in)
    print("betaml   :")
    @btime predict($BetaML_nn,$betaml_in)
    print("betaml2   :")
    @btime predict($BetaML_nn2,$betaml_in)
end

#= Output:
Batch size: 1
pytorch:  89.920 μs (6 allocations: 192 bytes)
flux   :  3.426 μs (6 allocations: 1.25 KiB)
betaml   :  3.046 μs (19 allocations: 3.55 KiB)
betaml2   :  3.089 μs (19 allocations: 3.55 KiB)
Batch size: 10
pytorch:  100.737 μs (6 allocations: 192 bytes)
flux   :  19.743 μs (6 allocations: 8.22 KiB)
betaml   :  33.137 μs (181 allocations: 34.77 KiB)
betaml2   :  32.259 μs (181 allocations: 34.77 KiB)
Batch size: 100
pytorch:  132.689 μs (6 allocations: 192 bytes)
flux   :  184.807 μs (8 allocations: 77.16 KiB)
betaml   :  306.326 μs (1801 allocations: 347.08 KiB)
betaml2   :  310.554 μs (1801 allocations: 347.08 KiB)
Batch size: 1000
pytorch:  392.295 μs (6 allocations: 192 bytes)
flux   :  1.838 ms (10 allocations: 766.19 KiB)
betaml   :  3.172 ms (18490 allocations: 3.40 MiB)
betaml2   :  3.116 ms (18490 allocations: 3.40 MiB)
=#
