using Test
using DelimitedFiles, LinearAlgebra, Statistics #, MLDatasets

#using StableRNGs
#rng = StableRNG(123)
using BetaML

import BetaML.Nn: buildNetwork, forward, loss, backward, train!, get_nparams, _get_n_layers_weights
import BetaML.Nn: ConvLayer, ReshaperLayer, _zComp!
TESTRNG = FIXEDRNG # This could change...
#TESTRNG = StableRNG(123)



x        = reshape(1:100*3*3*2,100,3*3*2) ./ 100
#x = rand(100,18)
y        = [norm(r[1:9])+2*norm(r[10:18],2) for r in eachrow(x) ]
(N,D)    = size(x)
l1       = ReshaperLayer((D,1),(3,3,2))
l2       = ConvLayer((3,3),(2,2),2,3,rng=copy(TESTRNG),f=identity)
l3       = ConvLayer(size(l2)[2],(2,2),8,rng=copy(TESTRNG), f=identity)
l4       = ReshaperLayer(size(l3)[2])
l5       = DenseLayer(size(l4)[2][1],1,f=relu, rng=copy(TESTRNG))
layers   = [l1,l2,l3,l4,l5]
mynn     = buildNetwork(layers,squared_cost,name="Regression with a convolutional layer")
preprocess!(mynn)
predict(mynn,x[1,:]')

l1out = forward(l1,x[1,:])
l2out = forward(l2,l1out)
@btime forward(l2,l1out)

@btime forward($l2,$l1out)

_, output_size = size(l2)
z = zeros(output_size)
@btime _zComp!($z,$l1out,$l2.weight,$l2.bias,$l2.y_ids,$l2.x_ids,$l2.w_ids,$l2.usebias) 
@btime _zComp!($z,$l2,$l1out)

@btime zeros(size($l2)[2])


train!(mynn,x,y,epochs=60,verbosity=NONE,rng=copy(TESTRNG))
ŷ        = predict(mynn,x)
rmeTrain = relative_mean_error(y,ŷ,normrec=false)
@test rmeTrain  < 0.01

using BenchmarkTools
@btime train!($mynn,$x,$y,epochs=60,verbosity=NONE,rng=copy(TESTRNG))

#  original (already int64 and no return value): 2.988 s (117156427 allocations: 3.13 GiB)
# vith vector instead of array:  1.111 s (44415127 allocations: 772.20 MiB)
# with _dedxComp!: 777.724 ms (22815127 allocations: 442.61 MiB)
# with _dedwComp!: 410.060 ms (1215127 allocations: 113.02 MiB)
# with all inbounds: 256.673 ms (1215127 allocations: 113.02 MiB)
y_id = [3,2,1,2]
x_id = [1,2,2,1]
w_id = [2,3,2,1]

x = [1.5,2.5]
w = [2.0,3.0,4.0]


function foo!(y,x,w,y_id,x_id,w_id)
    for i in 1:length(y_id)
        y[y_id[i]] += x[x_id[i]] * w[w_id[i]]
    end
    return y
end

foo(x,w,y_id,x_id,w_id)


# ---------------------------------------------------
y_id = [(3,1),(2,2),(2,2),(2,1)]
x_id = [(1,2),(2,1),(1,1),(2,2)]
w_id = [(2,2),(3,2),(2,1),(1,1)]

x = [1.5 2.5; 2.0 1.0]
w = [2.0 3.0 4.0; 1.0 1.5 2.5; 0.5 1.0 0.5]

y = zeros(3,2)

function foo!(y,x,w,y_id,x_id,w_id)
    for i in 1:length(y_id)
        y[y_id[i][1],y_id[i][2]] += x[x_id[i][1],x_id[i][2]] * w[w_id[i][1],w_id[i][2]]
    end
    return y
end


foo!(y,x,w,y_id,x_id,w_id)
@btime foo!($zeros(3,2),$x,$w,$y_id,$x_id,$w_id)



# ------------------------------------------------------------------------------

using StaticArrays, BenchmarkTools

mutable struct ConvLayerTest4{Int32}
    x_ids::Vector{SVector{3,Int32}}
    y_ids::Vector{SVector{3,Int32}}
    w_ids::Vector{SVector{4,Int32}} 
    w::Array{Float64,4}
    somethingelse
end

# Data generation for the MWE...
x = rand(64,64,3)
y = zeros(32,32,5)
w = rand(4,4,3,5)
N = 3000
x_ids = [SVector{3,Int64}([rand(1:id) for id in size(x)]...) for n in 1:N]
y_ids = [SVector{3,Int64}([rand(1:id) for id in size(y)]...) for n in 1:N]  
w_ids = [SVector{4,Int64}([rand(1:id) for id in size(w)]...) for n in 1:N]  
layer = ConvLayerTest4(x_ids, y_ids, w_ids, w, "foo")

function compute!(y,l,x)
    for i in 1:length(l.y_ids)
        y[l.y_ids[i][1],l.y_ids[i][2],l.y_ids[i][3]] += 
           x[l.x_ids[i][1],l.x_ids[i][2],l.x_ids[i][3]] * 
           l.w[l.w_ids[i][1],l.w_ids[i][2],l.w_ids[i][3],l.w_ids[i][4]]
    end
    return nothing
end

function compute!(y,x,w,y_ids,x_ids,w_ids)
    for i in 1:length(y_ids)
        y[y_ids[i][1],y_ids[i][2],y_ids[i][3]] += 
           x[x_ids[i][1],x_ids[i][2],x_ids[i][3]] * 
           w[w_ids[i][1],w_ids[i][2],w_ids[i][3],w_ids[i][4]]
    end
    return nothing
end

# The computation that I care...
@btime compute!($y,$layer,$x)
@btime compute!($y,$x,$w,$y_ids,$x_ids,$w_ids)

a = compute!(y,layer,x)
y = zeros(32,32,5)
b = compute!(y,x,w,y_ids,x_ids,w_ids)
a == b



# ------------------------------------------------------------------------------

foo(x,::Val{true}) = println("t $x")

foo(x,::Val{false}) = println("f $x")

foo(10,Val(true))