"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

export SGD,DebugOptAlg,ADAM


# ------------------------------------------------------------------------------
# SGD
"""
  SGD(;η=t -> 1/(1+t), λ=2)

Stochastic Gradient Descent algorithm (default)

# Fields:
- `η`: Learning rate, as a function of the current epoch [def: t -> 1/(1+t)]
- `λ`: Multiplicative constant to the learning rate [def: 2]
"""
struct SGD <: OptimisationAlgorithm
    η::Function
    λ::Float64
    function SGD(;η=t -> 1/(1+t), λ=2)
        return new(η,λ)
    end
end


function singleUpdate!(θ,▽,opt_alg::SGD;nEpoch,nBatch,nBatches,xbatch,ybatch)
    η    = opt_alg.η(nEpoch)*opt_alg.λ
    #newθ = gradSub.(θ,gradMul.(▽,η))
    θ =  θ - ▽ .* η
    #newθ = gradientDescentSingleUpdate(θ,▽,η)
    return (θ=θ,stop=false)
end

#gradientDescentSingleUpdate(θ::Number,▽::Number,η) = θ .- (η .* ▽)
#gradientDescentSingleUpdate(θ::AbstractArray,▽::AbstractArray,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))
#gradientDescentSingleUpdate(θ::Tuple,▽::Tuple,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))

#maxEpochs=1000, η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0




# ------------------------------------------------------------------------------
# ADAM
#

"""
  ADAM(;η, λ, β₁, β₂, ϵ)

The [ADAM](https://arxiv.org/pdf/1412.6980.pdf) algorithm, an adaptive moment estimation optimiser.

# Fields:
- `η`:  Learning rate (stepsize, α in the paper), as a function of the current epoch [def: t -> 0.001 (i.e. fixed)]
- `λ`:  Multiplicative constant to the learning rate [def: 1]
- `β₁`: Exponential decay rate for the first moment estimate [range: ∈ [0,1], def: 0.9]
- `β₂`: Exponential decay rate for the second moment estimate [range: ∈ [0,1], def: 0.999]
- `ϵ`:  Epsilon value to avoid division by zero [def: 10^-8]
"""
mutable struct ADAM <: OptimisationAlgorithm
    η::Function
    λ::Float64
    β₁::Float64
    β₂::Float64
    ϵ::Float64
    m::Vector{Learnable}
    v::Vector{Learnable}
    function ADAM(;η=t -> 0.001, λ=1.0, β₁=0.9, β₂=0.999, ϵ=1e-8)
        return new(η,λ,β₁,β₂,ϵ,[],[])
    end
end

"""
   initOptAlg!(opt_alg::ADAM;θ,batch_size,x,y,rng)

Initialize the ADAM algorithm with the parameters m and v as zeros and check parameter bounds
"""
function initOptAlg!(opt_alg::ADAM;θ,batch_size,x,y,rng = Random.GLOBAL_RNG)
    opt_alg.m = θ .- θ # setting to zeros
    opt_alg.v = θ .- θ # setting to zeros
    if opt_alg.β₁ <= 0 || opt_alg.β₁ >= 1 @error "The parameter β₁ must be ∈ [0,1]" end
    if opt_alg.β₂ <= 0 || opt_alg.β₂ >= 1 @error "The parameter β₂ must be ∈ [0,1]" end
end

function singleUpdate!(θ,▽,opt_alg::ADAM;nEpoch,nBatch,nBatches,xbatch,ybatch)
    β₁,β₂,ϵ  = opt_alg.β₁, opt_alg.β₂, opt_alg.ϵ
    η        = opt_alg.η(nEpoch)*opt_alg.λ
    t        = (nEpoch-1)*nBatches+nBatch
    opt_alg.m = @. β₁ * opt_alg.m + (1-β₁) * ▽
    opt_alg.v = @. β₂ * opt_alg.v + (1-β₂) * (▽*▽)
    #opt_alg.v = [β₂ .* opt_alg.v.data[i] .+ (1-β₂) .* (▽.data[i] .* ▽.data[i]) for i in 1:size(opt_alg.v.data)]
    m̂        = @. opt_alg.m /(1-β₁^t)
    v̂        = @. opt_alg.v /(1-β₂^t)
    θ        = @. θ - (η * m̂) /(sqrt(v̂)+ϵ)
    return     (θ=θ,stop=false)
end

# ------------------------------------------------------------------------------
# DebugOptAlg

struct DebugOptAlg <: OptimisationAlgorithm
    dString::String
    function DebugOptAlg(;dString="Hello World, I am a Debugging Algorithm. I done nothing to your Net.")
        return new(dString)
    end
end

function singleUpdate!(θ,▽,opt_alg::DebugOptAlg;nEpoch,nBatch,batch_size,ϵ_epoch,ϵ_epoch_l)
    println(opt_alg.dString)
    return (θ=θ,stop=false)
end
