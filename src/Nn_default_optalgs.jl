
export SGD,DebugOptAlg,singleUpdate


# ------------------------------------------------------------------------------
# DebugOptAlg
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


function singleUpdate(θ,▽,optAlg::SGD;nEpoch,nBatch,batchSize,xbatch,ybatch)
    η    = optAlg.η(nEpoch)*optAlg.λ
    newθ = gradSub.(θ,gradMul.(▽,η))
    #newθ = θ - ▽ * η
    #newθ = gradientDescentSingleUpdate(θ,▽,η)
    return (θ=newθ,stop=false)
end

#gradientDescentSingleUpdate(θ::Number,▽::Number,η) = θ .- (η .* ▽)
#gradientDescentSingleUpdate(θ::AbstractArray,▽::AbstractArray,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))
#gradientDescentSingleUpdate(θ::Tuple,▽::Tuple,η) = gradientDescentSingleUpdate.(θ,▽,Ref(η))

#maxEpochs=1000, η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0




# ------------------------------------------------------------------------------
# ADAM
#

"""
  ADAM(;η=0.001, β₁=0.9, β₂=0.999)

ADAM[https://arxiv.org/pdf/1412.6980.pdf] algorithm, an an adaptive moment estimation optimiser.

# Fields:
- `η`:  Learning rate (stepsize, α in the paper) [def: 0.001]
- `β₁`: Exponential decay rate for the first moment estimate [def: 0.9]
- `β₂`: Exponential decay rate for the second moment estimate [def: 0.999]
"""
struct ADAM <: OptimisationAlgorithm
    η::Float64
    β₁::Float64
    β₂::Float64
    function ADAM(;η=0.001, β₁=0.9, β₂=0.999)
        return new(η,β₁,β₂)
    end
end

function singleUpdate(θ::Array{Tuple{Vararg{Array{Float64,N} where N,N} where N},1},▽::Array{Tuple{Vararg{Array{Float64,N} where N,N} where N},1},optAlg::ADAM;nEpoch,nBatch,batchSize,xbatch,ybatch)
    η ,β₁,β₂   = optAlg.η, optAlg.β₁, optAlg.β₂

    newθ = gradSub.(θ,gradMul.(▽,η))

    return (θ=newθ,stop=false)
end

function apply!(o::ADAM, x, Δ)
  η, β = o.eta, o.beta
  mt, vt, βp = get!(o.state, x, (zero(x), zero(x), β))
  @. mt = β[1] * mt + (1 - β[1]) * Δ
  @. vt = β[2] * vt + (1 - β[2]) * Δ^2
  @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
  o.state[x] = (mt, vt, βp .* β)
  return Δ
end


# ------------------------------------------------------------------------------
# DebugOptAlg

struct DebugOptAlg <: OptimisationAlgorithm
    dString::String
    function DebugOptAlg(;dString="Hello World, I am a Debugging Algorithm. I done nothing to your Net.")
        return new(dString)
    end
end

function singleUpdate(θ,▽,optAlg::DebugOptAlg;nEpoch,nBatch,batchSize,ϵ_epoch,ϵ_epoch_l)
    println(optAlg.dString)
    return (θ=θ,stop=false)
end
