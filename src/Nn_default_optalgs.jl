
export SGD,DebugOptAlg,singleUpdate


# ------------------------------------------------------------------------------
# DebugOptAlg
"""
  SGD

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
