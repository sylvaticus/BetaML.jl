
export SD,train!,DebugOptAlg


# ------------------------------------------------------------------------------
# DebugOptAlg

struct SD <: OptimisationAlgorithm
    maxEpochs::Int64
    η::Function
    λ::Float64
    rShuffle::Bool
    nMsgs::Int64
    tol::Float64
    function SD(;maxEpochs=1000, η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0)
        return new(maxEpochs,η,λ,rShuffle,nMsgs,tol)
    end
end


#maxEpochs=1000, η=t -> 1/(1+t), λ=1, rShuffle=true, nMsgs=10, tol=0
function train!(nn::NN,x,y,optAlg::SD=SD())
    maxEpochs = optAlg.maxEpochs
    η = optAlg.η
    λ = optAlg.λ
    rShuffle = optAlg.rShuffle
    nMsgs = optAlg.nMsgs
    tol = optAlg.tol
    x = makeMatrix(x)
    y = makeMatrix(y)
    if nMsgs != 0
        println("***\n*** Training $(nn.name) for maximum $maxEpochs epochs. Random shuffle: $rShuffle")
    end
    #dyn_η = η == nothing ? true : false
    (ϵ,ϵl) = (0,Inf)
    converged = false
    @showprogress 1 "Training the Neural Network..." for t in 1:maxEpochs
        if rShuffle
           # random shuffle x and y
           ridx = shuffle(1:size(x)[1])
           x = x[ridx, :]
           y = y[ridx , :]
        end
        ϵ = 0
        #η = dyn_η ? 1/(1+t) : η
        ηₜ = η(t)*λ
        for i in 1:size(x)[1]
            xᵢ = x[i,:]'
            yᵢ = y[i,:]'
            W  = getParams(nn)
            dW = getGradient(nn,xᵢ,yᵢ)
            newW = gradientDescentSingleUpdate(W,dW,ηₜ)
            setParams!(nn,newW)
            ϵ += loss(nn,xᵢ,yᵢ)
        end
        if nMsgs != 0 && (t % ceil(maxEpochs/nMsgs) == 0 || t == 1 || t == maxEpochs)
          println("Avg. error after epoch $t : $(ϵ/size(x)[1])")
        end

        if abs(ϵl/size(x)[1] - ϵ/size(x)[1]) < (tol * abs(ϵl/size(x)[1]))
            if nMsgs != 0
                println((tol * abs(ϵl/size(x)[1])))
                println("*** Avg. error after epoch $t : $(ϵ/size(x)[1]) (convergence reached")
            end
            converged = true
            break
        else
            ϵl = ϵ
        end
    end
    if nMsgs != 0 && converged == false
        println("*** Avg. error after epoch $maxEpochs : $(ϵ/size(x)[1]) (convergence not reached)")
    end
    nn.trained = true
end

# ------------------------------------------------------------------------------
# DebugOptAlg

struct DebugOptAlg <: OptimisationAlgorithm
    dString::String
    function DebugOptAlg(;dString="Hello World, I am a Debugging Algorithm. I done nothing to your Net.")
        return new(dString)
    end
end

function train!(nn::NN,x,y,optAlg::DebugOptAlg)
    println(optAlg.dString)
end
