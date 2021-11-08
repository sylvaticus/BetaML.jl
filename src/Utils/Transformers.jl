# Part of submodule Utils of BetaML
# Function of a single argument (including scalars and vectors), like activation functions but also gini, entropy,...)


# ------------------------------------------------------------------------------
# Various neural network activation functions as well their derivatives

#identity(x)          = x already in Julia base
didentity(x)          = one(x)
""" relu(x) \n\n Rectified Linear Unit \n\n https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf"""
relu(x)               = max(zero(x), x)
""" drelu(x) \n\n Rectified Linear Unit \n\n https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf"""
drelu(x)              = x <= zero(x) ? zero(x) : one(x)
"""elu(x; α=1) with α > 0 \n\n https://arxiv.org/pdf/1511.07289.pdf"""
elu(x; α=one(x))      = x > zero(x) ? x : α *(exp(x) - one(x))
"""delu(x; α=1) with α > 0 \n\n https://arxiv.org/pdf/1511.07289.pdf"""
delu(x; α=one(x))      = x > zero(x) ? one(x) : elu(x, α=α) + α
"""celu(x; α=1) \n\n https://arxiv.org/pdf/1704.07483.pdf"""
celu(x; α=one(x))     = max(zero(x),x)+ min(zero(x), α *(exp(x / α) - one(x) ))
#celu(x; α=one(x))    = if x >= zero(x) x/α else exp(x/α)-one(x) end
"""dcelu(x; α=1) \n\n https://arxiv.org/pdf/1704.07483.pdf"""
dcelu(x; α=one(x))    = x >= zero(x) ? one(x) : exp(x/α)
"""plu(x;α=0.1,c=1) \n\n Piecewise Linear Unit \n\n https://arxiv.org/pdf/1809.09534.pdf"""
plu(x;α=0.1,c=one(x)) = max(α*(x+c)-c,min(α*(x-c)+c,x)) # convert(eltype(x), α)
"""dplu(x;α=0.1,c=1) \n\n Piecewise Linear Unit derivative \n\n https://arxiv.org/pdf/1809.09534.pdf"""
dplu(x;α=0.1,c=one(x)) = ( ( x >= (α*(x+c)-c)  &&  x <= (α*(x+c)+c) ) ? one(x) : α ) # convert(eltype(x), α)


"""
    pool1d(x,poolSize=2;f=mean)

Apply funtion `f` to a rolling poolSize contiguous (in 1d) neurons.

Applicable to `VectorFunctionLayer`, e.g. `layer2  = VectorFunctionLayer(nₗ,f=(x->pool1d(x,4,f=mean))`
**Attention**: to apply this funciton as activation function in a neural network you will need Julia version >= 1.6, otherwise you may experience a segmentation fault (see [this bug report](https://github.com/FluxML/Zygote.jl/issues/943))
"""
pool1d(x,poolSize=3;f=mean) = [f(x[i:i+poolSize-1]) for i in 1:length(x)-poolSize+1] # we may try to use CartesianIndices/LinearIndices for a n-dimensional generalisation


#tanh(x) already in Julia base
"""dtanh(x)"""
dtanh(x)              = sech(x)^2  # = 1-tanh(x)^2
"""sigmoid(x)"""
sigmoid(x)            = one(x)/(one(x)+exp(-x))
"""dsigmoid(x)"""
dsigmoid(x)           = exp(-x)*sigmoid(x)^2
"""softmax (x; β=1) \n\n The input x is a vector. Return a PMF"""
softmax(x; β=one.(x)) = exp.((β .* x) .- lse(β .* x))  # efficient implementation of softmax(x)  = exp.(x) ./  sum(exp.(x))
softmax(x, β) = softmax(x, β=β)
""" dsoftmax(x; β=1) \n\n Derivative of the softmax function \n\n https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/"""
function dsoftmax(x; β=one(x[1]))
    x = makeColVector(x)
    d = length(x)
    out = zeros(d,d)
    y = softmax(x,β=β)
    for i in 1:d
        smi = y[i]
        for j in 1:d
            if j == i
                out[i,j] = β*(smi-smi^2)
            else
                out[i,j] = - β*y[j]*smi
            end
        end
    end
    return out
end

"""softplus(x) \n\n https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus"""
softplus(x)           = log(one(x) + exp(x))
"""dsoftplus(x) \n\n https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus"""
dsoftplus(x)          = 1/(1+exp(-x))
""" mish(x) \n\n https://arxiv.org/pdf/1908.08681v1.pdf"""
mish(x)               = x*tanh(softplus(x))
""" dmish(x) \n\n https://arxiv.org/pdf/1908.08681v1.pdf"""
dmish(x) = x*(1 - tanh(log(exp(x) + 1))^2)*exp(x)/(exp(x) + 1) + tanh(log(exp(x) + 1))


"""
   autoJacobian(f,x;nY)

Evaluate the Jacobian using AD in the form of a (nY,nX) matrix of first derivatives

# Parameters:
- `f`: The function to compute the Jacobian
- `x`: The input to the function where the jacobian has to be computed
- `nY`: The number of outputs of the function `f` [def: `length(f(x))`]

# Return values:
- An `Array{Float64,2}` of the locally evaluated Jacobian

# Notes:
- The `nY` parameter is optional. If provided it avoids having to compute `f(x)`
"""
function autoJacobian(f,x;nY=length(f(x)))
    x = convert(Array{Float64,1},x)
    #j = Array{Float64, 2}(undef, size(x,1), nY)
    #for i in 1:nY
    #    j[:, i] .= gradient(x -> f(x)[i], x)[1]
    #end
    #return j'
    j = Array{Float64, 2}(undef, nY, size(x,1))
    for i in 1:nY
        j[i,:] = gradient(x -> f(x)[i], x)[1]'
    end
    return j
end


# ------------------------------------------------------------------------------
# Partition tasks..

"""
   gini(x)

Calculate the Gini Impurity for a list of items (or rows).

See: https://en.wikipedia.org/wiki/Decision_tree_learning#Information_gain
"""
function gini(x)

    counts = classCounts(x)
    N = size(x,1)
    impurity = 1.0
    for c in counts
        probₖ     = c / N
        impurity -= probₖ^2
    end
    return impurity
  #=
  counts = classCountsWithLabels(x)
  N = size(x,1)
  impurity = 1.0
  for k in keys(counts)
    probₖ = counts[k] / N
    impurity -= probₖ^2
  end
  return impurity
  =#
end

"""
   entropy(x)

Calculate the entropy for a list of items (or rows).

See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
"""
function entropy(x)
    counts = classCounts(x)
    N = size(x,1)
    entr = 0.0
    for c in counts
        probₖ = c / N
        entr -= probₖ * log2(probₖ)
    end
    return entr
end

"""variance(x) - population variance"""
variance(x) = var(x,corrected=false)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

"""bic(lL,k,n) -  Bayesian information criterion (lower is better)"""
bic(lL,k,n) = k*log(n)-2*lL
"""aic(lL,k) -  Akaike information criterion (lower is better)"""
aic(lL,k)   = 2*k-2*lL

# ------------------------------------------------------------------------------
# Various kernel functions (e.g. for Perceptron)
"""Radial Kernel (aka _RBF kernel_) parametrised with γ=1/2. For other gammas γᵢ use
`K = (x,y) -> radialKernel(x,y,γ=γᵢ)` as kernel function in the supporting algorithms"""
radialKernel(x,y;γ=1/2) = exp(-γ*norm(x-y)^2)
"""Polynomial kernel parametrised with `c=0` and `d=2` (i.e. a quadratic kernel).
For other `cᵢ` and `dᵢ` use `K = (x,y) -> polynomialKernel(x,y,c=cᵢ,d=dᵢ)` as
kernel function in the supporting algorithms"""
polynomialKernel(x,y;c=0,d=2) = (dot(x,y)+c)^d
