"""
  nn.jl

Neural Network algorithms

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/lmlj.jl/blob/master/src/nn.jl) - [Julia Package](https://github.com/sylvaticus/lmlj.jl)
- [Demonstrative static notebook](https://github.com/sylvaticus/lmlj.jl/blob/master/notebooks/nn.ipynb)
- [Demonstrative live notebook](https://mybinder.org/v2/gh/sylvaticus/lmlj.jl/master?filepath=notebooks%2Fnn.ipynb) (temporary personal online computational environment on myBinder) - it can takes minutes to start with!
- Theory based on [MITx 6.86x - Machine Learning with Python: from Linear Models to Deep Learning](https://github.com/sylvaticus/MITx_6.86x) ([Unit 3](https://github.com/sylvaticus/MITx_6.86x/blob/master/Unit%2003%20-%20Neural%20networks/Unit%2003%20-%20Neural%20networks.md))
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)
"""

# ==================================
# Neural Network Class
# ==================================

using Random

## Some utility functions..
import Base.reshape
"""
reshape(myNumber, dims..) - Reshape a number as a n dimensional Array
"""
function reshape(x::T, dims...) where {T <: Number}
   x = [x]
   reshape(x,dims)
end
function makeColVector(x::T) where {T <: Number}
    return [x]
end
function makeColVector(x::T) where {T <: AbstractArray}
    reshape(x,length(x))
end
function makeRowVector(x::T) where {T <: Number}
    return [x]'
end
function makeRowVector(x::T) where {T <: AbstractArray}
    reshape(x,1,length(x))
end


"""
   Layer

Representation of a layer in the network

# Fields:
* `w`:  Weigths matrix with respect to the input from previous layer or data (n pr. layer x n)
* `wb`: Biases (n)
* `f`:  Activation function
* `df`: Derivative of the activation function
"""
mutable struct Layer
     w::Array{Float64,2}
     wb::Array{Float64,1}
     f::Function
     df::Function
end

"""
   FNN

Representation of a Forward Neural Network

# Fields:
* `layers`:  Array of layers objects
* `cf`:      Cost function
* `dcf`:     Derivative of the cost function
* `trained`: Control flag for trained networks
"""
mutable struct FNN
    layers::Array{Layer,1}
    cf::Function
    dcf::Function
    trained::Bool
end

"""
   buildLayer(f,df,n,nₗ;w,wb)

Instantiate a new layer

Parameters:
* `f`:  Activation function
* `df`: Derivative of the activation function
* `n`:  Number of nodes
* `nₗ`: Number of nodes of the previous layer
* `w`:  Initial weigths with respect to input [default: `rand(nₗ,n)`]
* `wb`: Initial weigths with respect to bias [default: `rand(n)`]

"""
function buildLayer(f,df,n,nₗ;w=rand(nₗ,n),wb=rand(n))
    # To be sure w is a matrix and wb a column vector..
    w  = reshape(w,nₗ,n)
    wb = reshape(wb,n)
    return Layer(w,wb,f,df)
end

"""
   buildNetwork

Instantiate a new Feedforward Neural Network

Parameters:
* `layers`:  Array of layers objects
* `cf`:      Cost function
* `dcf`:     Derivative of the cost function

# Notes:
* Even if the network ends with a single output note, the cost function and its
derivative should always expect y and ŷ as column vectors.
"""
function buildNetwork(layers,cf,dcf)
    return FNN(layers,cf,dcf,false)
end

"""
   predict(layer,x)

Layer prediction of a single data point

# Parameters:
* `layer`:  Worker layer
* `x`:      Input to the layer
"""
function predict(layer::Layer,x)
  return layer.f.((reshape(x,1,length(x))*layer.w)' + layer.wb)
end

"""
   predict(fnn,x)

Network prediction of a single data point

# Parameters:
* `fnn`:  Worker network
* `x`:    Input to the network
"""
function predict(fnn::FNN,x)
    makeColVector(x)
    values = x
    for l in fnn.layers
        values = predict(l,values)
    end
    return values
end

"""
   error(fnn,x,y)

Compute network loss on a single data point

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network
* `y`:   Label input
"""
function error(fnn::FNN,x,y)
    x = makeColVector(x)
    y = makeColVector(y)
    ŷ = predict(fnn,x)
    return fnn.cf(ŷ,y)
end

"""
   errors(fnn,x,y)

Compute avg. network loss on a test set

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network (n x d)
* `y`:   Label input (n) or (n x d)
"""
function errors(fnn::FNN,x,y)
    fnn.trained ? "" : @warn "Seems you are trying to test a neural network that has not been tested. Use first `test!(rnn,x,y)`"
    ϵ = 0
    for i in 1:size(x)[1]
        xᵢ = x[i,:]'
        yᵢ = y[i,:]'
        ϵ += error(fnn,xᵢ,yᵢ)
    end
    return ϵ/size(x)[1]
end

"""
   getW(fnn)

Retrieve current weigthts

# Parameters:
* `fnn`: Worker network

# Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getW(fnn)
  w = Tuple{Array{Float64,2},Array{Float64,1}}[]
  for l in fnn.layers
      push!(w,(l.w,l.wb))
  end
  return w
end

"""
   getDW(fnn,x,y)

Retrieve the current gradient of the weigthts (i.e. derivative of the cost with respect to the weigths)

# Parameters:
* `fnn`: Worker network
* `x`:   Input to the network
* `y`:   Label input

#Notes:
* The output is a vector of tuples of each layer's input weigths and bias weigths
"""
function getDW(fnn,x,y)
  x = makeColVector(x)
  y = makeColVector(y)
  lz = Array{Float64,1}[]
  lo = Array{Float64,1}[]
  dW = Tuple{Array{Float64,2},Array{Float64,1}}[]

  push!(lz,x)
  push!(lo,x)

  for l in fnn.layers
      x = lo[end]
      z = dropdims((reshape(x,1,length(x))*l.w)' + l.wb,dims=2)
      o = l.f.(z)
      push!(lz, z)
      push!(lo, o)
  end
  dc = fnn.dcf(lo[end],y)
  δ = dc # derivative of the cost function with respect to the layer output

  # backpropagation step
  for lidx in length(fnn.layers):-1:1
     l = fnn.layers[lidx]
     # Note that lz and lo vectors includes x, so the second layer is the third element in the vector
     dwb = l.df.(lz[lidx+1]) .* δ # derivative with respect to the layer biases
     dw = lo[lidx] * dwb'         # derivative with respect to the layer input weigths
     push!(dW,(dw,dwb))
     # Computing derivatives of the cost function with respect of the output of the previous layer
     δ = l.w * dwb
  end
  return dW[end:-1:1] # reversing it, to start from the first layer
end

"""
   updateWeights!(fnn,w)

Update weigths of the network

# Parameters:
* `fnn`: Worker network
* `w`:   The new weights to set
"""
function updateWeights!(fnn,w)
    for lidx in 1:length(fnn.layers)
        fnn.layers[lidx].w = w[lidx][1]
        fnn.layers[lidx].wb = w[lidx][2]
    end
end

"""
   train!(fnn,x,y;epochs,η,rshuffle)

Train a fnn with the given x,y data

# Parameters:
* `fnn`:      Worker network
* `x`:        Training input to the network (records x dimensions)
* `y`:        Label input (records)
* `epochs`:   Number of passages over the training set [def = `1000`]
* `η`:        Learning rate. If not provided 1/(1+epoch) is used [def = `nothing`]
* `rshuffle`: Whether to random shuffle the training set at each epoch [def = `true`]
"""
function train!(fnn,x,y;epochs=1000, η=nothing, rshuffle=true)
    logStep = Int64(ceil(epochs/100))
    dyn_η = η == nothing ? true : false
    for t in 1:epochs
        if rshuffle
           # random shuffle x and y
           ridx = shuffle(1:size(x)[1])
           x = x[ridx, :]
           y = y[ridx , :]
        end
        ϵ = 0
        η = dyn_η ? 1/(1+t) : η
        for i in 1:size(x)[1]
            xᵢ = x[i,:]'
            yᵢ = makeColVector(y[i])
            w  = getW(fnn)
            dW = getDW(fnn,xᵢ,yᵢ)
            for (lidx,l) in enumerate(fnn.layers)
                l.w  = l.w -  η .* dW[lidx][1]
                l.wb = l.wb - η .* dW[lidx][2]
            end
            ϵ += error(fnn,xᵢ,yᵢ)
        end
        (t % logStep == 0) || t == 1 || t == epochs ? println("Avg. error after epoch $t : $(ϵ/size(x)[1])") : ""
    end
    fnn.trained = true
end


# ==================================
# Specific implementation - FNN definition
# ==================================

# Defining the functions we fill use as activation function as well their derivatives
# (yes, we could have used instead an automatic differentiation - AD - library..)
relu(x)     = max(0,x)
drelu(x)    = x <= 0 ? 0 : 1
linearf(x)  = x
dlinearf(x) = 1
cost(ŷ,y)   = (1/2)*(y[1]-ŷ[1])^2
dcost(ŷ,y)  = [- (y[1]-ŷ[1])]

l1 = buildLayer(relu,drelu,3,2,w=[1 1 1;1 1 1],wb=[0,0,0])
l2 = buildLayer(linearf,dlinearf,1,3,w=[1,1,1],wb=0)
myfnn = buildNetwork([l1,l2],cost,dcost)

# ==================================
# Usage of the FNN
# ==================================

xtrain = [2 1; 3 3; 4 5; 6 6]
ytrain = [10,21,32,42]
ytrain = [14,21,28,42]
xtest  = [1 1; 2 2; 3 3; 5 5; 10 10]
ytest  = [7,14,21,35,70]

train!(myfnn,xtrain,ytrain,epochs=10,η=0.001,rshuffle=false) # 1.86
errors(myfnn,xtest,ytest) # 0.108

dtanh(x)    = 1-tanh(x)^2
l1 = buildLayer(tanh,dtanh,3,2)
l2 = buildLayer(linearf,dlinearf,1,3)
myfnn2 = buildNetwork([l1,l2],cost,dcost)

train!(myfnn2,xtrain,ytrain,epochs=10000,η=0.001,rshuffle=false) # 0.011
errors(myfnn2,xtest,ytest) # 76.9
```
