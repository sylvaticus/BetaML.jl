
@eval Utils begin

export   AutoEncoder, AutoE_hp, AutoE_options

@force using ..Nn

import ..Nn: AbstractLayer, ADAM, SGD, NeuralNetworkEstimator, OptimisationAlgorithm, DenseLayer, NN
import ..Imputation


"""
$(TYPEDEF)

Hyperparameters for the AutoEncoder transformer

## Parameters
$(FIELDS)

"""
Base.@kwdef mutable struct AutoE_hp <: BetaMLHyperParametersSet
   "The desired size of the encoded data, that is the number of dimensions in output or the size of the latent space. This is the number of neurons of the layer sitting between the encoding and decoding layers. If the value is a float it is considered a percentage (to be rounded) of the dimensionality of the data [def: `0.333`]"
   encoded_size::Union{Float64,Int64}  = 0.333
   "Inner layers dimension (i.e. number of neurons). If the value is a float it is considered a percentage (to be rounded) of the dimensionality of the data [def: `nothing` that applies a specific heuristic]. Consider that the underlying neural network is trying to predict multiple values at the same time. Normally this requires many more neurons than a scalar prediction. If `e_layers` or `d_layers` are specified, this parameter is ignored for the respective part."
   layers_size::Union{Int64,Float64,Nothing} = nothing
   "The layers (vector of `AbstractLayer`s) responsible for the encoding of the data [def: `nothing`, i.e. three dense layers with the inner ones of `layers_size`]"
   e_layers::Union{Nothing,Vector{AbstractLayer}} = nothing
   "The layers (vector of `AbstractLayer`s) responsible for the decoding of the data [def: `nothing`, i.e. three dense layers with the inner ones of `layers_size`]"
   d_layers::Union{Nothing,Vector{AbstractLayer}} = nothing
   """Loss (cost) function [def: `squared_cost`]
   It must always assume y and ŷ as (n x d) matrices, possibly using `dropdims` inside.
   """
   loss::Union{Nothing,Function} = squared_cost
   "Derivative of the loss function [def: `dsquared_cost` if `loss==squared_cost`, `nothing` otherwise, i.e. use the derivative of the squared cost or autodiff]"
   dloss::Union{Function,Nothing}  = nothing
   "Number of epochs, i.e. passes through the whole training sample [def: `200`]"
   epochs::Int64 = 200
   "Size of each individual batch [def: `8`]"
   batch_size::Int64 = 8
   "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
   opt_alg::OptimisationAlgorithm = ADAM()
   "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
   shuffle::Bool = true  
   """
   The method - and its parameters - to employ for hyperparameters autotuning.
   See [`SuccessiveHalvingSearch`](@ref) for the default method.
   To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and optionally change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
   """
  tunemethod::AutoTuneMethod                  = SuccessiveHalvingSearch(hpranges = Dict("epochs"=>[100,200,400],"batch_size"=>[8,16],"encoded_size"=>[0.2,0.3,0.5],"layers_size"=>[1.3,2.0,5.0,10.0,nothing]),multithreads=true)
end

"""
$(TYPEDEF)

Options for the AutoEncoder transformer (the same as [`BML_options`](@ref), plus `device`)

## Parameters
$(FIELDS)

"""
Base.@kwdef mutable struct AutoE_options <: BetaMLOptionsSet
   "Cache the results of the fitting stage, as to allow predict(mod) [default: `true`]. Set it to `false` to save memory for large data."
   cache::Bool = true
   "An optional title and/or description for this model"
   descr::String = "" 
   "Option for hyper-parameters autotuning [def: `false`, i.e. no autotuning performed]. If activated, autotuning is performed on the first `fit!()` call. Control auto-tuning through the option `tunemethod` (see the model hyper-parameters)"
   autotune::Bool = false
   "The verbosity level to be used in training or prediction: `NONE`, `LOW`, `STD` [default], `HIGH` or `FULL`"
   verbosity::Verbosity = STD
   """
   The strategy to apply in case of failure during the fitting of the underlying neural network (loss not decreasing) [def: "tryagain"].
   See the option `onfail` of [`NeuralNetworkE_options`](@ref) for details.
   """
   onfail::String = "tryagain"
   "The epoch at which to try the onfail strategy [def: `5`]"
   fail_epoch::Int64 = 5
   "The (relative) threshold to use for the onfail strategy [def: `0.01`]"
   fail_threshold::Float64 = 0.01
   "The maximum number of attempts to use for the tryagain strategy [def: `10`]"
   fail_attempts::Int64 = 10
   "Random Number Generator (see [`?FIXEDSEED`](@ref FIXEDSEED)) [default: `Random.GLOBAL_RNG`]"
   rng::AbstractRNG = Random.GLOBAL_RNG
   """
   The device where to train the underlying neural network and compute the encodings/decodings [def: `:auto`, i.e. where the data is]. See the option `device` of [`NeuralNetworkE_options`](@ref) for details.
   """
   device::Union{Symbol,String,Nn.MLDataDevices.AbstractDevice} = :auto
end

Base.@kwdef mutable struct AutoEncoder_lp <: BetaMLLearnableParametersSet
   encoded_size_actual::Union{Int64,Nothing}                             = nothing
   fullnn::Union{NeuralNetworkEstimator,Nothing}                    = nothing
   n_el::Union{Nothing,Int64}                                       = nothing
   n_dl::Union{Nothing,Int64}                                       = nothing
end

"""
$(TYPEDEF)

Perform a (possibly non-linear) transformation ("encoding") of the data into a different space, e.g. for dimensionality reduction using a neural network trained to replicate the input data.

A neural network is trained to first transform the data (often "compress") to a subspace (the output of an inner layer) and then retransform (subsequent layers) to the original data.

`predict(mod::AutoEncoder,x)` returns the encoded data, `inverse_predict(mod::AutoEncoder,xtransformed)` performs the decoding.

For the parameters see [`AutoE_hp`](@ref) and [`AutoE_options`](@ref) 

# Notes:
- AutoEncoder doesn't automatically scale the data. It is suggested to apply the [`Scaler`](@ref) model before running it. 
- Missing data are not supported. Impute them first, see the [`Imputation`](@ref) module.
- Decoding layers can be optionally chosen (parameter `d_layers`) in order to suit the kind of data, e.g. a `relu` activation function for non-negative data

# Example:

```julia
julia> using BetaML

julia> x = [0.12 0.31 0.29 3.21 0.21;
            0.22 0.61 0.58 6.43 0.42;
            0.51 1.47 1.46 16.12 0.99;
            0.35 0.93 0.91 10.04 0.71;
            0.44 1.21 1.18 13.54 0.85];

julia> m    = AutoEncoder(encoded_size=1,epochs=400)
A AutoEncoder BetaMLModel (unfitted)

julia> x_reduced = fit!(m,x)
***
*** Training  for 400 epochs with algorithm ADAM.
Training..       avg loss on epoch 1 (1):        60.27802763757111
Training..       avg loss on epoch 200 (200):    0.08970099870421573
Training..       avg loss on epoch 400 (400):    0.013138484118673664
Training of 400 epochs completed. Final epoch error: 0.013138484118673664.
5×1 Matrix{Float64}:
  -3.5483740608901186
  -6.90396890458868
 -17.06296512222304
 -10.688936344498398
 -14.35734756603212

julia> x̂ = inverse_predict(m,x_reduced)
5×5 Matrix{Float64}:
 0.0982406  0.110294  0.264047   3.35501  0.327228
 0.205628   0.470884  0.558655   6.51042  0.487416
 0.529785   1.56431   1.45762   16.067    0.971123
 0.3264     0.878264  0.893584  10.0709   0.667632
 0.443453   1.2731    1.2182    13.5218   0.842298

julia> info(m)["rme"]
0.020858783340281222

julia> hcat(x,x̂)
5×10 Matrix{Float64}:
 0.12  0.31  0.29   3.21  0.21  0.0982406  0.110294  0.264047   3.35501  0.327228
 0.22  0.61  0.58   6.43  0.42  0.205628   0.470884  0.558655   6.51042  0.487416
 0.51  1.47  1.46  16.12  0.99  0.529785   1.56431   1.45762   16.067    0.971123
 0.35  0.93  0.91  10.04  0.71  0.3264     0.878264  0.893584  10.0709   0.667632
 0.44  1.21  1.18  13.54  0.85  0.443453   1.2731    1.2182    13.5218   0.842298
```
"""
mutable struct AutoEncoder <: BetaMLUnsupervisedModel
    hpar::AutoE_hp
    opt::AutoE_options
    par::Union{Nothing,AutoEncoder_lp}
    cres::Union{Nothing,AbstractMatrix}
    fitted::Bool
    info::Dict{String,Any}
end

function AutoEncoder(;kwargs...)
    m = AutoEncoder(AutoE_hp(),AutoE_options(),AutoEncoder_lp(),nothing,false,Dict{Symbol,Any}())
    thisobjfields  = fieldnames(nonmissingtype(typeof(m)))
    for (kw,kwv) in kwargs
       found = false
       for f in thisobjfields
          fobj = getproperty(m,f)
          if kw in fieldnames(typeof(fobj))
              setproperty!(fobj,kw,kwv)
              found = true
          end
        end
        # Correction for releasing without breaking.. to be removed from v0.12 onward...
        # found || error("Keyword \"$kw\" is not part of this model.")
        if !found
            if kw == :outdims
                setproperty!(m.hpar,:encoded_size,kwv)
                found = true
            elseif kw == :innerdims
                setproperty!(m.hpar,:layers_size,kwv) 
                found = true
            else
                error("Keyword \"$kw\" is not part of this model.")
            end
        end
    end
    return m
end

function fit!(m::AutoEncoder,X)
    (m.fitted) || autotune!(m,(X,))

    # Parameter alias..
    e_layers    = m.hpar.e_layers
    d_layers    = m.hpar.d_layers
    encoded_size     = m.hpar.encoded_size
    layers_size   = m.hpar.layers_size
    loss        = m.hpar.loss
    dloss       = m.hpar.dloss
    epochs      = m.hpar.epochs
    batch_size  = m.hpar.batch_size
    opt_alg     = m.hpar.opt_alg
    shuffle     = m.hpar.shuffle
    cache       = m.opt.cache
    descr       = m.opt.descr
    verbosity   = m.opt.verbosity
    #cb          = m.opt.cb
    rng         = m.opt.rng
    device      = m.opt.device
    fitted      = m.fitted

    (N,D) = size(X)
    if fitted
        size(m.par.fullnn.par.nnstruct.layers[1])[1][1] == D || @error "The data used to re-fit the model have different dimensionality than the original data. [`reset!`](@ref) the model first."
        verbosity >= HIGH && @info "Re-fitting of the model on new data"
        encoded_size_actual  = m.par.encoded_size_actual
        fullnn          = m.par.fullnn
        n_el            = m.par.n_el
        n_dl            = m.par.n_dl 
    else
        typeof(encoded_size) <: Integer ?  encoded_size_actual = encoded_size : encoded_size_actual = max(1,Int(round(D * encoded_size))) 
        if isnothing(layers_size) 
            if D == 1
                innerSize = 3
            elseif D < 5
                innerSize = max(1,Int(round(D*D)))   
            elseif D < 10   
                innerSize = max(1,Int(round(D*1.3*D/3)))
            else
                innerSize = max(1,Int(round(D*1.3*log(2,D)))) 
            end
        elseif typeof(layers_size) <: Integer
            innerSize = layers_size
        else
            innerSize = max(1,Int(round(D*layers_size)) )
        end

        if isnothing(e_layers)
            l1 = DenseLayer(D,innerSize, f=relu, df=drelu, rng=rng)
            l2 = DenseLayer(innerSize,innerSize, f=relu, df=drelu, rng=rng)
            l3 = DenseLayer(innerSize, encoded_size_actual, f=identity, df=didentity, rng=rng)
            e_layers_actual = [l1,l2,l3]
        else
            e_layers_actual = copy(e_layers)
        end
        if isnothing(d_layers)
            l1d = DenseLayer(encoded_size_actual,innerSize, f=relu, df=drelu, rng=rng)
            l2d = DenseLayer(innerSize,innerSize, f=relu, df=drelu, rng=rng)
            l3d = DenseLayer(innerSize, D, f=identity, df=didentity, rng=rng)
            d_layers_actual = [l1d,l2d,l3d]
        else
            d_layers_actual = copy(d_layers)
        end
        fullnn = NeuralNetworkEstimator(layers=[e_layers_actual...,d_layers_actual...],loss=loss,dloss=dloss,epochs=epochs,batch_size=batch_size,opt_alg=opt_alg,shuffle=shuffle,cache=cache,descr=descr,verbosity=verbosity,rng=rng,
                                        onfail=m.opt.onfail,fail_epoch=m.opt.fail_epoch,fail_threshold=m.opt.fail_threshold,fail_attempts=m.opt.fail_attempts)
        n_el = length(e_layers_actual)
        n_dl = length(d_layers_actual)
    end

    fullnn.opt.device = device # also when re-fitting, in case the option has been changed
    x̂ =  fit!(fullnn,X,X)

    par                 = AutoEncoder_lp()
    par.encoded_size_actual  = encoded_size_actual
    par.fullnn          = fullnn
    par.n_el            = n_el
    par.n_dl            = n_dl
    m.par               = par

    m.fitted=true
    
    rme = cache ? relative_mean_error(X,x̂) : missing

    m.info["nepochs_ran"]     = info(fullnn)["nepochs_ran"]
    m.info["loss_per_epoch"]  = info(fullnn)["loss_per_epoch"]
    m.info["final_loss"]      = verbosity >= STD ? info(fullnn)["loss_per_epoch"][end] : missing
    m.info["rme"]             = rme
    m.info["par_per_epoch"]   = info(fullnn)["par_per_epoch"]
    m.info["xndims"]          = info(fullnn)["xndims"]
    m.info["fitted_records"]  = info(fullnn)["fitted_records"]
    m.info["nepochs_ran"]     = info(fullnn)["nepochs_ran"]
    m.info["nLayers"]         = info(fullnn)["nLayers"]
    m.info["nELayers"]   = m.par.n_el
    m.info["nDLayers"]   = m.par.n_dl
    m.info["nPar"]       = info(fullnn)["nPar"]

    if cache
        m.cres = _ae_subnet(fullnn,1:m.par.n_el,X) |> makematrix
    end
    m.fitted=true
    verbosity >= HIGH && println("Relative mean error of the encoded vs original data: $rme")
    return cache ? m.cres : nothing
end   

# Pass the data through a subset of the layers of the full (encoding + decoding) network
function _ae_subnet(fullnn,layer_ids,X)
    nn = fullnn.par.nnstruct
    return predict(NN(nn.layers[layer_ids],nn.cf,nn.dcf,nn.trained,nn.name),X)
end

# Put the network where to compute the predictions, as `predict(::NeuralNetworkEstimator,X)` does
function _ae_place!(m::AutoEncoder,X)
    fullnn = m.par.fullnn
    fullnn.opt.device = m.opt.device
    if Nn._device_spec(m.opt.device) === :auto
        Nn._restore_saved_device!(fullnn)
    else
        delete!(fullnn.info,"device_at_save")
        Nn._place_model!(fullnn.par.nnstruct,m.opt.device,X;verbosity=m.opt.verbosity)
    end
end

function predict(m::AutoEncoder,X)
    _ae_place!(m,X)
    return _ae_subnet(m.par.fullnn,1:m.par.n_el,X) |> makematrix
end  
function inverse_predict(m::AutoEncoder,X)
    _ae_place!(m,X)
    return _ae_subnet(m.par.fullnn,m.par.n_el+1:length(m.par.fullnn.par.nnstruct.layers),X) |> makematrix
end

function Api._model_to_save(m::AutoEncoder)
    (isnothing(m.par) || isnothing(m.par.fullnn)) && return m
    par = AutoEncoder_lp(m.par.encoded_size_actual,Api._model_to_save(m.par.fullnn),m.par.n_el,m.par.n_dl)
    return AutoEncoder(m.hpar,m.opt,par,isnothing(m.cres) ? nothing : Nn._tocpu(m.cres),m.fitted,m.info)
end
end



