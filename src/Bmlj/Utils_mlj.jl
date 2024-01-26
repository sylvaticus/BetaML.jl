"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for the Utils models of BetaML


export AutoEncoder







# ------------------------------------------------------------------------------
# Start AutoEncoder


# Model Structure declarations..
"""
$(TYPEDEF)

A ready-to use AutoEncoder, from the Beta Machine Learning Toolkit (BetaML) for ecoding and decoding of data using neural networks

# Parameters:
$(FIELDS)

# Notes:
- data must be numerical
- use `transform` to obtain the encoded data, and `inverse_trasnform` to decode to the original data

# Example:
```julia
julia> using MLJ

julia> X, y        = @load_iris;

julia> modelType   = @load AutoEncoder pkg = "BetaML" verbosity=0;

julia> model       = modelType(encoded_size=2,layers_size=10);

julia> mach        = machine(model, X)
untrained Machine; caches model-specific representations of data
  model: AutoEncoder(e_layers = nothing, …)
  args: 
    1:	Source @334 ⏎ Table{AbstractVector{Continuous}}

julia> fit!(mach,verbosity=2)
[ Info: Training machine(AutoEncoder(e_layers = nothing, …), …).
***
*** Training  for 200 epochs with algorithm BetaML.Nn.ADAM.
Training.. 	 avg loss on epoch 1 (1): 	 35.48243542158747
Training.. 	 avg loss on epoch 20 (20): 	 0.07528042222678126
Training.. 	 avg loss on epoch 40 (40): 	 0.06293071729378613
Training.. 	 avg loss on epoch 60 (60): 	 0.057035588828991145
Training.. 	 avg loss on epoch 80 (80): 	 0.056313167754822875
Training.. 	 avg loss on epoch 100 (100): 	 0.055521461091809436
Training the Neural Network...  52%|██████████████████████████████████████                                   |  ETA: 0:00:01Training.. 	 avg loss on epoch 120 (120): 	 0.06015206472927942
Training.. 	 avg loss on epoch 140 (140): 	 0.05536835903285201
Training.. 	 avg loss on epoch 160 (160): 	 0.05877560142428245
Training.. 	 avg loss on epoch 180 (180): 	 0.05476302769966953
Training.. 	 avg loss on epoch 200 (200): 	 0.049240864053557445
Training the Neural Network... 100%|█████████████████████████████████████████████████████████████████████████| Time: 0:00:01
Training of 200 epoch completed. Final epoch error: 0.049240864053557445.
trained Machine; caches model-specific representations of data
  model: AutoEncoder(e_layers = nothing, …)
  args: 
    1:	Source @334 ⏎ Table{AbstractVector{Continuous}}


julia> X_latent    = transform(mach, X)
150×2 Matrix{Float64}:
 7.01701   -2.77285
 6.50615   -2.9279
 6.5233    -2.60754
 ⋮        
 6.70196  -10.6059
 6.46369  -11.1117
 6.20212  -10.1323

julia> X_recovered = inverse_transform(mach,X_latent)
150×4 Matrix{Float64}:
 5.04973  3.55838  1.43251  0.242215
 4.73689  3.19985  1.44085  0.295257
 4.65128  3.25308  1.30187  0.244354
 ⋮                          
 6.50077  2.93602  5.3303   1.87647
 6.38639  2.83864  5.54395  2.04117
 6.01595  2.67659  5.03669  1.83234

julia> BetaML.relative_mean_error(MLJ.matrix(X),X_recovered)
0.03387721261716176


```
"""
Base.@kwdef mutable struct AutoEncoder <: MMI.Unsupervised
    "The layers (vector of `AbstractLayer`s) responsable of the encoding of the data [def: `nothing`, i.e. two dense layers with the inner one of `layers_size`]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    e_layers::Union{Nothing,Vector{AbstractLayer}} = nothing
    "The layers (vector of `AbstractLayer`s) responsable of the decoding of the data [def: `nothing`, i.e. two dense layers with the inner one of `layers_size`]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    d_layers::Union{Nothing,Vector{AbstractLayer}} = nothing
    "The number of neurons (i.e. dimensions) of the encoded data. If the value is a float it is consiered a percentual (to be rounded) of the dimensionality of the data [def: `0.33`]"
    encoded_size::Union{Float64,Int64}  = 0.333
    "Inner layer dimension (i.e. number of neurons). If the value is a float it is considered a percentual (to be rounded) of the dimensionality of the data [def: `nothing` that applies a specific heuristic]. Consider that the underlying neural network is trying to predict multiple values at the same times. Normally this requires many more neurons than a scalar prediction. If `e_layers` or `d_layers` are specified, this parameter is ignored for the respective part."
    layers_size::Union{Int64,Float64,Nothing} = nothing 
    """Loss (cost) function [def: `BetaML.squared_cost`]. Should always assume y and ŷ as (n x d) matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = BetaML.Utils.squared_cost
    "Derivative of the loss function [def: `BetaML.dsquared_cost` if `loss==squared_cost`, `nothing` otherwise, i.e. use the derivative of the squared cost or autodiff]"
    dloss::Union{Function,Nothing}  = nothing
    "Number of epochs, i.e. passages trough the whole training sample [def: `200`]"
    epochs::Int64 = 200
    "Size of each individual batch [def: `8`]"
    batch_size::Int64 = 8
    "The optimisation algorithm to update the gradient at each batch [def: `BetaML.ADAM()`] See `subtypes(BetaML.OptimisationAlgorithm)` for supported optimizers"
    opt_alg::OptimisationAlgorithm = BetaML.Nn.ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    """
    The method - and its parameters - to employ for hyperparameters autotuning.
    See [`SuccessiveHalvingSearch`](@ref) for the default method.
    To implement automatic hyperparameter tuning during the (first) `fit!` call simply set `autotune=true` and eventually change the default `tunemethod` options (including the parameter ranges, the resources to employ and the loss function to adopt).
    """
   tunemethod::AutoTuneMethod                  = BetaML.Utils.SuccessiveHalvingSearch(hpranges = Dict("epochs"=>[100,150,200],"batch_size"=>[8,16,32],"encoded_size"=>[0.2,0.3,0.5],"layers_size"=>[1.3,2.0,5.0]),multithreads=false)
    "An optional title and/or description for this model"
    descr::String = "" 
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end

"""
$(TYPEDSIGNATURES)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::AutoEncoder, verbosity, X)
    x = MMI.matrix(X)                     # convert table to matrix   
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = mljverbosity_to_betaml_verbosity(verbosity)
   
    mi = BetaML.Utils.AutoEncoder(;e_layers=m.e_layers,d_layers=m.d_layers,encoded_size=m.encoded_size,layers_size=m.layers_size,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, tunemethod=m.tunemethod, cache=false, descr=m.descr, rng=m.rng, verbosity=verbosity)
    Api.fit!(mi,x)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 #MMI.predict(m::AutoEncoder, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 # MMI.transform(m::AutoEncoder, fitresult, Xnew) = MMI.predict(m::AutoEncoder, fitresult, Xnew)

 MMI.transform(m::AutoEncoder, fitresult, Xnew) = BetaML.Api.predict(fitresult, MMI.matrix(Xnew))

 MMI.inverse_transform(m::AutoEncoder, fitresult, Xnew) = BetaML.Api.inverse_predict(fitresult, MMI.matrix(Xnew))


 MMI.metadata_model(AutoEncoder,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    output_scitype   = AbstractMatrix{<: Union{MMI.Continuous,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Bmlj.AutoEncoder"
)