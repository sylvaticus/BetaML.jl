"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Neural Networks models

import MLJModelInterface       # It seems that having done this in the top module is not enought
const MMI = MLJModelInterface  # We need to repeat it here
using CategoricalArrays

export NeuralNetworkRegressor, MultitargetNeuralNetworkRegressor, NeuralNetworkClassifier


# Model Structure declarations..
"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for regression of a single dimensional target.

# Parameters:
$(FIELDS)

# Notes:
- data must be numerical
- the label should be be a _n-records_ vector.

# Example:
```julia
julia> modelType                   = @load NeuralNetworkRegressor pkg = "BetaML"
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
BetaML.Nn.NeuralNetworkRegressor

julia> layers                      = [BetaML.DenseLayer(12,20,f=BetaML.relu),BetaML.DenseLayer(20,20,f=BetaML.relu),BetaML.DenseLayer(20,1,f=BetaML.relu)];

julia> model                       = modelType(layers=layers,opt_alg=BetaML.ADAM())
NeuralNetworkRegressor(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.32801116352654236 0.19721617381409956 … 0.17423147551933688 -0.3203352184325144; 0.20325978849525422 0.2753359303406094 … -0.054177947724910136 0.040744621813733006; … ; 0.3614670391623493 0.4184392845285712 … -0.14577760559119207 -0.12430574279080464; -0.04477463648956215 -0.04575413793278821 … 0.2586292045719249 -0.4146332506686543], [0.386016400524039, -0.4120960765923787, -0.37660375260656787, 0.3754674172848425, 0.3933763861297827, -0.09574612456235942, 0.28147281593639867, -0.11333754049443168, -0.19680033976399594, -0.24747338342736486, 0.022885791740458516, -0.34253183385897484, 0.22126071792632201, -0.3539779424727334, -0.37335255502088455, -0.2462814314064721, 0.01620706528968724, -0.3724728631729394, 0.21311037493715396, -0.20613597904524303], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([0.37603456115187256 0.3542546426240723 … -0.0024023384912328916 0.1834672226168586; -0.1535424198342724 -0.07672083294894799 … -0.1433915698536904 -0.1633699269469485; … ; -0.16189872793833512 0.32683924051358165 … -0.08638288054654059 -0.3802058507922781; -0.19558165681593773 0.16664095708205845 … 0.2503794347207368 -0.031688833520039705], [0.021102385823098146, 0.22228546967483392, 0.1300959971946743, -0.20976715493972442, 0.04091175703653677, 0.023810417350970836, 0.2781644696873053, -0.3057357809062001, 0.10103624908600595, 0.12700817756236799, 0.08642857384856573, -0.1675652351991456, -0.17329950695590257, 0.12896500307404696, -0.1484448116427858, -0.24124008136893604, -0.08216916194774915, 0.33079670478470163, 0.19806334350809457, 0.32549757061401846], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([-0.035318774804408926 0.2774737129427495 … 0.07256585990736009 0.229332566953939], [0.39178172498331654], BetaML.Utils.relu, nothing)], 
  loss = BetaML.Utils.squared_cost, 
  dloss = BetaML.Utils.dsquared_cost, 
  epochs = 100, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#69#72"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X, y);

julia> y_est                       = predict(model, fitResults, X)
506-element Vector{Float64}:
 29.67359458542452
 27.72073250260763
  ⋮
 27.259757923962265

```
"""
Base.@kwdef mutable struct NeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`]. Should always assume y and ŷ as matrices, even if the regression task is 1-D
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end

"""
$(TYPEDSIGNATURES)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::NeuralNetworkRegressor, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    ndims(y) > 1 && error("The label should have only 1 dimensions. Use `MultitargetNeuralNetworkRegressor` or `NeuralNetworkClassifier` for multi_dimensional outputs.")
    mi = NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::NeuralNetworkRegressor, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(NeuralNetworkRegressor,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype   = AbstractVector{<: Union{MMI.Continuous,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Nn.NeuralNetworkRegressor"
)

# ------------------------------------------------------------------------------
# Model Structure declarations..
"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for regression of multiple dimensional targets.

# Parameters:
$(FIELDS)

# Notes:
- data must be numerical
- the label should be a _n-records_ by _n-dimensions_ matrix 

# Example:
```julia
julia> using MLJ

julia> X, y                        = @load_boston;

julia> ydouble                     = hcat(y,y);

julia> modelType                   = @load MultitargetNeuralNetworkRegressor pkg = "BetaML"
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
BetaML.Nn.MultitargetNeuralNetworkRegressor

julia> layers                      = [BetaML.DenseLayer(12,50,f=BetaML.relu),BetaML.DenseLayer(50,50,f=BetaML.relu),BetaML.DenseLayer(50,2,f=BetaML.relu)];

julia> model                       = modelType(layers=layers,opt_alg=BetaML.ADAM())
MultitargetNeuralNetworkRegressor(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.14268958168480084 0.1556430517823459 … -0.08125686623988268 -0.2544570399728793; 0.28423814923214763 -0.1372659640176363 … 0.2264470618518154 -0.06631320101636362; … ; 0.02789179672476405 0.28348513690171906 … 0.2871912147350063 0.11554385516710886; -0.06320205436628074 -0.10694711454519892 … -0.10253686449899962 -0.26585990317571573], [0.09338448989761905, 0.2718624735230576, 0.023797261385177626, -0.17917031167475778, 0.15385702004431373, 0.012842680042847276, -0.10232304504376691, -0.13099353498374394, -0.11649189067696844, 0.30591295324151513  …  -0.2972600758671511, -0.177382174249729, -0.26266997240771395, 0.20268565473608047, 0.014804452498253184, 0.24784415091647882, 0.27962551308477157, -0.2880952267241536, 0.26057211923117796, -0.044009535090302976], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([-0.10136741184492606 -0.13038485207770573 … 0.1165162505227173 -0.025817955934162834; -0.20802525780664402 0.15425857417999556 … -0.19434363128519133 0.17652319228668767; … ; -0.10027182894787812 -0.16280219623873593 … -0.16389190054287556 -0.16859625236026915; 0.03561207609341421 -0.05272100409252414 … 0.18362700621532496 -0.11053112518410535], [0.2049701239390826, 0.04727896759708039, 0.22583290172299525, 0.13866713565359567, -0.032397509451043055, 0.041099957445332624, -0.2401413229195337, -0.022035553374859435, -0.2420707290337102, -0.0007123143227169282  …  -0.04350755341649204, 0.13228009527783768, -0.1313043131118029, -0.09176750039253359, 0.17829147060531736, -0.22431760512441942, 0.022861161675965136, -0.022343912739403338, -0.15410438565251305, -0.16252399721019406], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([-0.1865482260376025 -0.12501419399141886 … 0.1502899731849523 0.26034732010433115; -0.2829352616445401 -0.13834226908657268 … -0.016410622720088086 0.0022255074057040414], [0.1750612378638422, 0.16520212643140864], BetaML.Utils.relu, nothing)], 
  loss = BetaML.Utils.squared_cost, 
  dloss = BetaML.Utils.dsquared_cost, 
  epochs = 100, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#69#72"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, -1, X, ydouble);

julia> y_est                       = predict(model, fitResults, X)
506×2 Matrix{Float64}:
 29.7411  28.8886
 25.8501  26.5058
 29.3501  29.9779
  ⋮       
 30.3606  30.6514
 28.2101  28.3246
 24.1113  23.9118
```

"""
Base.@kwdef mutable struct MultitargetNeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `squared_cost`].  Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = squared_cost
    "Derivative of the loss function [def: `dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
    "Random Number Generator (see [`FIXEDSEED`](@ref)) [deafult: `Random.GLOBAL_RNG`]
    "
    rng::AbstractRNG = Random.GLOBAL_RNG
end
"""
$(TYPEDSIGNATURES)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::MultitargetNeuralNetworkRegressor, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    ndims(y) > 1 || error("The label should have multiple dimensions. Use `NeuralNetworkRegressor` for single-dimensional outputs.")
    mi = NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::MultitargetNeuralNetworkRegressor, fitresult, Xnew) = predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(MultitargetNeuralNetworkRegressor,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype   = Union{AbstractVector{<: Union{MMI.Continuous,MMI.Count}},AbstractMatrix{<: Union{MMI.Continuous,MMI.Count}}},
    supports_weights = false,
    load_path        = "BetaML.Nn.MultitargetNeuralNetworkRegressor"
)

# ------------------------------------------------------------------------------

"""
$(TYPEDEF)

A simple but flexible Feedforward Neural Network, from the Beta Machine Learning Toolkit (BetaML) for classification  problems.

# Parameters:
$(FIELDS)

# Notes:
- data must be numerical
- the label should be a _n-records_ by _n-dimensions_ matrix (e.g. a one-hot-encoded data for classification), where the output columns should be interpreted as the probabilities for each categories.

# Example:
```julia
julia> using MLJ

julia> X, y                        = @load_iris;

julia> modelType                   = @load NeuralNetworkClassifier pkg = "BetaML"
[ Info: For silent loading, specify `verbosity=0`. 
import BetaML ✔
BetaML.Nn.NeuralNetworkClassifier

julia> layers                      = [BetaML.DenseLayer(4,8,f=BetaML.relu),BetaML.DenseLayer(8,8,f=BetaML.relu),BetaML.DenseLayer(8,3,f=BetaML.relu),BetaML.VectorFunctionLayer(3,f=BetaML.softmax)];

julia> model                       = modelType(layers=layers,opt_alg=BetaML.ADAM())
NeuralNetworkClassifier(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.13065425957999977 0.3006718045454293 -0.14208182654389845 -0.010396909703178414; 0.048520032692515036 -0.015206389893573924 0.10185996867206404 0.3322496808168578; … ; -0.35259614611009477 0.6482620436066895 0.008337847389667918 -0.12305204287019345; 0.4658422589725906 0.6934957957952972 -0.3085357878320247 0.20222661286207866], [0.36174111580772195, -0.35269496628536656, 0.26811746239579826, 0.5528187653581791, -0.3510634981562191, 0.10825967870150688, 0.3022797568475024, 0.4981155176339185], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([-0.10421417572899494 -0.35499611903472195 … -0.3335182269175171 -0.3985778486065036; 0.23543572035878935 0.59952318489473 … 0.2795331413389591 -0.5720523377542953; … ; 0.2647745208772335 -0.3248093104701972 … 0.3974038426324087 -0.08540125672267229; 0.5192880535413722 0.484381279307307 … 0.5908202412047914 0.3565865691496263], [-0.43847147676332937, -0.0792557647479405, 0.28527379769156247, 0.472161396182901, 0.5499454540456155, -0.24120815998677952, 0.07292491907243237, 0.6046011380800786], BetaML.Utils.relu, nothing), BetaML.Nn.DenseLayer([0.07404458231451261 -0.6297338418338474 … -0.5203349840135756 0.2659245561353357; -0.03739230431842255 -0.7175051212845613 … 0.7131622720546834 -0.6340542706678468; -0.14453639566110688 0.38900994015838364 … 0.5074513955919556 0.34154609716155104], [-0.39346454660088837, -0.3091008284310222, -0.03586152622920202], BetaML.Utils.relu, nothing), BetaML.Nn.VectorFunctionLayer{0}(fill(NaN), 3, 3, BetaML.Utils.softmax, nothing, nothing)], 
  loss = BetaML.Utils.crossentropy, 
  dloss = BetaML.Utils.dcrossentropy, 
  epochs = 100, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#69#72"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  categories = nothing, 
  handle_unknown = "error", 
  other_categories_name = nothing, 
  rng = Random._GLOBAL_RNG())

julia> (fitResults, cache, report) = MLJ.fit(model, 0, X, y);

julia> est_classes                 = predict(model, fitResults, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt8, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>0.57, versicolor=>0.215, virginica=>0.215)
 UnivariateFinite{Multiclass{3}}(setosa=>0.565, versicolor=>0.217, virginica=>0.217)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.255, versicolor=>0.255, virginica=>0.49)
 UnivariateFinite{Multiclass{3}}(setosa=>0.254, versicolor=>0.254, virginica=>0.492)
 UnivariateFinite{Multiclass{3}}(setosa=>0.263, versicolor=>0.263, virginica=>0.473)
```
"""
Base.@kwdef mutable struct NeuralNetworkClassifier <: MMI.Probabilistic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers. The last \"softmax\" layer is automatically added."
    layers::Union{Array{AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `crossentropy`]. Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = crossentropy
    "Derivative of the loss function [def: `dcrossentropy`, i.e. the derivative of the cross-entropy]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = dcrossentropy
    "Number of epochs, i.e. passages trough the whole training sample [def: `1000`]"
    epochs::Int64 = 100
    "Size of each individual batch [def: `32`]"
    batch_size::Int64 = 32
    "The optimisation algorithm to update the gradient at each batch [def: `ADAM()`]"
    opt_alg::OptimisationAlgorithm = ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`"
    cb::Function=fitting_info
    "The categories to represent as columns. [def: `nothing`, i.e. unique training values]."  
    categories::Union{Vector,Nothing} = nothing
    "How to handle categories not seens in training or not present in the provided `categories` array? \"error\" (default) rises an error, \"infrequent\" adds a specific column for these categories."
    handle_unknown::String = "error"
    "Which value during prediction to assign to this \"other\" category (i.e. categories not seen on training or not present in the provided `categories` array? [def: ` nothing`, i.e. typemax(Int64) for integer vectors and \"other\" for other types]. This setting is active only if `handle_unknown=\"infrequent\"` and in that case it MUST be specified if Y is neither integer or strings"
    other_categories_name = nothing
    "Random Number Generator [deafult: `Random.GLOBAL_RNG`]"
    rng::AbstractRNG = Random.GLOBAL_RNG
end
"""

MMI.fit(model::NeuralNetworkClassifier, verbosity, X, y)

For the `verbosity` parameter see [`Verbosity`](@ref))

"""
function MMI.fit(m::NeuralNetworkClassifier, verbosity, X, y)
    x = MMI.matrix(X)                     # convert table to matrix   
    typeof(verbosity) <: Integer || error("Verbosity must be a integer. Current \"steps\" are 0, 1, 2 and 3.")  
    verbosity = Utils.mljverbosity_to_betaml_verbosity(verbosity)
    categories = deepcopy(m.categories)
    if categories == nothing
        #if occursin("CategoricalVector",string(typeof(y))) # to avoid dependency to CategoricalArrays or MLJBase 
        if typeof(y) <: CategoricalVector
            categories = levels(y)
        end
    end

    ohmod = OneHotEncoder(categories=categories,handle_unknown=m.handle_unknown,other_categories_name=m.other_categories_name, verbosity=verbosity)
    Y_oh = fit!(ohmod,y)

    nR,nD       = size(x)
    (nRy,nDy)   = size(Y_oh)         
    
    nR == nRy || error("X and Y have different number of records (rows)")

    if isnothing(m.layers)
        layers = nothing
    else
        layers = deepcopy(m.layers)
        push!(layers,VectorFunctionLayer(nDy,f=softmax))
    end
    mi = NeuralNetworkEstimator(;layers=layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    fit!(mi,x,Y_oh)
    fitresults = (mi,ohmod)
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

function MMI.predict(m::NeuralNetworkClassifier, fitresult, Xnew) 
    nnmod, ohmod = fitresult
    yhat = predict(nnmod, MMI.matrix(Xnew))
    classes = parameters(ohmod).categories_applied
    predictions = MMI.UnivariateFinite(classes, yhat,pool=missing)
    #return yhat
    return predictions
end

 MMI.metadata_model(NeuralNetworkClassifier,
    input_scitype    = MMI.Table(Union{MMI.Continuous,MMI.Count}),
    target_scitype = AbstractVector{<: Union{MMI.Multiclass,MMI.Finite,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Nn.NeuralNetworkClassifier"
)
