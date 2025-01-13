"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# MLJ interface for Neural Networks models

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
julia> using MLJ

julia> X, y        = @load_boston;

julia> modelType   = @load NeuralNetworkRegressor pkg = "BetaML" verbosity=0
BetaML.Nn.NeuralNetworkRegressor

julia> layers                      = [BetaML.DenseLayer(12,20,f=BetaML.relu),BetaML.DenseLayer(20,20,f=BetaML.relu),BetaML.DenseLayer(20,1,f=BetaML.relu)];

julia> model       = modelType(layers=layers,opt_alg=BetaML.ADAM());
NeuralNetworkRegressor(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.23249759178069676 -0.4125090172711131 … 0.41401934928739 -0.33017881111237535; -0.27912169279319965 0.270551221249931 … 0.19258414323473344 0.1703002982374256; … ; 0.31186742456482447 0.14776438287394805 … 0.3624993442655036 0.1438885872964824; 0.24363744610286758 -0.3221033024934767 … 0.14886090419299408 0.038411663101909355], [-0.42360286004241765, -0.34355377040029594, 0.11510963232946697, 0.29078650404397893, -0.04940236502546075, 0.05142849152316714, -0.177685375947775, 0.3857630523957018, -0.25454667127064756, -0.1726731848206195, 0.29832456225553444, -0.21138505291162835, -0.15763643112604903, -0.08477044513587562, -0.38436681165349196, 0.20538016429104916, -0.25008157754468335, 0.268681800562054, 0.10600581996650865, 0.4262194464325672], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.08534180387478185 0.19659398307677617 … -0.3413633217504578 -0.0484925247381256; 0.0024419192794883915 -0.14614102508129 … -0.21912059923003044 0.2680725396694708; … ; 0.25151545823147886 -0.27532269951606037 … 0.20739970895058063 0.2891938885916349; -0.1699020711688904 -0.1350423717084296 … 0.16947589410758873 0.3629006047373296], [0.2158116357688406, -0.3255582642532289, -0.057314442103850394, 0.29029696770539953, 0.24994080694366455, 0.3624239027782297, -0.30674318230919984, -0.3854738338935017, 0.10809721838554087, 0.16073511121016176, -0.005923262068960489, 0.3157147976348795, -0.10938918304264739, -0.24521229198853187, -0.307167732178712, 0.0808907777008302, -0.014577497150872254, -0.0011287181458157214, 0.07522282588658086, 0.043366500526073104], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.021367697115938555 -0.28326652172347155 … 0.05346175368370165 -0.26037328415871647], [-0.2313659199724562], BetaML.Utils.relu, BetaML.Utils.drelu)], 
  loss = BetaML.Utils.squared_cost, 
  dloss = BetaML.Utils.dsquared_cost, 
  epochs = 100, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#90#93"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);

julia> ŷ    = predict(mach, X);

julia> hcat(y,ŷ)
506×2 Matrix{Float64}:
 24.0  30.7726
 21.6  28.0811
 34.7  31.3194
  ⋮    
 23.9  30.9032
 22.0  29.49
 11.9  27.2438
```
"""
Base.@kwdef mutable struct NeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{BetaML.Nn.AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `BetaML.squared_cost`]. Should always assume y and ŷ as matrices, even if the regression task is 1-D
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = BetaML.Utils.squared_cost
    "Derivative of the loss function [def: `BetaML.dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = BetaML.Utils.dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `200`]"
    epochs::Int64 = 200
    "Size of each individual batch [def: `16`]"
    batch_size::Int64 = 16
    "The optimisation algorithm to update the gradient at each batch [def: `BetaML.ADAM()`]. See `subtypes(BetaML.OptimisationAlgorithm)` for supported optimizers"
    opt_alg::OptimisationAlgorithm = BetaML.Nn.ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `fitting_info`]"
    cb::Function=BetaML.Nn.fitting_info
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
    verbosity = mljverbosity_to_betaml_verbosity(verbosity)
    ndims(y) > 1 && error("The label should have only 1 dimensions. Use `MultitargetNeuralNetworkRegressor` or `NeuralNetworkClassifier` for multi_dimensional outputs.")
    mi = BetaML.Nn.NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::NeuralNetworkRegressor, fitresult, Xnew) = BetaML.Api.predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(NeuralNetworkRegressor,
    input_scitype = Union{
        MMI.Table(Union{MMI.Continuous,MMI.Count}),
        AbstractMatrix{<:Union{MMI.Continuous,MMI.Count}},
    },
    target_scitype   = AbstractVector{<: Union{MMI.Continuous,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Bmlj.NeuralNetworkRegressor"
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

julia> X, y        = @load_boston;

julia> ydouble     = hcat(y, y .*2  .+5);

julia> modelType   = @load MultitargetNeuralNetworkRegressor pkg = "BetaML" verbosity=0
BetaML.Nn.MultitargetNeuralNetworkRegressor

julia> layers                      = [BetaML.DenseLayer(12,50,f=BetaML.relu),BetaML.DenseLayer(50,50,f=BetaML.relu),BetaML.DenseLayer(50,50,f=BetaML.relu),BetaML.DenseLayer(50,2,f=BetaML.relu)];

julia> model       = modelType(layers=layers,opt_alg=BetaML.ADAM(),epochs=500)
MultitargetNeuralNetworkRegressor(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.2591582523441157 -0.027962845131416225 … 0.16044535560124418 -0.12838827994676857; -0.30381834909561184 0.2405495243851402 … -0.2588144861880588 0.09538577909777807; … ; -0.017320292924711156 -0.14042266424603767 … 0.06366999105841187 -0.13419651752478906; 0.07393079961409338 0.24521350531110264 … 0.04256867886217541 -0.0895506802948175], [0.14249427336553644, 0.24719379413682485, -0.25595911822556566, 0.10034088778965933, -0.017086404878505712, 0.21932184025609347, -0.031413516834861266, -0.12569076082247596, -0.18080140982481183, 0.14551901873323253  …  -0.13321995621967364, 0.2436582233332092, 0.0552222336976439, 0.07000814133633904, 0.2280064379660025, -0.28885681475734193, -0.07414214246290696, -0.06783184733650621, -0.055318068046308455, -0.2573488383282579], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.0395424111703751 -0.22531232360829911 … -0.04341228943744482 0.024336206858365517; -0.16481887432946268 0.17798073384748508 … -0.18594039305095766 0.051159225856547474; … ; -0.011639475293705043 -0.02347011206244673 … 0.20508869536159186 -0.1158382446274592; -0.19078069527757857 -0.007487540070740484 … -0.21341165344291158 -0.24158671316310726], [-0.04283623889330032, 0.14924461547060602, -0.17039563392959683, 0.00907774027816255, 0.21738885963113852, -0.06308040225941691, -0.14683286822101105, 0.21726892197970937, 0.19784321784707126, -0.0344988665714947  …  -0.23643089430602846, -0.013560425201427584, 0.05323948910726356, -0.04644175812567475, -0.2350400292671211, 0.09628312383424742, 0.07016420995205697, -0.23266392927140334, -0.18823664451487, 0.2304486691429084], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.11504184627266828 0.08601794194664503 … 0.03843129724045469 -0.18417305624127284; 0.10181551438831654 0.13459759904443674 … 0.11094951365942118 -0.1549466590355218; … ; 0.15279817525427697 0.0846661196058916 … -0.07993619892911122 0.07145402617285884; -0.1614160186346092 -0.13032002335149 … -0.12310552194729624 -0.15915773071049827], [-0.03435885900946367, -0.1198543931290306, 0.008454985905194445, -0.17980887188986966, -0.03557204910359624, 0.19125847393334877, -0.10949700778538696, -0.09343206702591, -0.12229583511781811, -0.09123969069220564  …  0.22119233518322862, 0.2053873143308657, 0.12756489387198222, 0.11567243705173319, -0.20982445664020496, 0.1595157838386987, -0.02087331046544119, -0.20556423263489765, -0.1622837764237961, -0.019220998739847395], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.25796717031347993 0.17579536633402948 … -0.09992960168785256 -0.09426177454620635; -0.026436330246675632 0.18070899284865127 … -0.19310119102392206 -0.06904005900252091], [0.16133004882307822, -0.3061228721091248], BetaML.Utils.relu, BetaML.Utils.drelu)], 
  loss = BetaML.Utils.squared_cost, 
  dloss = BetaML.Utils.dsquared_cost, 
  epochs = 500, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#90#93"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, ydouble);

julia> fit!(mach);

julia> ŷdouble    = predict(mach, X);

julia> hcat(ydouble,ŷdouble)
506×4 Matrix{Float64}:
 24.0  53.0  28.4624  62.8607
 21.6  48.2  22.665   49.7401
 34.7  74.4  31.5602  67.9433
 33.4  71.8  33.0869  72.4337
  ⋮                   
 23.9  52.8  23.3573  50.654
 22.0  49.0  22.1141  48.5926
 11.9  28.8  19.9639  45.5823
```

"""
Base.@kwdef mutable struct MultitargetNeuralNetworkRegressor <: MMI.Deterministic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers"
    layers::Union{Array{BetaML.Nn.AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `BetaML.squared_cost`].  Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = BetaML.Utils.squared_cost
    "Derivative of the loss function [def: `BetaML.dsquared_cost`, i.e. use the derivative of the squared cost]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = BetaML.Utils.dsquared_cost
    "Number of epochs, i.e. passages trough the whole training sample [def: `300`]"
    epochs::Int64 = 300
    "Size of each individual batch [def: `16`]"
    batch_size::Int64 = 16
    "The optimisation algorithm to update the gradient at each batch [def: `BetaML.ADAM()`]. See `subtypes(BetaML.OptimisationAlgorithm)` for supported optimizers"
    opt_alg::OptimisationAlgorithm = BetaML.Nn.ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `BetaML.fitting_info`]"
    cb::Function=BetaML.Nn.fitting_info
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
    verbosity = mljverbosity_to_betaml_verbosity(verbosity)
    ndims(y) > 1 || error("The label should have multiple dimensions. Use `NeuralNetworkRegressor` for single-dimensional outputs.")
    mi = BetaML.Nn.NeuralNetworkEstimator(;layers=m.layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    BetaML.Api.fit!(mi,x,y)
    fitresults = mi
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

 MMI.predict(m::MultitargetNeuralNetworkRegressor, fitresult, Xnew) = BetaML.Api.predict(fitresult, MMI.matrix(Xnew))

 MMI.metadata_model(MultitargetNeuralNetworkRegressor,
    input_scitype = Union{
        MMI.Table(Union{MMI.Continuous,MMI.Count}),
        AbstractMatrix{<:Union{MMI.Continuous,MMI.Count}},
    },
    target_scitype   = AbstractMatrix{<: Union{MMI.Continuous,MMI.Count}},
    supports_weights = false,
    load_path        = "BetaML.Bmlj.MultitargetNeuralNetworkRegressor"
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

julia> X, y        = @load_iris;

julia> modelType   = @load NeuralNetworkClassifier pkg = "BetaML" verbosity=0
BetaML.Nn.NeuralNetworkClassifier

julia> layers      = [BetaML.DenseLayer(4,8,f=BetaML.relu),BetaML.DenseLayer(8,8,f=BetaML.relu),BetaML.DenseLayer(8,3,f=BetaML.relu),BetaML.VectorFunctionLayer(3,f=BetaML.softmax)];

julia> model       = modelType(layers=layers,opt_alg=BetaML.ADAM())
NeuralNetworkClassifier(
  layers = BetaML.Nn.AbstractLayer[BetaML.Nn.DenseLayer([-0.376173352338049 0.7029289511758696 -0.5589563304592478 -0.21043274001651874; 0.044758889527899415 0.6687689636685921 0.4584331114653877 0.6820506583840453; … ; -0.26546358457167507 -0.28469736227283804 -0.164225549922154 -0.516785639164486; -0.5146043550684141 -0.0699113265130964 0.14959906603941908 -0.053706860039406834], [0.7003943613125758, -0.23990840466587576, -0.23823126271387746, 0.4018101580410387, 0.2274483050356888, -0.564975060667734, 0.1732063297031089, 0.11880299829896945], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([-0.029467850439546583 0.4074661266592745 … 0.36775675246760053 -0.595524555448422; 0.42455597698371306 -0.2458082732997091 … -0.3324220683462514 0.44439454998610595; … ; -0.2890883863364267 -0.10109249362508033 … -0.0602680568207582 0.18177278845097555; -0.03432587226449335 -0.4301192922760063 … 0.5646018168286626 0.47269177680892693], [0.13777442835428688, 0.5473306726675433, 0.3781939472904011, 0.24021813428130567, -0.0714779477402877, -0.020386373530818958, 0.5465466618404464, -0.40339790713616525], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.DenseLayer([0.6565120540082393 0.7139211611842745 … 0.07809812467915389 -0.49346311403373844; -0.4544472987041656 0.6502667641568863 … 0.43634608676548214 0.7213049952968921; 0.41212264783075303 -0.21993289366360613 … 0.25365007887755064 -0.5664469566269569], [-0.6911986792747682, -0.2149343209329364, -0.6347727539063817], BetaML.Utils.relu, BetaML.Utils.drelu), BetaML.Nn.VectorFunctionLayer{0}(fill(NaN), 3, 3, BetaML.Utils.softmax, BetaML.Utils.dsoftmax, nothing)], 
  loss = BetaML.Utils.crossentropy, 
  dloss = BetaML.Utils.dcrossentropy, 
  epochs = 100, 
  batch_size = 32, 
  opt_alg = BetaML.Nn.ADAM(BetaML.Nn.var"#90#93"(), 1.0, 0.9, 0.999, 1.0e-8, BetaML.Nn.Learnable[], BetaML.Nn.Learnable[]), 
  shuffle = true, 
  descr = "", 
  cb = BetaML.Nn.fitting_info, 
  categories = nothing, 
  handle_unknown = "error", 
  other_categories_name = nothing, 
  rng = Random._GLOBAL_RNG())

julia> mach        = machine(model, X, y);

julia> fit!(mach);

julia> classes_est = predict(mach, X)
150-element CategoricalDistributions.UnivariateFiniteVector{Multiclass{3}, String, UInt8, Float64}:
 UnivariateFinite{Multiclass{3}}(setosa=>0.575, versicolor=>0.213, virginica=>0.213)
 UnivariateFinite{Multiclass{3}}(setosa=>0.573, versicolor=>0.213, virginica=>0.213)
 ⋮
 UnivariateFinite{Multiclass{3}}(setosa=>0.236, versicolor=>0.236, virginica=>0.529)
 UnivariateFinite{Multiclass{3}}(setosa=>0.254, versicolor=>0.254, virginica=>0.492)
```
"""
Base.@kwdef mutable struct NeuralNetworkClassifier <: MMI.Probabilistic
    "Array of layer objects [def: `nothing`, i.e. basic network]. See `subtypes(BetaML.AbstractLayer)` for supported layers. The last \"softmax\" layer is automatically added."
    layers::Union{Array{BetaML.Nn.AbstractLayer,1},Nothing} = nothing
    """Loss (cost) function [def: `BetaML.crossentropy`]. Should always assume y and ŷ as matrices.
    !!! warning
        If you change the parameter `loss`, you need to either provide its derivative on the parameter `dloss` or use autodiff with `dloss=nothing`.
    """
    loss::Union{Nothing,Function} = BetaML.Utils.crossentropy
    "Derivative of the loss function [def: `BetaML.dcrossentropy`, i.e. the derivative of the cross-entropy]. Use `nothing` for autodiff."
    dloss::Union{Function,Nothing}  = BetaML.Utils.dcrossentropy
    "Number of epochs, i.e. passages trough the whole training sample [def: `200`]"
    epochs::Int64 = 200
    "Size of each individual batch [def: `16`]"
    batch_size::Int64 = 16
    "The optimisation algorithm to update the gradient at each batch [def: `BetaML.ADAM()`]. See `subtypes(BetaML.OptimisationAlgorithm)` for supported optimizers"
    opt_alg::OptimisationAlgorithm = BetaML.Nn.ADAM()
    "Whether to randomly shuffle the data at each iteration (epoch) [def: `true`]"
    shuffle::Bool = true  
    "An optional title and/or description for this model"
    descr::String = "" 
    "A call back function to provide information during training [def: `BetaML.fitting_info`]"
    cb::Function=BetaML.Nn.fitting_info
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
    verbosity = mljverbosity_to_betaml_verbosity(verbosity)
    categories = deepcopy(m.categories)
    if categories == nothing
        #if occursin("CategoricalVector",string(typeof(y))) # to avoid dependency to CategoricalArrays or MLJBase 
        if typeof(y) <: CategoricalVector
            categories = levels(y)
        end
    end

    ohmod = BetaML.Utils.OneHotEncoder(categories=categories,handle_unknown=m.handle_unknown,other_categories_name=m.other_categories_name, verbosity=verbosity)
    Y_oh = BetaML.Api.fit!(ohmod,y)

    nR,nD       = size(x)
    (nRy,nDy)   = size(Y_oh)         
    
    nR == nRy || error("X and Y have different number of records (rows)")

    if isnothing(m.layers)
        layers = nothing
    else
        layers = deepcopy(m.layers)
        push!(layers,BetaML.Nn.VectorFunctionLayer(nDy,f=BetaML.Utils.softmax))
    end
    mi = BetaML.Nn.NeuralNetworkEstimator(;layers=layers,loss=m.loss, dloss=m.dloss, epochs=m.epochs, batch_size=m.batch_size, opt_alg=m.opt_alg,shuffle=m.shuffle, cache=false, descr=m.descr, cb=m.cb, rng=m.rng, verbosity=verbosity)
    BetaML.Api.fit!(mi,x,Y_oh)
    fitresults = (mi,ohmod)
    cache      = nothing
    report     = nothing
    return fitresults, cache, report
 end

function MMI.predict(m::NeuralNetworkClassifier, fitresult, Xnew) 
    nnmod, ohmod = fitresult
    yhat = BetaML.Api.predict(nnmod, MMI.matrix(Xnew))
    classes = BetaML.Api.parameters(ohmod).categories
    predictions = MMI.UnivariateFinite(classes, yhat,pool=missing)
    #return yhat
    return predictions
end

 MMI.metadata_model(NeuralNetworkClassifier,
    input_scitype = Union{
        MMI.Table(Union{MMI.Continuous,MMI.Count}),
        AbstractMatrix{<:Union{MMI.Continuous,MMI.Count}},
    },
    target_scitype = AbstractVector{<: Union{MMI.Multiclass,MMI.Finite}},
    supports_weights = false,
    load_path        = "BetaML.Bmlj.NeuralNetworkClassifier"
)
