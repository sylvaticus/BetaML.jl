"""
    BetaML

The Beta Machine Learning toolkit
https://github.com/sylvaticus/BetaML.jl

Licence is MIT

For documentation, please look at the individual modules or online.

While the code is organised in different sub-modules, all objects are re-exported at the BetaML root level, hence the functionality of this package can be accessed by simply `using BetaML` and then employing the required function directly. 

"""
module BetaML

import MLJModelInterface
const MMI = MLJModelInterface

#import StatsBase

using ForceImport, Reexport, PrecompileTools

include("Api.jl")                   # Shared names across modules
include("Utils/Utils.jl")           # Utility function
include("Stats/Stats.jl")           # Classical statistical functions
include("Nn/Nn.jl")                 # Neural Networks
include("Perceptron/Perceptron.jl") # Perceptron-like algorithms
include("Trees/Trees.jl")           # Decision Trees and ensembles (Random Forests)
include("Clustering/Clustering.jl") # Clustering (hard) algorithms
include("GMM/GMM.jl")               # GMM-based learners (clustering, fitter, regression) 
include("Imputation/Imputation.jl") # (Missing) imputation algorithms
include("Utils/Utils_extra.jl")     # Utility functions that depend on some BetaML functionality. Set them here to avoid recursive dependence
include("Bmlj/Bmlj.jl")               # MLJ Interface module

# "Merging" of the modules...
@force    using .Api
@reexport using .Api
@force    using .Utils
@reexport using .Utils
@force    using .Stats
@reexport using .Stats
@force    using .Nn
@reexport using .Nn
@force    using .Perceptron
@reexport using .Perceptron
@force    using .Trees
@reexport using .Trees
@force    using .Clustering
@reexport using .Clustering
@force    using .GMM
@reexport using .GMM
@force    using .Imputation
@reexport using .Imputation
import .Bmlj # some MLJ models have the same name as BetaML models, set them in a separate interface submodule

# ------------------------------------------------------------------------------
#MLJ interface...
const MLJ_PERCEPTRON_MODELS = (Bmlj.PerceptronClassifier, Bmlj.KernelPerceptronClassifier, Bmlj.PegasosClassifier)
const MLJ_TREES_MODELS      = (Bmlj.DecisionTreeClassifier, Bmlj.DecisionTreeRegressor, Bmlj.RandomForestClassifier, Bmlj.RandomForestRegressor)
const MLJ_CLUSTERING_MODELS = (Bmlj.KMeansClusterer, Bmlj.KMedoidsClusterer, Bmlj.GaussianMixtureClusterer)
const MLJ_IMPUTERS_MODELS   = (Bmlj.SimpleImputer, Bmlj.GaussianMixtureImputer, Bmlj.RandomForestImputer,Bmlj.GeneralImputer) # these are the name of the MLJ models, not the BetaML ones...
const MLJ_NN_MODELS         = (Bmlj.NeuralNetworkRegressor,Bmlj.MultitargetNeuralNetworkRegressor, Bmlj.NeuralNetworkClassifier)
const MLJ_OTHER_MODELS      = (Bmlj.GaussianMixtureRegressor,Bmlj.MultitargetGaussianMixtureRegressor,Bmlj.AutoEncoder)
const MLJ_INTERFACED_MODELS = (MLJ_PERCEPTRON_MODELS..., MLJ_TREES_MODELS..., MLJ_CLUSTERING_MODELS..., MLJ_IMPUTERS_MODELS..., MLJ_NN_MODELS..., MLJ_OTHER_MODELS...) 


#function __init__()
    MMI.metadata_pkg.(MLJ_INTERFACED_MODELS,
        name       = "BetaML",
        uuid       = "024491cd-cc6b-443e-8034-08ea7eb7db2b",     # see your Project.toml
        url        = "https://github.com/sylvaticus/BetaML.jl",  # URL to your package repo
        julia      = true,     # is it written entirely in Julia?
        license    = "MIT",    # your package license
        is_wrapper = false,    # does it wrap around some other package?
    )
#end

include("Precompilation.jl") 

end # module
