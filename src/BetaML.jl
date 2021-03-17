"""
    BetaML

The Beta Machine Learning toolkit
https://github.com/sylvaticus/BetaML.jl

For documentation, please look at the individual modules:

- `?BetaML.Perceptron`: Linear and kernel classifiers
- `?BetaML.Nn`: Neural networks
- `?BetaML.Clusters`: Clustering algorithms and collaborative filering using clusters
- `?BetaML.Utils`: Various utility functions (scale, one-hot, distances, kernels,..)

You can access the functionality of this package either by using the submodule and
then directly the provided functionality (utilities are reexported by each of the
other submodule) or using this root module and then using it to prefix each object
provided by it, e.g.:

```
using BetaML.Nn
myLayer = DenseLayer(2,3)
```

or

```
using BetaML
myLayer = BetaML.DenseLayer(2,3)
```

"""
module BetaML

import MLJModelInterface
const MMI = MLJModelInterface

using ForceImport, Reexport

include("Api.jl")        # Shared names across modules
include("Utils.jl")      # Utility function
include("Nn.jl")         # Neural Networks
include("Perceptron.jl") # Perceptron-like algorithms
include("Trees.jl")      # Decision Trees and ensembles (Random Forests)
include("Clustering.jl") # Clustering algorithms

# "Merging" of the modules...
@force    using .Api
@reexport using .Api
@force    using .Utils
@reexport using .Utils
@force    using .Nn
@reexport using .Nn
@force    using .Perceptron
@reexport using .Perceptron
@force    using .Trees
@reexport using .Trees
@force    using .Clustering
@reexport using .Clustering

# ------------------------------------------------------------------------------
#MLJ interface...
const MLJ_PERCEPTRON_MODELS = (PerceptronClassifier, KernelPerceptronClassifier, PegasosClassifier)
const MLJ_TREES_MODELS      = (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor)
const MLJ_CLUSTERING_MODELS = (KMeans, KMedoids, GMM, MissingImputator)
const MLJ_INTERFACED_MODELS = (MLJ_PERCEPTRON_MODELS..., MLJ_TREES_MODELS..., MLJ_CLUSTERING_MODELS...)

function __init__()
    MMI.metadata_pkg.(MLJ_INTERFACED_MODELS,
        name       = "BetaML",
        uuid       = "024491cd-cc6b-443e-8034-08ea7eb7db2b", # see your Project.toml
        url        = "https://github.com/sylvaticus/BetaML.jl",  # URL to your package repo
        julia      = true,          # is it written entirely in Julia?
        license    = "MIT",       # your package license
        is_wrapper = false,    # does it wrap around some other package?
    )
end


end # module
