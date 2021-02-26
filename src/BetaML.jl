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

include("Utils.jl")
using .Utils
include("Nn.jl")
using .Nn
include("Perceptron.jl")
using .Perceptron
include("Trees.jl")
using .Trees
include("Clustering.jl")
using .Clustering

# ------------------------------------------------------------------------------
#MLJ interface...
#=
const ALL_MODELS = (DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier, RandomForestRegressor)

MMI.metadata_pkg.(ALL_MODELS,
    name       = "BetaML",
    uuid       = "024491cd-cc6b-443e-8034-08ea7eb7db2b", # see your Project.toml
    url        = "https://github.com/sylvaticus/BetaML.jl",  # URL to your package repo
    julia      = true,          # is it written entirely in Julia?
    license    = "MIT",       # your package license
    is_wrapper = false,    # does it wrap around some other package?
)
=#
end # module
