"""
    Bmlt

The Beta Machine Toolkit
https://github.com/sylvaticus/Bmlt.jl

For documentation, please look at the individual modules:

- `?Bmlt.Perceptron`: Linear and kernel classifiers
- `?Bmlt.Nn`: Neural networks
- `?Bmlt.Clusters`: Clustering algorithms and collaborative filering using clusters
- `?Bmlt.Utils`: Various utility functions (scale, one-hot, distances, kernels,..)

You can access the functionality of this package either by using the submodule and
then directly the provided functionality (utilities are reexported by each of the
other submodule) or using this root module and then using it to prefix each object
provided by it, e.g.:

```
using Bmlt.Nn
myLayer = DenseLayer(2,3)
```

or

```
using Bmlt
myLayer = Bmlt.DenseLayer(2,3)
```

"""
module Bmlt

include("Utils.jl")
using .Utils
include("Nn.jl")
using .Nn
include("Perceptron.jl")
using .Perceptron
include("Clustering.jl")
using .Clustering

end # module
