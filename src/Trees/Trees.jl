"""
  Trees.jl File

Decision Trees implementation (Module BetaML.Trees)

`?BetaML.Trees` for documentation

- [Importable source code (most up-to-date version)](https://github.com/sylvaticus/BetaML.jl/blob/master/src/Trees.jl) - [Julia Package](https://github.com/sylvaticus/BetaML.jl)
- New to Julia? [A concise Julia tutorial](https://github.com/sylvaticus/juliatutorial) - [Julia Quick Syntax Reference book](https://julia-book.com)

"""


"""
    BetaML.Trees module

Implement the functionality required to build a Decision Tree or a whole Random Forest, predict data and assess its performances.

Both Decision Trees and Random Forests can be used for regression or classification problems, based on the type of the labels (numerical or not). You can override the automatic selection with the parameter `forceClassification=true`, typically if your labels are integer representing some categories rather than numbers. For classification problems the output of `predictSingle` is a dictionary with the key being the labels with non-zero probabilitity and the corresponding value its proobability; for regression it is a numerical value.

Please be aware that, differently from most other implementations, the Random Forest algorithm collects and averages the probabilities from the trees, rather than just repording the mode, i.e. no information is lost and the output of the forest classifier is still a PMF.

Missing data on features are supported, both on training and on prediction.

The module provide the following functions. Use `?[type or function]` to access their full signature and detailed documentation:

# Model definition and training:

- `buildTree(xtrain,ytrain)`: Build a single Decision Tree
- `buildForest(xtrain,ytrain)`: Build a "forest" of Decision Trees


# Model predictions and assessment:

- `predict(tree or forest, x)`: Return the prediction given the feature matrix
- `oobError(forest,x,y)`: Return the out-of-bag error estimate
- `Utils.accuracy(ŷ,y))`: Categorical output accuracy
- `Utils.meanRelError(ŷ,y,p)`: L-p norm based error

Features are expected to be in the standard format (nRecords × nDimensions matrices) and the labels (either categorical or numerical) as a nRecords column vector.

Acknowlegdments: originally based on the [Josh Gordon's code](https://www.youtube.com/watch?v=LDRbO9a6XPU)
"""
module Trees

using LinearAlgebra, Random, Statistics, Reexport, CategoricalArrays
using AbstractTrees

using  ForceImport
@force using ..Api
#using ..Api
@force using ..Utils

import Base.print
import Base.show

include("DecisionTrees.jl") # Decision Trees algorithm and API
include("RandomForests.jl") # Random Forests algorithm and API
include("Trees_MLJ.jl")     # MLJ interface
include("abstract_trees.jl")

end # end module
