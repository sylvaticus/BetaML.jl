
module Api


# Shared api trough the modules, i.e. names used by more than one module
# Modules are free to use other functions but these are defined here to avoid name conflicts
# and allows instead Multiple Dispatch to handle them

export BetaMLModel, BetaMLSupervisedModel, BetaMLUnsupervisedModel,
       BetaMLOptionsSet, BetaMLHyperParametersSet, BetaMLLearnableParametersSet,
       predict, fit, fit!, train!, partition, info



abstract type BetaMLModel end
abstract type BetaMLSupervisedModel <: BetaMLModel end
abstract type BetaMLUnsupervisedModel <: BetaMLModel end
abstract type BetaMLOptionsSet end
abstract type BetaMLHyperParametersSet end
abstract type BetaMLLearnableParametersSet end

"""
   train!(m::BetaMLModel,X,[y])

Train a BetaMLModel (i.e. learn the algorithm's parameters) based on data, either only features or features and labels
""" 
train!(::BetaMLModel)  = nothing
"""
   predict(m::BetaMLModel,[X])

Predict new information (including transformation) based on a trained BetaMLModel, eventually applied to new features when the algorithms generalise to new data.
""" 
predict(::BetaMLModel) = nothing

function info(m::BetaMLModel)
   return m.info
end

partition()            = nothing

# old to remove
fit()       = nothing
fit!()      = nothing


end

