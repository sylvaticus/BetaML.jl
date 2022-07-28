
module Api


# Shared api trough the modules, i.e. names used by more than one module
# Modules are free to use other functions but these are defined here to avoid name conflicts
# and allows instead Multiple Dispatch to handle them

export BetaMLModel, BetaMLSupervisedModel, BetaMLUnsupervisedModel,
       BetaMLOptionsSet, BetaMLHyperParametersSet, BetaMLLearnableParametersSet,
       predict, fit!, partition, info, reset!

abstract type BetaMLModel end
abstract type BetaMLSupervisedModel <: BetaMLModel end
abstract type BetaMLUnsupervisedModel <: BetaMLModel end
abstract type BetaMLOptionsSet end
abstract type BetaMLHyperParametersSet end
abstract type BetaMLLearnableParametersSet end

"""
   fit!(m::BetaMLModel,X,[y])

Fit ("train") a BetaMLModel (i.e. learn the algorithm's parameters) based on data, either only features or features and labels
""" 
fit!(::BetaMLModel)  = nothing
"""
   predict(m::BetaMLModel,[X])

Predict new information (including transformation) based on a trained BetaMLModel, eventually applied to new features when the algorithms generalise to new data.
""" 
predict(::BetaMLModel) = nothing

function info(m::BetaMLModel)
   return m.info
end
function reset!(m::BetaMLModel)
   m.par     = nothing
   m.info    = Dict{Symbol,Any}()
   m.trained = false 
   return true
end

partition()            = nothing


end

