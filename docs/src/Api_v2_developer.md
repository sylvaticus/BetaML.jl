# Api v2 - developer documentation (API implementation)

```
BetaMLOptionsSet
BetaMLHyperParametersSet
BetaMLLearnedParametersSet
BetaMLModel

BetaMLSuperVisedModel   <: BetaMLModel
BetaMLUnsupervisedModel <: BetaMLModel
RFModel                 <: BetaMLSuperVisedModel


RFOptionsSet           <: BetaMLOptionsSet
RFHyperParametersSet   <: BetaMLHyperParametersSet
RFLearnedParametersSet <: BetaMLLearnedParametersSet

mutable struct DTModel <: BetaMLSupervisedModel
    hpar::DTHyperParametersSet
    opt::DTOptionsSet
    par::DTLearnableParameters
    cres::T # cached results
    trained::Bool
    info
end
```
