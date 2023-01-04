"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
Implementation of the `AbstractTrees.jl`-interface 
(see: [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)).

The functions `children` and `printnode` make up the interface traits of `AbstractTrees.jl`. 
This enables the visualization of a `BetaML/DecisionTree` using a plot recipe.

For more information see [JuliaAI/DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl). 
The file `src/abstract_trees.jl` in that repo serves as a model implementation.
"""

export InfoNode, InfoLeaf, wrap, DecisionNode, Leaf

"""
These types are introduced so that additional information currently not present in 
a `DecisionTree`-structure -- namely the feature names  -- 
can be used for visualization.
"""
struct InfoNode{T} <: AbstractTrees.AbstractNode{DecisionNode{T}}
    node    :: DecisionNode{T}
    info    :: NamedTuple
end
AbstractTrees.nodevalue(n::InfoNode) = n.node

struct InfoLeaf{T} <: AbstractTrees.AbstractNode{Leaf{T}}
    leaf    :: Leaf{T}
    info    :: NamedTuple
end
AbstractTrees.nodevalue(l::InfoLeaf) = l.leaf

"""
    wrap(node:: DecisionNode, ...)

Called on the root node of a `DecsionTree` `dc` in order to add visualization information.
In case of a `BetaML/DecisionTree` this is typically a list of feature names as follows:

`wdc = wrap(dc, (featurenames = feature_names, ))`
"""

wrap(node::DecisionNode, info::NamedTuple = NamedTuple()) = InfoNode(node, info)
wrap(leaf::Leaf,         info::NamedTuple = NamedTuple()) = InfoLeaf(leaf, info)
wrap(mod::DecisionTreeEstimator, info::NamedTuple = NamedTuple()) = wrap(mod.par.tree, info)
wrap(m::Union{DecisionNode,Leaf,DecisionTreeEstimator};feature_names=[]) = wrap(m,(featurenames=feature_names,))




#### Implementation of the `AbstractTrees`-interface

AbstractTrees.children(node::InfoNode) = (
    wrap(node.node.trueBranch, node.info),
    wrap(node.node.falseBranch, node.info)
)
AbstractTrees.children(node::InfoLeaf) = ()

function AbstractTrees.printnode(io::IO, node::InfoNode)
    q = node.node.question
    condition = isa(q.value, Number) ?  ">=" : "=="
    col = :featurenames ∈ keys(node.info) ? node.info.featurenames[q.column] : q.column
    print(io, "$(col) $condition $(q.value)?")
end

function AbstractTrees.printnode(io::IO, leaf::InfoLeaf)
    for p in leaf.leaf.predictions
        println(io, p)
    end
end

function show(io::IO,node::Union{InfoNode,InfoLeaf})
    #print(io, "Is col $(question.column) $condition $(question.value) ?")
    print(io, "A wrapped Decision Tree")
end

function show(io::IO, ::MIME"text/plain", node::Union{InfoNode,InfoLeaf})
    print(io, "A wrapped Decision Tree")
end