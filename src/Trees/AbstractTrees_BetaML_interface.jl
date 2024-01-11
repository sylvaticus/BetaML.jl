"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

"""
Implementation of the `AbstractTrees.jl`-interface 
(see: [AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)).

The functions `children` and `printnode` make up the interface traits of `AbstractTrees.jl`. 
This enables the visualization of a `BetaML/DecisionTree` using a plot recipe.

For more information see [JuliaAI/DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl). 
The file `src/abstract_trees.jl` in that repo serves as a model implementation.
"""

export InfoNode, InfoLeaf, wrapdn, DecisionNode, Leaf

"""
These types are introduced so that additional information currently not present in 
a `DecisionTree`-structure -- namely the feature names  -- 
can be used for visualization.
"""
struct InfoNode{T} <: AbstractTrees.AbstractNode{DecisionNode{T}}
    node    :: DecisionNode{T}
    info    :: NamedTuple
end
AbstractTrees.nodevalue(n::InfoNode) = n.node # round(n.node,sigdigits=4)

struct InfoLeaf{T} <: AbstractTrees.AbstractNode{Leaf{T}}
    leaf    :: Leaf{T}
    info    :: NamedTuple
end
AbstractTrees.nodevalue(l::InfoLeaf) = l.leaf # round(l.leaf,sigdigits=4)

"""
    wrapdn(node:: DecisionNode, ...)

Called on the root node of a `DecsionTree` `dc` in order to add visualization information.
In case of a `BetaML/DecisionTree` this is typically a list of feature names as follows:

`wdc = wrapdn(dc, featurenames = ["Colour","Size"])`
"""

wrapdn(node::DecisionNode, info::NamedTuple = NamedTuple()) = InfoNode(node, info)
wrapdn(leaf::Leaf,         info::NamedTuple = NamedTuple()) = InfoLeaf(leaf, info)
wrapdn(mod::DecisionTreeEstimator, info::NamedTuple = NamedTuple()) = wrapdn(mod.par.tree, info)
wrapdn(m::Union{DecisionNode,Leaf,DecisionTreeEstimator};featurenames=[]) = wrapdn(m,(featurenames=featurenames,))




#### Implementation of the `AbstractTrees`-interface

AbstractTrees.children(node::InfoNode) = (
    wrapdn(node.node.trueBranch, node.info),
    wrapdn(node.node.falseBranch, node.info)
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
        if isa(p, Pair)
            println(io, Pair(p[1],round(p[2],sigdigits=4)))
        elseif isa(p,Number)
            println(io, round(p,sigdigits=4))
        else
            println(io, p)
        end
    end
end

function show(io::IO,node::Union{InfoNode,InfoLeaf})
    #print(io, "Is col $(question.column) $condition $(question.value) ?")
    print(io, "A wrapped Decision Tree")
end

function show(io::IO, ::MIME"text/plain", node::Union{InfoNode,InfoLeaf})
    print(io, "A wrapped Decision Tree")
end