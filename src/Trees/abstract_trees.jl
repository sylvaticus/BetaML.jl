abstract type AbstractNode end      # will be placed elsewhere in the future

struct InfoNode{T} <: AbstractNode
    node    :: DecisionNode{T}
    info    :: NamedTuple
end

struct InfoLeaf{T} <: AbstractNode
    leaf    :: Leaf{T}
    info    :: NamedTuple
end

wrap(node::DecisionNode, info::NamedTuple = NamedTuple()) = InfoNode(node, info)
wrap(leaf::Leaf,         info::NamedTuple = NamedTuple()) = InfoLeaf(leaf, info)

AbstractTrees.children(node::InfoNode) = (
    wrap(node.node.trueBranch, node.info),
    wrap(node.node.falseBranch, node.info)
)
AbstractTrees.children(node::InfoLeaf) = ()

function AbstractTrees.printnode(io::IO, node::InfoNode)
    print(io, node.node)
end

function AbstractTrees.printnode(io::IO, leaf::InfoLeaf)
    print(io, leaf.leaf)
end