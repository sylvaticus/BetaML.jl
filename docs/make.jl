
# To build the documentation:
#    - julia --project="." make.jl
#    - empty!(ARGS); include("make.jl")
# To build the documentation without running the tests:
#    - julia --project="." make.jl preview
#    - push!(ARGS,"preview"); include("make.jl")

# Format notes:


# # A markdown H1 title
# A non-code markdown normal line

## A comment within the code chunk

#src: line exclusive to the source code and thus filtered out unconditionally
using Test

using Documenter, Literate, BetaML
import BetaML.Bmlj

if "preview" in ARGS
    println("*** Attention: code in the tutorial will not be run/tested")
else
    println("*** Building documentation and testing tutorials...")
end

# Documentation source and temp dir
const SRC_ROOTDIR = joinpath(@__DIR__, "src/")
#const SRC_TEMPDIR = joinpath(@__DIR__, "srctemp/")


push!(LOAD_PATH,"../src/") # this is the source of the code, not the documentation


const _TUTORIAL_DIR = joinpath(SRC_ROOTDIR, "tutorials")
# Important: If some tutorial is removed but the md file is left, this may still  be used by Documenter
_TUTORIAL_SUBDIR = [
    #"Getting started",
    "Classification - cars",
    "Regression - bike sharing",
    "Clustering - Iris",
    "Multi-branch neural network",
    "Feature importance"
]

function link_example(content)
    edit_url = match(r"EditURL = \"(.+?)\"", content)[1]
    footer = match(r"^(---\n\n\*This page was generated using)"m, content)[1]
    content = replace(
        content, footer => "[View this file on Github]($(edit_url)).\n\n" * footer
    )
    return content
end

function _file_list(full_dir, relative_dir, extension)
    return map(
        file -> joinpath(relative_dir, file),
        filter(file -> endswith(file, extension), sort(readdir(full_dir))),
    )
end

"""
    _include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function _include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function literate_directory(dir)
    rm.(_file_list(dir, dir, ".md"))
    for filename in _file_list(dir, dir, ".jl")
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        if ! ("preview" in ARGS)
            @testset "$(filename)" begin
               _include_sandbox(filename)
             end
             Literate.markdown(
                 filename,
                 dir;
                 documenter = true,
                 postprocess = link_example
             )
            # Literate.notebook(
            #     filename,
            #     dir;
            #     documenter = true,
            #     postprocess = link_example
            # )
        else
            Literate.markdown(
                filename,
                dir;
                documenter = true,
                postprocess = link_example,
                codefence =  "```text" => "```"
            )
           # Literate.notebook(
           #     filename,
           #     dir;
           #     documenter = true,
           #     postprocess = link_example,
           #     codefence =  "```text" => "```"
           # )
        end

    end
    return nothing
end

function preprocessDoc()
    rm(SRC_TEMPDIR,recursive=true,force=true)
    mkdir(SRC_TEMPDIR)
    cp(SRC_ROOTDIR, SRC_TEMPDIR; force=true)
    repl = r"`\?(\w+)`" => s"[`\1`](@ref)"
    for (root, dirs, files) in walkdir(SRC_TEMPDIR)
        #println("Files in $root")
        for file in files
            if endswith(file,".md") || endswith(file,".jl")
                pathfile = joinpath(root, file)
                #println(pathfile) # path to files
                write(pathfile, replace(read(pathfile, String), repl))
    
            end
        end
    end
    touch(joinpath(SRC_TEMPDIR,".gitkeep"))
end




println("Starting literating tutorials (.jl --> .md)...")
literate_directory.(joinpath.(_TUTORIAL_DIR, _TUTORIAL_SUBDIR))

println("Starting making the documentation...")
#preprocessDoc()
makedocs(sitename="BetaML.jl Documentation",
         authors = "Antonello Lobianco",
         pages = [
            "Index" => "index.md",
            "Tutorial" => vcat("tutorials/Betaml_tutorial_getting_started.md",map(
                    subdir -> subdir => map(
                        file -> joinpath("tutorials", subdir, file),
                        filter(
                            file -> endswith(file, ".md"),
                            sort(readdir(joinpath(_TUTORIAL_DIR, subdir))),
                        ),
                    ),
                    _TUTORIAL_SUBDIR,
                ),
            ),
            "API (Reference manual)"     => [
              "API V2 (current)" => [
                "Introduction for user"  => "Api_v2_user.md",
                "For developers"         => [
                    "API implementation" => "Api_v2_developer.md",
                    "Style guide"        => "StyleGuide_templates.md"],
                "The Api module"         => "Api.md",
                ],
              "Perceptron"               => "Perceptron.md",
              "Trees"                    => "Trees.md",
              "Nn"                       => "Nn.md",
              "Clustering"               => "Clustering.md",
              "GMM"                      => "GMM.md",
              "Imputation"               => "Imputation.md",
              "Utils"                    => "Utils.md",
            ],
            "MLJ interface"            => "MLJ_interface.md",
#            "Benchmarks"               => "Benchmarks.md", #TODO: until https://github.com/IanButterworth/SystemBenchmark.jl/issues/64 is solved
         ],
         format = Documenter.HTML(prettyurls = false, analytics = "G-JYKX8QY5JW"),
         warnonly = true,
         #strict = true,
         #doctest = false,
         #source  = SRC_TEMPDIR, # Attention here !!!!!!!!!!!
)
println("Starting deploying the documentation...")
deploydocs(
    repo = "github.com/sylvaticus/BetaML.jl.git",
)
