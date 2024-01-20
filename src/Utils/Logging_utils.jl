"Part of [BetaML](https://github.com/sylvaticus/BetaML.jl). Licence is MIT."

# Part of submodule Utils of BetaML _ the Beta Machine Learning Toolkit
# Vatious utils to help in logging/debugging



"""
    @codelocation()

Helper macro to print during runtime an info message concerning the code being executed position

"""
macro codelocation()
    return quote
        st = stacktrace(backtrace())
        myf = ""
        for frm in st
            funcname = frm.func
            if frm.func != :backtrace && frm.func!= Symbol("macro expansion")
                myf = frm.func
                break
            end
        end
        println("Running function ", $("$(__module__)"),".$(myf) at ",$("$(__source__.file)"),":",$("$(__source__.line)"))
        println("Type `]dev BetaML` to modify the source code (this would change its location on disk)")
    end
end

"""
$(TYPEDSIGNATURES)

Convert any integer to one of the defined betaml verbosity levels.
Currently "steps" are 0, 10, 20 and 30
"""
function to_betaml_verbosity(i::Integer)
    if i <= 0
        return NONE
    elseif i <= 10
        return LOW
    elseif i <= 20
        return STD
    elseif i <= 30
        return HIGH
    else
        return FULL
    end
end



