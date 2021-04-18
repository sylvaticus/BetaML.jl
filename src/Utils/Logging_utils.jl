# Part of submodule Utils of BetaML _ the Beta Machine Learning Toolkit
# Vatious utils to help in logging/debugging

@enum Verbosity NONE=0 LOW=10 STD=20 HIGH=30 FULL=40

"""
    @codeLocation()

Helper macro to print during runtime an info message concerning the code being executed position

"""
macro codeLocation()
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
