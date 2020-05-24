
# WIP
risk(x,y,θ,λ) = ((y - dot(x,θ))^2 )/2

function risk(x,y,θ,λ)
    x = makeMatrix(x)
    (n,d) = size(x)
    y = makeColVector(y)
    totRisk = 0.0
    for i in 1:n
        totRisk += ((y[i] - dot(x[i,:],θ))^2 )/2
    end
    return (totRisk / n) + (λ/2)*norm(θ)^2
end

function dRrisk(x,y,θ,λ)
    x = makeMatrix(x)
    (n,d) = size(x)
    y = makeColVector(y)

end
#=
η = 0.1
θ = [1.1 2; 3 4]
▽ = [1.1 2.1;3.1 4]

θ2 = (θ,θ)
▽2 = (▽,▽)

θ3 = 1
▽3 = 1

a = gradientDescentSingleUpdate!(θ,▽,η)
b = gradientDescentSingleUpdate(θ2,▽2,η)
gradientDescentSingleUpdate(θ3,▽3,η)


gradientDescent(θ,▽;η = t ->1/(1+t),λ,tol)


abs(ϵl/size(x)[1] - ϵ/size(x)[1]) < (tol * abs(ϵl/size(x)[1]))

abc(x,y) =
=#
