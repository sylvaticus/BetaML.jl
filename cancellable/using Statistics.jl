using Statistics

scalars = [1.0,1.1,1.2]
vectors = [[1.0,10.0],[1.1,11],[1.2,12]]
vofv    = [[[1.0,10.0],[100,1000]], [[1.1,11],[111,1111]],[[1.2,12],[122,1222]]]

mean(scalars)
std(scalars)
mean(vectors)
std(vectors)
mean(vofv)
std.(vofv)


mean([[1.1,3.1],[1.3,3.3]])

using DataFrames, Plots

df = DataFrame(group=["A", "B", "C"], total=[7.7, 4.6, 5.1], std_error = [0.04, 0.05, 0.06])

bar(df.group, df.total, c=:blues, lw=0, widen=false)
plot!(1/2:(ncol(df)-1/2), df.total, lw=0, yerror=20*df.std_error, ms=10)

group=["A", "B", "C"]
total=[7.7, 4.6, 5.1]
std_error = [0.04, 0.05, 0.06]

bar(group, total, c=:blues, lw=0, widen=false)
plot!(1/2:(3-1/2), total, lw=0, yerror=20*std_error, ms=10)
