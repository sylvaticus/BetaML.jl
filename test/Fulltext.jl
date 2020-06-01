using Test

# https://discourse.julialang.org/t/getting-data-directly-from-a-website/12241/6

fNames = ["train-images-idx3-ubyte","train-labels-idx1-ubyte","t10k-images-idx3-ubyte","t10k-labels-idx1"]
origPath = ["http://yann.lecun.com/exdb/mnist/"]
destPath = joinpath(dirname(Base.find_package("Bmlt")),"..","test","data","mnist")



iris     = readdlm(joinpath(dirname(Base.find_package("Bmlt")),"..","test","data","iris.csv"),',',skipstart=1)

http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

r = HTTP.get(origPath*fNames[1]*".gz", cookies=true);

using HTTP, GZip, IDX # https://github.com/jlegare/IDX.git
r = HTTP.get("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", cookies=true);
destPath = joinpath(dirname(Base.find_package("Bmlt")),"..","test","data","minst")
zippedFile = joinpath(destPath,"test.gz")
unZippedFile = joinpath(destPath,"test.idx3")
open(zippedFile,"w") do f
    write(f,String(r.body))
end
fh = GZip.open(zippedFile)
open(unZippedFile,"w") do f
    write(f,read(fh))
end
train_set = load(unZippedFile)
img1 = train_set[3][:,:,1]



train_set = load(read(fh))

a = "aaa"
b = "bb"
c = a*b
