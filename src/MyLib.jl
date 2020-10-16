module MyLib

using Augmentor
using Images

include("mldata/datasets.jl")


# mldata/datasets.jl
export Dataset, ImageDataset, ImageDatasetLoader, 
       shuffle, split 

end # module
