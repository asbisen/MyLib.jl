# datasets.jl

import Random: shuffle, randperm
import Base: split
using Random


include("utils.jl")

# abstract type for all datasets
abstract type Dataset; end


# image file datasets exists of
# 1. list of filenames of the data
# 2. corresponding target variable
# 3. total number of records (computed by constructor)
struct ImageDataset <: Dataset
    filenames
    targets
    nobs
end


function ImageDataset(filenames, targets)
  # TODO: ensure the vectors are of equal length
  # TODO: should the type be enforced in parameters?
  nobs = length(filenames)
  return ImageDataset(filenames, targets, nobs)
end


Base.length(o::ImageDataset) = return o.nobs
Base.lastindex(o::ImageDataset) = return o.nobs
Base.getindex(o::ImageDataset, n) = return (data=o.filenames[n], target=o.targets[n])

function Base.show(io::IO, o::ImageDataset)
    println("ImageDataset: $(length(o)) Images")
    println("\tdata: consists of filenames")
    println("\ttarget: describes the target")
end

function Random.shuffle(d::ImageDataset; seed=nothing)
    if seed != nothing
        seed = Random.MersenneTwister(seed)
        idx = randperm(seed, d.nobs)
    else
        idx = randperm(d.nobs)
    end
    
    nfilenames, ntargets = d.filenames[idx], d.targets[idx]
    return ImageDataset(nfilenames, ntargets, d.nobs)
end



function Base.split(d::ImageDataset, at; randomize=true, seed=nothing)
    (at <= 0.0) && error("value of at $(at) should be >0 & <1")
    (at >= 1.0) && error("value of at $(at) should be >0 & <1")

    if randomize == true
        d = shuffle(d, seed=seed)
    end

    atidx = ceil(Int, length(d) * at)

    train = ImageDataset(d.filenames[1:atidx], d.targets[1:atidx])
    test = ImageDataset(d.filenames[atidx+1:end], d.targets[atidx+1:end])

    return (train, test)
end



struct ImageDatasetLoader
    data::ImageDataset
    bs::Int
    tfrm::Union{Nothing, Augmentor.Pipeline}

    function ImageDatasetLoader(data, bs, tfrm)
      if bs > length(data)
        println("ERROR: batch size ($bs) cannot be larger than dataset ($(length(data)))")
        throw(BoundsError)
      end
      new(data, bs, tfrm)
    end

end

function ImageDatasetLoader(data::ImageDataset, bs, tfrm, randomize::Bool; seed=nothing)
    if randomize==true
      data = shuffle(data, seed=seed)
    end
    return ImageDatasetLoader(data, bs, tfrm)
end


function Base.iterate(iter::ImageDatasetLoader, state=(_load_to_tensor(iter.data[1:iter.bs]; tfrm=iter.tfrm), 0))
    element, count = state
    count = count + 1
    start_idx = (count*iter.bs)+1

    if count*iter.bs >= length(iter.data)
        return nothing

    elseif (count*iter.bs)+iter.bs > length(iter.data)
        end_idx = lastindex(iter.data)

    else
        end_idx = start_idx + iter.bs
    end

    return (element, (_load_to_tensor(iter.data[start_idx:end_idx]; tfrm=iter.tfrm), count))
end

Base.length(o::ImageDatasetLoader) = o.data.nobs
Base.getindex(o::ImageDatasetLoader, n) = (_img_to_array(o.data[n].data; tfrm=o.tfrm), o.data[n].target)
Base.lastindex(o::ImageDatasetLoader) = length(o.data.nobs)

function Base.show(io::IO, o::ImageDatasetLoader)
    println("ImageDatasetLoader: $(length(o)) Images")
    println("\tbs: represents batch size")
    println("\ttfrm: describes the Augmentor transformations on applied on images") 
end


function colorview(o::Tuple)
    img = colorview(RGB, permutedims(o[1], (3,1,2)))
    return img
end
