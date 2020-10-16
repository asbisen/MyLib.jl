
function _img_to_array(filename; tfrm=nothing)
    img = Images.load(filename)
    if tfrm != nothing
        img = augment(img, tfrm)
    end
    img = permutedims(channelview(img), (2,3,1))
    return Float64.(img)
end


function _load_to_tensor(d; tfrm=nothing)
    data=_img_to_array.(d.data, tfrm=tfrm)
    dtensor = cat(data..., dims=4)
    target = d.target
    return (dtensor, target)
end
