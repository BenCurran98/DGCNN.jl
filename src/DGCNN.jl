module DGCNN

using Colors
using DelimitedFiles
using FixedPointNumbers
using Flux
using FugroGeometry
using FugroLAS
using H5IO
using JSON
using LinearAlgebra
using NearestNeighbors
using StaticArrays
using Statistics
using Random
using TypedTables
using Zygote
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib, @functor
using Zygote: @nograd, Params

export Dgcnn_classifier

include("defaults.jl")
include("data.jl")
include("model.jl")
include("utils.jl")
include("train.jl")

end # module
