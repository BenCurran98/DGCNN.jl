module DGCNN

using BSON
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
using ParameterSchedulers
using ProgressMeter
using StaticArrays
using Statistics
using Random
using TypedTables
using UUIDs
using Zygote
using BSON: @save, @load
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib, @functor
using ParameterSchedulers: next!
using Zygote: @nograd, Params

export get_data
export Dgcnn_classifier
export train!

include("defaults.jl")
include("data.jl")
include("model.jl")
include("utils.jl")
include("train.jl")

end # module
