using DGCNN

using BSON
using Colors
using CUDA
using DelimitedFiles
using Distributed
using FixedPointNumbers
using Flux
using FugroGeometry
using FugroLAS
using H5IO
using JSON
using KernelAbstractions
using LinearAlgebra
using Logging
using NearestNeighbors
using ParameterSchedulers
using ProgressMeter
using SharedArrays
using StaticArrays
using Statistics
using Random
using TensorBoardLogger
using TypedTables
using UUIDs
using Zygote
using BSON: @save, @load
using Flux: onehotbatch, onecold, onehot, crossentropy, throttle, NNlib, @functor
using ParameterSchedulers: next!
using Zygote: @nograd, Params


using Flux.Losses
using ParameterSchedulers: Stateful

orig_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining-orig"
tmp_tiles_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining-tiled-old-labels"
tiles_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining-tiled"
outdir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining"

tile_size = 30

# DGCNN.process_pc_dir(orig_dir, tmp_tiles_dir, tiles_dir, outdir, tile_size)

# η = 0.001
η = 0.1
β1 = 0.9
β2 = 0.999

opt = ADAM(η, (β1, β2))

epochs = 3

η_min = 1e-3
scheduler = Stateful(Cos(λ0 = η_min, λ1 = η, period = epochs))

model_dir = joinpath(@__DIR__, "model_store")

num_neigh = 30
num_classes = 5

batch_size = 1

model = Dgcnn_classifier(num_neigh, num_classes)

loss = crossentropy

function accuracy(x, y)
    preds = model(x)
    preds = reshape(preds, size(y))
    return mean(Flux.onecold(preds, 1:num_classes) .== Flux.onecold(y, 1:num_classes))
end

num_box_samples = 3
box_size = 30
class_min = 1
classes = [1, 2, 3, 4, 5]
num_points = 100
sample_points = 5_000

data_dir = "/media/ben/T7 Touch/InnovationConference/Datasets/QualityTraining"
training_batches, test_batches = get_data(data_dir; 
                                            sample_points = sample_points,
                                            num_points = num_points, 
                                            test_prop = 0.2,
                                            num_box_samples = num_box_samples,
                                            box_size = box_size,
                                            class_min = class_min,
                                            classes = classes,
                                            training_batch_size = batch_size,
                                            test_batch_size = batch_size);

@show length(training_batches)

train!(model, training_batches, test_batches, loss, opt, epochs, accuracy; scheduler = scheduler, model_dir = model_dir)