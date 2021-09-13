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

η = 0.1
β1 = 0.9
β2 = 0.999

opt = ADAM(η, (β1, β2))


function parallel_loss(model, data, labels)
    loss = crossentropy

    data_chunks = map(i -> view(data, :, :, i), 1:size(data, 3))
    data_chunks = map(chunk -> reshape(chunk, size(data, 1), size(data, 2), 1), data_chunks)

    label_chunks = map(i -> view(labels, :, :, i), 1:size(labels, 3))
    label_chunks = map(chunk -> reshape(chunk, size(labels, 1), size(labels, 2), 1), label_chunks)
    # println("Separated")
    ls = Zygote.bufferfrom(zeros(Float32, size(data, 3)))

    # models = [deepcopy(model) for _ in 1:size(data, 3)]
    @sync begin
    for i ∈ 1:size(data, 3)
        Base.Threads.@spawn begin
            this_loss = 0
            this_loss += Float32(loss(model(data_chunks[i]), label_chunks[i]))
            ls[i] = this_loss
            @show i, this_loss
        end
    end
    end
    @show ls
    return sum(ls)
end

# function Base.:(+)(psA::Params, psB::Params)

function parallel_grad(model, data, labels)
    loss = crossentropy

    data_chunks = map(i -> view(data, :, :, i), 1:size(data, 3))
    data_chunks = map(chunk -> reshape(chunk, size(data, 1), size(data, 2), 1), data_chunks)

    label_chunks = map(i -> view(labels, :, :, i), 1:size(labels, 3))
    label_chunks = map(chunk -> reshape(chunk, size(labels, 1), size(labels, 2), 1), label_chunks)
    # println("Separated")
    ls = Zygote.bufferfrom(zeros(Float32, size(data, 3)))

    # models = [deepcopy(model) for _ ∈ 1:size(data, 3)]
    # opts = [deepcopy(opt) for _ ∈ 1:size(data)]

    # models = [deepcopy(model) for _ in 1:size(data, 3)]
    # gs = gradient(params(model)) do 
    #     loss(model(data_chunks[i]), label_chunks[i])
    # end
    all_gs = Vector{Zygote.Grads}()
    @sync begin
        for i ∈ 1:size(data, 3)
            Base.Threads.@spawn begin
                this_gs = gradient(params(model)) do 
                    loss(model(data_chunks[i]), label_chunks[i]) 
                end
                push!(all_gs, this_gs)
            end
        end
    end
    gs = all_gs[1]
    for this_gs ∈ all_gs[2:end]
        gs .+= this_gs
    end

    return gs
end

model = Chain(
                Conv((1,), 3 => 5), 
                BatchNorm(5), 
                x -> Float64.(leakyrelu.(x, 0.2)), 
                x -> Float64.(softmax(x, dims = 2)), 
                x -> PermutedDimsArray(x, (2, 1, 3))
            )
loss = crossentropy

for epoch in 1:2
    training_accuracy = 0
    println("Training for epoch $epoch")
    training_loss = 0
    n = 1
    for j ∈ 1:2
        # @show n
        n += 1
        data = rand(10, 3, 2)
        orig_labels = rand(collect(1:5), (10, 2))
        labels = Flux.onehotbatch(orig_labels, collect(1:5))

        # @time gs = gradient(params(model)) do
        #     this_loss = parallel_loss(model, data, labels)
        #     training_loss += this_loss
        #     return this_loss
        # end

        @time gs = DGCNN.calc_gradient(parallel_loss, model, data, labels)

        # @time gs = parallel_grad(model, data, labels)

        @info "done grads"
        Flux.update!(opt, params(model), gs)
        @info "done update"
        preds = model(data)
        training_accuracy += DGCNN.accuracy(preds, labels)
    end
    
    # GC.gc()
end