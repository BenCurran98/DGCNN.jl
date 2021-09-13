function calc_gradient(loss, model, data, labels, mask; num_tries::Int = 3)
    n = 1
    gs = nothing
    while n < num_tries
        @show n
        try
            @show gs = gradient(params(model)) do 
                loss(model, data, labels, mask)
            end
            @show  gs
            return gs
        catch e
            @show e
            sleep(0.5)
            n += 1
            if n >= num_tries
                throw(e)
            end
        end
    end
    return gs
end

function parallel_loss(model, data, labels, mask, loss = crossentropy)
    ls = Zygote.bufferfrom(zeros(Float32, size(data, 3)))
    data_chunks = map(i -> view(data, :, :, i), 1:size(data, 3))
    data_chunks = map(chunk -> reshape(chunk, size(data, 1), size(data, 2), 1), data_chunks)

    label_chunks = map(i -> view(labels, :, :, i), 1:size(labels, 3))
    label_chunks = map(chunk -> reshape(chunk, size(labels, 1), size(labels, 2), 1), label_chunks)

    mask_chunks = map(i -> view(mask, :, :, i), 1:size(mask, 3))
    mask_chunks = map(chunk -> reshape(chunk, size(mask, 1), size(mask, 2), 1), mask_chunks)

    @sync begin
        for i ∈ 1:size(data, 3)
            Base.Threads.@spawn begin
                this_loss = loss(model(data_chunks[i], mask_chunks[i]), label_chunks[i])
                # this_loss = Float32(loss(model(reshape(view(data, :, :, i), (size(data, 1), size(data, 2), 1))), reshape(view(labels, :, :, i), (size(labels, 1), size(labels, 2), 1))))
                ls[i] += this_loss
            end
        end
    end
    return sum(ls)
end

Zygote.@adjoint CUDA.fill(x::Real, dims...) = CUDA.fill(x, dims...), Δ->(sum(Δ), map(_->nothing, dims)...)
Zygote.@adjoint CUDA.zeros(x...) = CUDA.zeros(x...), _ -> map(_ -> nothing, x)

function train!(model, training_data_loader, test_data_loader, loss, opt, epochs, accuracy; scheduler = nothing, model_dir::String = joinpath(@__DIR__, "model_store"), cuda::Bool = false)
    device = cuda && CUDA.functional() ? gpu : cpu
    @show device
    model |> device

    ps = Flux.params(model)

    if !isdir(model_dir)
        mkpath(model_dir)
    end

    num_iter = 1
    @showprogress 0.01 "Training..." for epoch in 1:epochs
        training_accuracy = 0
        println("Training for epoch $epoch")
        training_loss = 0
        n = 1
        for batch in training_data_loader
            @info "$n/$(length(training_data_loader))"

            data, labels, mask = batch
            
            rng = MersenneTwister(50)
            orig_mask_idxs = map(i -> findall(mask[:, :, i]), 1:size(mask, 3))
            mask_sizes = map(i -> length(orig_mask_idxs[i]), 1:size(mask, 3))
            min_mask_size = minimum(mask_sizes)
            mask_idxs = cat(map(i ->  orig_mask_idxs[i][shuffle!(rng, collect(1:length(orig_mask_idxs[i])))[1:min_mask_size]], 1:size(labels, 3))..., dims = 2)

            focus_labels = zeros(size(labels))
            for i ∈ 1:size(mask_idxs, 2)
                for j ∈ 1:size(mask_idxs, 1)
                    focus_labels[mask_idxs[:, i][j][1], mask_idxs[:, i][j][2], i] = labels[mask_idxs[:, i][j][1], mask_idxs[:, i][j][2], i]
                end
            end

            data = data |> device
            focus_labels = focus_labels |> device
            mask = mask |> device

            @time gs = gradient(params(model)) do 
                this_loss = loss(model(data, mask; device = device), focus_labels)
                training_loss += this_loss
                return this_loss
            end
            
            Flux.update!(opt, params(model), gs)
            
            @info "done update"
            
            preds = model(data; device = device) |> cpu

            training_accuracy += DGCNN.accuracy(preds, labels)
            
            n += 1
        end
        
        println("Mean Train Accuracy = $(training_accuracy/n)")
        println("Testing for epoch $epoch")

        if !isnothing(scheduler)
            opt.eta = next!(scheduler)
        end
        
        this_model_dir = joinpath(model_dir, "model_$(UUIDs.uuid4())")
        mkdir(this_model_dir)

        # save the model
        @save joinpath(model_dir, "dgcnn_model_epoch_$(epoch).bson") model
        
        test_loss = 0
        test_accuracy = 0
        for batch in test_data_loader
            data, labels = batch
            preds = model(data)
            this_test_loss = loss(preds, labels)
            this_test_accuracy = DGCNN.accuracy(preds, labels)
            test_loss += this_test_loss
            test_accuracy += this_test_accuracy
        end
        
        println("Mean Test Loss = $(test_loss/length(test_data_loader))")
        println("Mean Test Accuracy = $(test_accuracy/length(test_data_loader))")

        GC.gc()
    end
end

function parallel_loss(model, data, labels, mask)
    loss = crossentropy
    
    data_chunks = map(i -> view(data, :, :, i), 1:size(data, 3))
    data_chunks = map(chunk -> reshape(chunk, size(data, 1), size(data, 2), 1), data_chunks)

    label_chunks = map(i -> view(labels, :, :, i), 1:size(labels, 3))
    label_chunks = map(chunk -> reshape(chunk, size(labels, 1), size(labels, 2), 1), label_chunks)

    mask_chunks = map(i -> view(mask, :, :, i), 1:size(mask, 3))
    mask_chunks = map(chunk -> reshape(chunk, size(mask, 1), size(mask, 2), 1), mask_chunks)

    # println("Separated")
    ls = Zygote.bufferfrom(zeros(Float32, size(data, 3)))

    # models = [deepcopy(model) for _ in 1:size(data, 3)]
    @sync begin
        for i ∈ 1:size(data, 3)
            Base.Threads.@spawn begin
                this_loss = 0
                this_loss += Float32(loss(model(data_chunks[i], mask_chunks[i]), label_chunks[i]))
                ls[i] = this_loss
            end
        end
    end
    return sum(ls)
end

# function parallel_loss(model, data, labels, mask)
#     # @show batches = map(i -> view(reshape(data[:, :, i], (size(data, 1), size(data, 2), 1)), size()), size(data, 3))
#     # @show batch_masks = map(i -> view(reshape(mask[:, :, i], (size(mask, 1), size(mask, 2), 1))), size(mask, 3))
#     # @show batch_labels = map(i -> view(reshape(labels[:, :, i], (size(labels, 1), size(labels, 2), 1))), size(labels, 3))
#     loss = crossentropy
#     # batches = map(i -> view(reshape(data[:, :, i], (size(data, 1), size(data, 2), 1))), 1:size(data, 3))
#     batches = map(i -> reshape(view(data, :, :, i), (size(data, 1), size(data, 2), 1)), 1:size(data, 3))
#     # println("1")
#     # @show size(batches[1])
#     # batch_masks = map(i -> view(reshape(mask[:, :, i], (size(mask, 1), size(mask, 2), 1))), 1:size(mask, 3))
#     batch_masks = map(i -> reshape(view(mask, :, :, i), (size(mask, 1), size(mask, 2), 1)), 1:size(mask, 3))
#     # println("2")
#     # batch_labels = map(i -> view(reshape(labels[:, :, i], (size(labels, 1), size(labels, 2), 1))), 1:size(labels, 3))
#     batch_labels = map(i -> reshape(view(labels, :, :, i), (size(labels, 1), size(labels, 2), 1)), 1:size(labels, 3))
#     # println("Separated")
#     ls = Zygote.bufferfrom(zeros(Float32, size(data, 3)))

#     models = [deepcopy(model) for _ in 1:size(data, 3)]
#     @sync begin
#         for i ∈ 1:size(data, 3)
#             Base.Threads.@spawn begin
#                 this_loss = 0
#                 this_loss += Float32(loss(models[i](reshape(view(data, :, :, i), (size(data, 1), size(data, 2), 1)), reshape(view(mask, :, :, i), (size(mask, 1), size(mask, 2), 1))), reshape(view(labels, :, :, i), (size(labels, 1), size(labels, 2), 1))))
#                 ls[i] = this_loss
#             end
#         end
#     end
#     println("Done sync")
#     @show ls
#     return sum(ls)
# end

function train_from_dir!(model, dir::String, loss, opt, epochs; sample_points::Int = 100000, num_points = 5000, test_prop::Float64 = 0.2)
    training_data, test_data = DGCNN.get_data(dir; sample_points = sample_points, num_points = num_points, test_prop = test_prop)
    train!(model, training_data, test_data, loss, opt, epochs)
end