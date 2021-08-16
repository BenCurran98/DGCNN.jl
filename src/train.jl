function train!(model, training_data_loader, test_data_loader, loss, opt, epochs)
    model_params = params(model)
    model_params = Zygote.Params(model_params)
    for epoch in 1:epochs
        training_accuracies = []
        @info "Training for epoch $epoch"
        training_loss = 0
        for batch in training_data_loader
            data, labels, mask = batch
            pred = model(data)
            pred = maximum(pred, dims = 2)
            pred = reshape(pred, (size(pred)[1] * size(pred)[3]))
            focus_preds = pred[findall(mask)]
            focus_labels = labels[findall(mask)]
            gs = gradient(params) do 
                this_round_loss = loss(focus_pred, focus_labels)
                training_loss += this_round_loss
                return  training_loss
            end
            Flux.update!(opt, model_params, gs)
            push!(training_accuracies, DGCNN.accuracy(pred, labels))
        end
        @info "Mean Train Loss = $(mean(training_loss)), Mean Train Accuracy = $(mean(training_accuracies))"
        @info "Testing for epoch $epoch"
        
        test_loss = 0
        test_accuracies = []
        for batch in test_data_loader
            data, labels = batch
            pred = model(data)
            this_round_loss = loss(pred, labels)
            test_loss += this_round_loss
            push!(test_accuracies, DGCNN.accuracy(pred, labels))
        end
        @info "Mean Test Loss = $(mean(test_loss)), Mean Test Accuracy = $(mean(test_accuracies))"
    end
end

function train_from_dir!(model, dir::String, loss, opt, epochs; test_prop::Float64 = 0.2)
    training_data, test_data = DGCNN.get_data(dir, test_prop)
    train!(model, training_data, test_data, loss, opt, epochs)
end