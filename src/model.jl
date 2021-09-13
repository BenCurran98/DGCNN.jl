function conv_bn_block(input, output, kernel_size)
    return Chain(
        Conv((kernel_size,), input=>output),
        BatchNorm(output),
        x -> Float32.(leakyrelu.(x, 0.2)))
end

function fc_bn_block(input, output)
    return Chain(
        Dense(input,output),
        BatchNorm(output),
        x -> Float32.(leakyrelu.(x)))
end

function create_knn_graph(X::AbstractArray, k::Int = 30; device = cpu)
    knn_func = device == cpu ? knn_cpu : knn_gpu
    Z = knn_func(X, X, k; exclude_duplicates = true)
    return Z
end

@nograd create_knn_graph

struct EdgeConv
    layers::AbstractArray
    k::Int
    mlp
    maxpool 
end

function EdgeConv(layers::AbstractArray, k::Int)
    conv_layers = [conv_bn_block(2 * layers[1], layers[2], 1)]
    for i in 2:length(layers) - 1
        push!(conv_layers, conv_bn_block(layers[i], layers[i + 1], 1))
    end
    EdgeConv(layers, k, Chain(conv_layers...), MaxPool((k,)))
end

function (model::EdgeConv)(data; device = cpu)
    orig_data = copy(data)
    # data starts as (F, N, B)
    F, N, B = size(data)

    # batch_knns: (F, k, N, B)
    batch_knns = create_knn_graph(data, model.k + 1; device = device)

    data = reshape(data, F, 1, N, B)
    # data: (F, 1, N, B)
    data = cat([data for _ in 1:model.k]..., dims = 2)
    # data: (F, k, N, B)
    data = cat(data, batch_knns - data, dims = 1)

    # @info "14: $(size(data))"
    # data: (2F, k, N, B)
    data = PermutedDimsArray(data, (2, 3, 1, 4))
    # @info "15: $(size(data))"
    
    # data: (k, N, 2F, B)
    data = reshape(data, N * model.k, 2 * F, B)
    # data: (kN, an, B)
    data = model.mlp(data)
    an = size(data)[2]
    data = reshape(data, model.k, an * N, B)
    # data: (K, anN, B)
    data = model.maxpool(data)
    # data: (1, anN, B)
    data = reshape(data, N, an, B)
    # data: (N, an, B)
    data = PermutedDimsArray(data, (2, 1, 3))
    # data: (an, N, B)
    return data, orig_data
end

@functor EdgeConv

struct Dgcnn_classifier
    edge_conv1::EdgeConv
    edge_conv2::EdgeConv
    edge_conv3::EdgeConv
    conv_bn1
    maxpool
    conv_bn2
    conv_bn3
    drop
    fc
end

function Dgcnn_classifier(num_neigh::Int, num_classes::Int)
    Dgcnn_classifier(
        EdgeConv([3, 64, 64], num_neigh),
        EdgeConv([64, 64, 64], num_neigh),
        EdgeConv([64, 64], num_neigh),
        conv_bn_block(192, 1024, 1),
        x -> maximum(x, dims = 1),
        conv_bn_block(1216, 512, 1),
        conv_bn_block(512, 256, 1),
        Dropout(0.5),
        conv_bn_block(256, num_classes, 1)
    )
end

function (model::Dgcnn_classifier)(data, mask = missing; device = cpu)
    # data: (N, 3, B)
    data = PermutedDimsArray(data, (2, 1, 3))
    # data: (3, N, B)
    F, N, B = size(data)
    data_layer1, ~ = model.edge_conv1(data; device = device)
    # data_layer1: (64, N, B)
    data_layer2, ~ = model.edge_conv2(data_layer1; device = device)
    # data_layer2: (64, N, B)
    data_layer3, ~ = model.edge_conv3(data_layer2; device = device)
    # data_layer3: (64, N, B)
    data = cat([data_layer1, data_layer2, data_layer3]..., dims = 1)
    # data: (192, N, B)
    data = PermutedDimsArray(data, (2, 1, 3))
    # data: (N, 192, B)
    data = model.conv_bn1(data)
    # data: (N, 1024, B)
    data = model.maxpool(data)
    # data: (1, 1024, B)
    data = cat([data for _ in 1:N]..., dims = 1)
    # data: (N, 1024, B)
    data_layer1 = PermutedDimsArray(data_layer1, (2, 1, 3))
    data_layer2 = PermutedDimsArray(data_layer2, (2, 1, 3))
    data_layer3 = PermutedDimsArray(data_layer3, (2, 1, 3))
    
    data = cat([data, data_layer1, data_layer2, data_layer3]..., dims = 2)
    # data: (N, 1216, B)
    data = model.conv_bn2(data)
    # data: (N, 512, B)
    data = model.conv_bn3(data)
    # data: (N, 256, B)
    data = model.drop(data)
    data = model.fc(data)
    # data: (N, num_classes, B)
    data = Float32.(softmax(data, dims = 2))
    # data: (num_classes, N, B)
    data = PermutedDimsArray(data, (2, 1, 3))

    mask = ismissing(mask) ? ones(Float32, size(data)) : mask

    data = data .* mask
    
    # data: (num_classes, N, B)
    return data
end