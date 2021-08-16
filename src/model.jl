function conv_bn_block(input, output, kernel_size)
    return Chain(
        Conv((kernel_size,), input=>output),
        BatchNorm(output),
        x -> leakyrelu.(x, 0.2))
end

function fc_bn_block(input, output)
    return Chain(
        Dense(input,output),
        BatchNorm(output),
        x -> leakyrelu.(x))
end

function create_knn_graph(X::AbstractArray, k::Int)
    ~, N = size(X)
    kdtree = KDTree(X)
    cat([X[:,(knn(kdtree, X[:,i], k+1, true)[1][2:k+1])] for i in 1:N]..., dims=3)
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

function (model::EdgeConv)(data)
    # data starts as (F, N, B)
    F, N, B = size(data)
    # knn: (F, k, N, B)
    knn = cat([create_knn_graph(data[:, :, i], model.k) for i in 1:B]..., dims = 4)
    data = reshape(data, F, 1, N, B)
    # data: (F, 1, N, B)
    data = cat([data for _ in 1:model.k]..., dims = 2)
    # data: (F, k, N, B)
    data = cat(data, knn - data, dims = 1)
    # data: (2F, k, N, B)
    data = PermutedDimsArray(data, (2, 3, 1, 4))
    
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
    return data
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

function (model::Dgcnn_classifier)(data)
    # data: (N, 3, B)
    data = PermutedDimsArray(data, (2, 1, 3))
    # data: (3, N, B)
    F, N, B = size(data)
    data_layer1 = model.edge_conv1(data)
    # data_layer1: (64, N, B)
    data_layer2 = model.edge_conv2(data)
    # data_layer2: (64, N, B)
    data_layer3 = model.edge_conv3(data)
    # data_layer3: (64, N, B)
    data = cat([data_layer1, data_layer2, data_layer3]..., dims = 1)
    # data: (192, N, B)
    data = PermutedDimsArray(data, (2, 1, 3))
    # data: (N, 192, B)
    data = model.conv_bn1(data)
    # data: (N, 1024, B)
    data = model.maxpool(data)
    # data: (1, 1024, B)
    data = cat([data for _ in 1:N], dims = 3)
    # data: (N, 1024, B)
    data = cat([data, data_layer1, data_layer2, data_layer3]..., dims = 2)
    # data: (N, 1216, B)
    data = data.conv_bn2(data)
    # data: (N, 512, B)
    data = model.conv_bn3(data)
    # data: (N, 256, B)
    data = model.drop(data)
    data = model.fc(data)
    # data: (N, num_classes, B)
    data = softmax(data, dims = 2)
    # data: (N, num_classes, B)
    return data
end