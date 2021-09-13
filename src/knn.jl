@kernel function knn_kernel(
    @Const(points1),
    @Const(points2),
    @Const(lengths1),
    @Const(lengths2),
    @Const(D),
    @Const(K),
    dists,
    idxs)

    p1, n = @index(Global, NTuple)

    # MinK parameters
    size = 0                    # Size of mink (when less than K)
    max_key = dists[1, p1, n]   # Placeholder for max_key
    max_idx = idxs[1, p1, n]    # Placeholder for max_idx

    # Runs in current thread
    for p2 = 1:lengths2[n]
        dist = eltype(dists)(0)
        for d = 1:D
            dist += (points1[d, p1, n] - points2[d, p2, n])^2
        end
        # Add (dist, p2) to MinK data structure
        if size < K           # Runtime: O(1)
            dists[size + 1, p1, n] = dist
            idxs[size + 1, p1, n] = p2
            if size == 0 || dist > max_key
                max_key = dist
                max_idx = size + 1
            end
            size += 1
        elseif dist < max_key       # Runtime: O(K)
            # Current key replaces old max
            dists[max_idx, p1, n] = dist
            idxs[max_idx, p1, n] = p2
            # Find new max from all dists
            max_key, max_idx = dist, -1
            for i = 1:K
                if dists[i, p1, n] ≥ max_key
                    max_key, max_idx = dists[i, p1, n], i
                end
            end
        end
    end
end

function knn_gpu(X::TArray, Y::TArray, K::Int; exclude_duplicates::Bool = true) where {T <: Real, TArray <: Union{CuArray{T, 3}, PermutedDimsArray}}
    D, P1, N = size(X)
    D2, P2, N2 = size(Y)

    # @assert (D == D2) && (N == N2)

    lengths1 = CUDA.fill(P1, (N,))
    lengths2 = CUDA.fill(P2, (N,))

    dists = CUDA.zeros(Float32, (K, P1, N))
    idxs = CUDA.zeros(Int, (K, P1, N))

    # kernel! = knn_kernel(CUDADevice(), 8)
    # event = kernel!(X, Y, lengths1, lengths2, D, K, dists, idxs, ndrange = (P1, N))
    # wait(event)

    num = 0                    # Size of mink (when less than K)
    max_key = dists[1, :, :]   # Placeholder for max_key
    max_idx = idxs[1, :, :]    # Placeholder for max_idx

    # Runs in current thread
    for p2 = 1:N
        dist = eltype(dists)(0)
        for d = 1:D
            dist += (X[d, :, :] - Y[d, p2, :])^2
        end
        # Add (dist, p2) to MinK data structure
        if size < K           # Runtime: O(1)
            dists[size + 1, p1, n] = dist
            idxs[size + 1, p1, n] = p2
            if size == 0 || dist > max_key
                max_key = dist
                max_idx = size + 1
            end
            size += 1
        elseif dist < max_key       # Runtime: O(K)
            # Current key replaces old max
            dists[max_idx, p1, n] = dist
            idxs[max_idx, p1, n] = p2
            # Find new max from all dists
            max_key, max_idx = dist, -1
            for i = 1:K
                if dists[i, p1, n] ≥ max_key
                    max_key, max_idx = dists[i, p1, n], i
                end
            end
        end
    end

    # lb = exclude_duplicates ? 0 : -Inf
    # k = exclude_duplicates ? K - 1 : K

    Z = CUDA.zeros(Float32, (D, K - 1, P1, N))
    # for i ∈ 1:size(X, 2)
    #     for j ∈ 1:size(X, 3)
    #         Z[:, :, i, j] = X[:, idxs[findall(dists[:, i, j] .> lb), i, j], j]
    #     end
    # end

    # Z = cat(map(_ -> X, 1:K)

    return Z
end

function knn_cpu(X::TArray, Y::TArray, K::Int; exclude_duplicates::Bool = true) where {TArray <: Union{AbstractArray, PermutedDimsArray}}
    F, N, B = size(X)
    Z = zeros(eltype(X), (F, K - 1, N, B))
    lb = exclude_duplicates ? 0 : Inf
    for i ∈ 1:B
        kdtree = KDTree(Y[:, :, i])
        idxs, dists = knn(kdtree, X[:, :, i], K, true)
        Z[:, :, :, i] = cat(map(j -> Y[:, idxs[j][2:end], i], 1:N)..., dims = 3)
    end
    return Z
end

@nograd knn_cpu
@nograd knn_gpu
@nograd knn_kernel