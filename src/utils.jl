function accuracy(predicted, target)
    correct_preds = predicted .== target
    return mean(correct_preds)
end

function get_training_mask(labels::AbstractArray, num_points::Int)
    unique_labels = unique(labels)
    class_counts = map(c -> count(labels .== c), unique(labels))
    min_class_count = minimum(class_counts)
    num_points = min(min_class_count, num_points)
    rng = MersenneTwister(1234)
    mask = zeros(Bool, size(labels))
    for class in unique_labels
        this_class_idxs = findall(labels .== class)
        shuffled_idxs = shuffle!(rng, this_class_idxs)
        selected_idxs = shuffled_idxs[1:num_points]
        mask[selected_idxs] .= true
    end
    return mask
end

function load_pointcloud(filename::String)
    pc = nothing
    if split(filename, ".")[end] == "h5"
        pc = load_h5(filename)
    elseif split(filename, ".")[end] == "las"
        pc = load_las(filename)
    else
        error("Unrecognised pointcloud format!")
    end
    return pc
end

function points_to_matrix(points::Vector{Point3}, classification::Vector{I}) where {I <: Integer}
    data_mat = zeros((length(points), 4))
    for i in 1:length(points)
        data_mat[i, :] = [points[i][1], points[i][2], points[i][3], classification[i]]
    end
    return data_mat
end

# function shared_range()