function get_data(data_dir; sample_points::Int = 5000, num_train_points::Int = 5000, test_prop::Float64 = 0.2, training_batch_size::Int = 8, test_batch_size::Int = 8)
    all_data = []
    all_files = readdir(data_dir)
    data_files = filter(f -> f[end - 3:end] == ".txt", all_files)
    for f in data_files
        data = readdlm(f, Float64)
        rng = MersenneTwister(4321)
        sample_idxs = shuffle!(rng, 1:size(data)[1])
        data = data[sample_idxs[1:sample_points], :]
        points = data[:, 1:3]
        labels = Int.(data[:, 4])
        mask = DGCNN.get_training_mask(labels, num_points)
        push!(all_data, (points, labels, mask))
    end
    rng = MersenneTwister(420)
    sorted_idxs = shuffle!(rng, 1:length(all_data))
    training_data = all_data[sorted_idxs[1:floor((length(all_data) * (1 - test_prop)))]]
    test_data = all_data[sorted_idxs[floor((length(all_data) * (1 - test_prop))) + 1:end]]

    num_training_batches = floor(length(training_data)/training_batch_size)
    num_test_batches = floor(length(test_data)/test_batch_size)

    training_batches = []
    for training_batch_num in 1:num_training_batches
        this_batch_training_data = training_data[(training_batch_num - 1) * num_training_batches + 1:training_batch_num * num_training_batches]
        batch_data = cat(map(data -> reshape(data[1], (size(data[1])[1], size(data[1])[2], 1)), this_batch_training_data)..., dims = 3)
        batch_labels = cat(map(data -> data[2], this_batch_training_data)..., dims = 1)
        batch_masks = cat(map(data -> data[2], this_batch_training_data)..., dims = 1)
        this_batch = (batch_data, batch_labels, batch_masks)
        push!(training_batches, this_batch)
    end

    test_batches = []
    for test_batch_num in 1:num_test_batches
        this_batch_test_data = test_data[(test_batch_num - 1) * num_test_batches + 1:test_batch_num * num_test_batches]
        batch_data = cat(map(data -> reshape(data[1], (size(data[1])[1], size(data[1])[2], 1)), this_batch_test_data)..., dims = 3)
        batch_labels = cat(map(data -> data[2], this_batch_test_data)..., dims = 1)
        # no mask needed for test sets
        this_batch = (batch_data, batch_labels)
        push!(test_batches, this_batch)
    end
    return training_batches, test_batches
end

function process_pc(filename::String, converted_labels_file::String, outfile::String; class_map::Dict{Int, Int} = POWERCOR_CLASS_MAP)
    convert_pc_labels(filename, converted_labels_file, class_map = class_map)
    parse_pc_data(converted_labels_file; outfile = outfile)
end

function process_pc_dir(orig_dir::String, tmp_tiles_dir::String, tiles_dir::String, outdir::String, tile_size::Real; class_map::Dict{Int, Int} = POWERCOR_CLASS_MAP)
    if !isdir(tiles_dir)
        mkpath(tiles_dir)
    end
    if !isdir(outdir)
        mkpath(outdir)
    end
    # split_tile_dir(orig_dir, tiles_dir, tile_size)
    all_tile_files = readdir(tmp_tiles_dir)
    tile_files = filter(f -> f[end - 3:end] == ".las", all_tile_files)
    n = 1
    for file in tile_files
        @show file
        @info "$n/$(length(tile_files))"
        process_pc(joinpath(tmp_tiles_dir, file), joinpath(tiles_dir, file), joinpath(outdir, split(file, ".", keepempty=false)[1] * ".txt"); class_map = class_map)
        GC.gc()
        n += 1
    end
end

function parse_pc_data(filename::String; outfile::String = "")
    pc = load_pointcloud(filename)
    pc = pc[findall(map(c -> c in values(POWERCOR_CLASS_MAP), unique(pc.classification)))]

    data = DGCNN.points_to_matrix(pc.position, pc.classification)

    if outfile != ""
        open(outfile, "w") do io
            writedlm(io, data)
        end
    end

    GC.gc()

    return data
end

function parse_pc_dir(dir::String)
    all_files = readdir(dir)
    to_process = filter(f -> f[end - 2:end] == ".h5", all_files)
    n = 1
    for f ∈ to_process
        @info "$n/$(length(to_process))"
        n += 1
        parse_pc_data(joinpath(dir, f); outfile = joinpath(dir, f[1:end - 3] * ".txt"))
    end
end

function convert_pc_labels(filename::String, outfile::String; class_map::Dict{Int, Int} = POWERCOR_CLASS_MAP)
    pc = DGCNN.load_pointcloud(filename)
    pc = pc[findall(map(c -> c in collect(keys(class_map)), pc.classification))]
    pc = Table(pc, intensity = Float64.(pc.intensity))

    classifications = zeros(length(pc))
    for lab in unique(collect(keys(class_map)))
        classifications[findall(pc.classification .== lab)] .= class_map[lab]
    end
    
    new_pc = Table(pc, classification = classifications)
    @show unique(new_pc.classification) 
    save_h5(outfile[1:end - 4] * ".h5", new_pc)
    save_las(outfile, new_pc)
end



function convert_dir_labels(dir::String, outdir::String; class_map::Dict{Int, Int} = POWERCOR_CLASS_MAP)
    all_files = readdir(dir)
    h5files = filter(f -> f[end - 3:end] == ".las", all_files)
    if !isdir(outdir)
        mkpath(outdir)
    end

    n = 1
    for file ∈ h5files
        @info "$n/$(length(h5files))"
        convert_pc_labels(joinpath(dir, file), joinpath(outdir, "Area_$n.las"); class_map = class_map)
        GC.gc()
        n += 1
    end
end

function split_boundingbox(bb::BoundingBox, tile_size::Real)
    xs = [x for x in bb.xmin:tile_size:bb.xmax]
    ys = [y for y in bb.ymin:tile_size:bb.ymax]

    if !(bb.xmax in xs) 
        push!(xs, bb.xmax)
    end
    if !(bb.ymax in ys)
        push!(ys, bb.ymax)
    end

    bbs = Vector{BoundingBox}()
    for i ∈ 1:length(xs) - 1
        for j in 1:length(ys) - 1
            this_bb = BoundingBox(xs[i], ys[j], bb.zmin, xs[i + 1], ys[j + 1], bb.zmax)
            push!(bbs, this_bb)
        end
    end
    return bbs
end

function split_tile(filename::String, tile_size::Real; tile_file_base::String = "tile")
    pc = load_las(filename)
    bb = boundingbox(pc.position)
    bbs = split_boundingbox(bb, tile_size)
    for i ∈ 1:length(bbs)
        @info "$i/$(length(bbs))"
        tile_pc = pc[findall(map(p -> p ∈ bbs[i], pc.position))]
        tile_pc = Table(tile_pc, intensity = Float64.(tile_pc.intensity))
        save_las(tile_file_base * "_$(i).las", tile_pc)
        save_h5(tile_file_base * "_$(i).h5", tile_pc)
    end
end
    
function split_tile_dir(dir::String, outdir::String, tile_size::Real)
    all_files = readdir(dir)
    las_files = filter(f -> f[end - 3:end] == ".las", all_files)
    for i ∈ 1:length(las_files)
        @info "file: $i/$(length(las_files))"
        tile_file_base = "Area_$i"
        split_tile(joinpath(dir, las_files[i]), tile_size; tile_file_base = joinpath(outdir, tile_file_base))
    end
end

function get_good_tiles(dir::String, good_dir::String)
    all_files = readdir(dir)
    tile_files = filter(f -> f[end - 3:end] == ".las" && f[1:4] == "Area", all_files)
    for filename ∈ tile_files
        @show filename
        pc = load_las(joinpath(dir, filename))
        num_building = length(findall(pc.classification .== 0))
        num_ground = length(findall(pc.classification .== 1))
        num_veg = length(findall(pc.classification .== 2))
        num_pts = length(pc)
        proportion = 1/10
        if num_building > num_pts * proportion && num_ground > num_pts * proportion && num_veg > num_pts * proportion
            @info "Found good one! File: $filename"
            save_las(joinpath(good_dir, filename), pc)
            save_h5(joinpath(good_dir, filename[1:end - 4] * ".h5"), Table(pc, intensity = Float64.(pc.intensity)))
        end
    end
end