function parse_h5_data(filename::String; outfile::String = "")
    pc = load_h5(filename)
    xs = map(p -> p[1], pc.position)
    ys = map(p -> p[2], pc.position)
    zs = map(p -> p[3], pc.position)
    
    # rs = map(c -> red(c), pc.color)
    # gs = map(c -> green(c), pc.color)
    # bs = map(c -> blue(c), pc.color)

    rs = zeros(length(pc))
    gs = zeros(length(pc))
    bs = zeros(length(pc))

    classifications = copy(pc.classification)
    @info "Classes: $(unique(classifications))"
    # classifications[findall(pc.classification .== 6)] .= 1
    # classifications[findall(pc.classification .== 5)] .= 3
    # classifications[findall(pc.classification .== 14)] .= 1
    # classifications[findall(pc.classification .== 17)] .= 1
    # classifications = map(c -> c - 1, classifications)
    # # classifications[findall((pc.classification .== 2))] .= 3
    # # classifications[findall((map(c -> c ∉ [1, 2, 4, 5], classifications)))] .= 3
    # classifications[findall(pc.classification .== 3)] .= 2
    # @info "Final: $(unique(classifications))"
    
    data = hcat(
        xs, ys, zs, rs, gs, bs, pc.point_source_id, zeros(length(pc)), zeros(length(pc)), zeros(length(pc)), pc.numberofreturns, pc.returnnumber, pc.gps_time, pc.intensity, classifications
    )

    if outfile != ""
        open(outfile, "w") do io
            writedlm(io, data)
        end
    end

    GC.gc()

    return data
end

function parse_h5_dir(dir::String)
    all_files = readdir(dir)
    to_process = filter(f -> f[end - 2:end] == ".h5", all_files)
    n = 1
    for f ∈ to_process
        @info "$n/$(length(to_process))"
        n += 1
        parse_h5_data(joinpath(dir, f); outfile = joinpath(dir, f[1:end - 3] * ".txt"))
    end
end

function read_pred_file(filename::String, outfile::String = "pred.las")
    data = readdlm(filename)
    points = map(i -> Point3(data[i, 1:3]...), 1:size(data)[1])
    classification = data[:, 7]
    pc = Table(position = points, classification = classification)
    save_las(outfile, pc)
    return pc
end

function read_pred_dir(dir::String, join::Bool = true)
    all_files = readdir(dir)
    pred_files = filter(f -> f[end - 11:end] == "_pred_gt.txt", all_files)
    pc = nothing
    pc_ground_truth = nothing
    pc_comp = nothing
    for filename in pred_files
        @show filename
        this_pc = read_pred_file(joinpath(dir, filename), "pred_$(filename[1:end - 7]).las")
        ground_truth_labels = vec(readdlm(joinpath(dir, filename[1:end - 12] * "_true_labels.txt")))
        @show length(this_pc)
        @show size(ground_truth_labels)
        this_pc_ground_truth = Table(this_pc, classification = ground_truth_labels)
        save_las("ground_truth_$(filename[1:end - 7]).las", this_pc_ground_truth)

        # truth      | predicted  | class label
        # ground     | building   | 4
        # ground     | vegetation | 5
        # building   | ground     | 6
        # building   | vegetation | 7
        # vegetation | ground     | 8
        # vegetation | building   | 9

        comp_classes = 9 * ones(Int, length(this_pc))
        comp_classes[findall((this_pc_ground_truth.classification .== 0) .& (this_pc.classification .== 1))] .= 3
        comp_classes[findall((this_pc_ground_truth.classification .== 0) .& (this_pc.classification .== 2))] .= 4
        comp_classes[findall((this_pc_ground_truth.classification .== 1) .& (this_pc.classification .== 0))] .= 5
        comp_classes[findall((this_pc_ground_truth.classification .== 1) .& (this_pc.classification .== 2))] .= 6
        comp_classes[findall((this_pc_ground_truth.classification .== 2) .& (this_pc.classification .== 0))] .= 7
        comp_classes[findall((this_pc_ground_truth.classification .== 2) .& (this_pc.classification .== 1))] .= 8
        this_pc_comp = Table(this_pc, classification = comp_classes)
        save_las("class_compare_$(filename[1:end - 7]).las", this_pc_comp)

        if join
            if isnothing(pc)
                pc = this_pc
                pc_ground_truth = this_pc_ground_truth
                pc_comp = this_pc_comp
            else
                if !isnothing(this_pc)
                    pc = vcat(pc, this_pc)
                    pc_ground_truth = vcat(pc_ground_truth, this_pc_ground_truth)
                    pc_comp = vcat(pc_comp, this_pc_comp)
                end
            end
        end
    end
    if join
        save_las("pred_tnris_pc.las", pc)
        save_las("pc_tnris_ground_truth.las", pc_ground_truth)
        save_las("pc_tnris_comp.las", pc_comp)
    end
    return pc, pc_ground_truth, pc_comp
end

function convert_pc_labels(file::String, outfile::String, labels_json::String = "/home/ben/InnovationConference/AHN3-dgcnn.pytorch/params/categories.json")
    # pc = load_h5(file)
    pc = load_las(file)
    # TNRIS
    pc = pc[findall(map(p -> p.classification ∈ [2, 3, 4, 5, 6], pc))]
    @show unique(pc.classification)
    pc = Table(pc, intensity = Float64.(pc.intensity))
    new_pc = copy(pc)
    class_dict = JSON.parsefile(labels_json, null = missing)
    classes = Int.(parse.(Float64, (sort(collect(keys(class_dict)), by = k -> class_dict[k]))))
    # @show map(i -> class_dict[classes[i]], 1:length(classes))
    # @show classes
    classifications = copy(pc.classification)
    # POWERCOR
    # classifications[findall(pc.classification .== 1)] .= 0
    # classifications[findall(pc.classification .== 2)] .= 1
    # classifications[findall(pc.classification .== 4)] .= 2
    # classifications[findall(pc.classification .== 5)] .= 2
    
    # TNRIS
    classifications[findall(pc.classification .== 6)] .= 0
    classifications[findall(pc.classification .== 3)] .= 2
    classifications[findall(pc.classification .== 4)] .= 2
    classifications[findall(pc.classification .== 5)] .= 2
    classifications[findall(pc.classification .== 2)] .= 1
    # classifications[findall(pc.classification .== 14)] .= 1
    # classifications[findall(pc.classification .== 17)] .= 1

    # classifications[findall((pc.classification .== 4))] .= 3
    # classifications[findall((map(c -> c ∉ [1, 3, 5, 7], classifications)))] .= 0
    # new_classes = zeros(length(classifications))
    # for i ∈ 1:length(classes)
    #     new_classes[findall(classifications .== classes[i])] .= i
    #     new_pc = Table(new_pc, classification = new_classes)
    # end
    new_pc = Table(new_pc, classification = classifications)
    @show unique(new_pc.classification) 
    save_h5(outfile[1:end - 4] * ".h5", new_pc)
    save_las(outfile, new_pc)
end

function convert_dir_labels(dir::String, outdir::String, labels_json::String = "/home/ben/InnovationConference/AHN3-dgcnn.pytorch/params/categories.json")
    all_files = readdir(dir)
    h5files = filter(f -> f[end - 3:end] == ".las", all_files)
    if !isdir(outdir)
        mkpath(outdir)
    end

    n = 1
    for file ∈ h5files
        @show file
        convert_pc_labels(joinpath(dir, file), joinpath(outdir, "Area_$n.las"), labels_json)
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
    # @show maximum(xs)

    # @show xs
    # @show ys
    bbs = Vector{BoundingBox}()
    for i ∈ 1:length(xs) - 1
        for j in 1:length(ys) - 1
            # @show xs[i], ys[j], xs[i + 1], ys[j + 1]
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
    
function split_tile_dir(dir::String, tile_size::Real)
    all_files = readdir(dir)
    las_files = filter(f -> f[end - 3:end] == ".las", all_files)
    for i ∈ 1:length(las_files)
        @info "file: $i/$(length(las_files))"
        tile_file_base = "Area_$i"
        split_tile(joinpath(dir, las_files[i]), tile_size; tile_file_base = joinpath(dir, tile_file_base))
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