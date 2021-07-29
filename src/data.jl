function parse_h5_data(filename::String; outfile::String = "")
    pc = load_h5(filename)
    xs = map(p -> p[1], pc.position)
    ys = map(p -> p[2], pc.position)
    zs = map(p -> p[3], pc.position)
    
    rs = map(c -> red(c), pc.color)
    gs = map(c -> green(c), pc.color)
    bs = map(c -> blue(c), pc.color)

    classifications = copy(pc.classification)
    classifications[findall((pc.classification .== 2))] .= 3
    classifications[findall((map(c -> c ∉ [1, 3, 5], classifications)))] .= 0
    
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

function convert_pc_labels(file::String, outfile::String, labels_json::String = "/media/ben/T7 Touch/InnovationConference/AHN3-dgcnn.pytorch/params/categories.json")
    pc = load_h5(file)
    new_pc = copy(pc)
    class_dict = JSON.parsefile(labels_json, null = missing)
    classes = Int.(parse.(Float64, (sort(collect(keys(class_dict)), by = k -> class_dict[k]))))
    # @show map(i -> class_dict[classes[i]], 1:length(classes))
    # @show classes
    classifications = copy(pc.classification)
    classifications[findall((pc.classification .== 4))] .= 3
    classifications[findall((map(c -> c ∉ [1, 3, 5, 7], classifications)))] .= 0
    new_classes = zeros(length(classifications))
    for i ∈ 1:length(classes)
        new_classes[findall(classifications .== classes[i])] .= i
        new_pc = Table(new_pc, classification = new_classes)
    end
    save_h5(outfile, new_pc)
end

function convert_dir_labels(dir::String, outdir::String, labels_json::String = "/media/ben/T7 Touch/InnovationConference/AHN3-dgcnn.pytorch/params/categories.json")
    all_files = readdir(dir)
    h5files = filter(f -> f[end - 2:end] == ".h5", all_files)
    if !isdir(outdir)
        mkpath(outdir)
    end
    for file ∈ h5files
        @show file
        convert_pc_labels(joinpath(dir, file), joinpath(outdir, file), labels_json)
    end
end