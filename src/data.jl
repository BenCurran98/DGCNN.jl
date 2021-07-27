function parse_h5_data(filename::String; outfile::String = "")
    pc = load_h5(filename)
    xs = map(p -> p[1], pc.position)
    ys = map(p -> p[2], pc.position)
    zs = map(p -> p[3], pc.position)
    
    rs = map(c -> red(c), pc.color)
    gs = map(c -> green(c), pc.color)
    bs = map(c -> blue(c), pc.color)

    classifications = copy(pc.classification)
    classifications[findall((pc.classification .== 4) .| (pc.classification .== 3))] .= 5
    classifications[findall((map(c -> c ∉ [2, 5, 6, 14, 22], classifications)))] .= 0
    
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