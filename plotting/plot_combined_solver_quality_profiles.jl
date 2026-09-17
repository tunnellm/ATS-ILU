#!/usr/bin/env julia

using CSV
using CairoMakie
using DataFrames
using Printf
using Statistics

include(joinpath(@__DIR__, "ATSILUPaperStyle.jl"))
using .ATSILUPaperStyle

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_OUTPUT_DIR = joinpath(
    ROOT,
    "results",
    "analysis_2026-08-02",
    "combined_quality_figures",
)
const SWEEPS = [1, 2, 3]
const DEFAULT_RATIO_MAX = 8.0
const LOGICAL_TO_LAYOUT_ROW = Dict(1 => 1, 2 => 2, 3 => 4, 4 => 5)

const BLOCKS = [
    (
        name = "unsymmetric",
        input = joinpath(
            ROOT,
            "results",
            "analysis_2026-07-31",
            "unsymmetric_gmres",
            "combined_solver_outcomes.csv",
        ),
        pair_column = :matrix,
        baseline_method = "sequential_ilu",
        ratio_label = "GMRES iterations / sequential ILU iterations, τ",
        ratio_subscript = "1",
        methods = [
            (method = "ats_ilu", label = "ATS-ILU", sync_variant = "sync_folded"),
            (method = "parilu", label = "ParILU", sync_variant = "sync"),
        ],
    ),
    (
        name = "spd",
        input = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-02",
            "spd_cg",
            "combined_solver_outcomes.csv",
        ),
        pair_column = :matrix_id,
        baseline_method = "sequential_ic",
        ratio_label = "CG iterations / sequential IC iterations, τ",
        ratio_subscript = "2",
        methods = [
            (method = "ats_ic", label = "ATS-IC", sync_variant = nothing),
            (method = "par_ic", label = "ParIC", sync_variant = nothing),
        ],
    ),
]

function validate_columns(data::DataFrame, pair_column::Symbol)
    required = [
        pair_column,
        :k,
        :method,
        :variant,
        :repeat,
        :sweeps,
        :threads,
        :iterations,
        :converged,
    ]
    missing_columns = setdiff(required, propertynames(data))
    isempty(missing_columns) || error("missing columns: $(join(missing_columns, ", "))")
end

function sequential_reference(data::DataFrame, block)
    rows = filter(:method => ==(block.baseline_method), data)
    rows.pair_id = string.(rows[!, block.pair_column])
    nrow(unique(rows, [:pair_id, :k])) == nrow(rows) ||
        error("$(block.baseline_method) rows are not unique by pair and k")

    eligible = filter(
        row -> coalesce(row.converged == 1, false) && !ismissing(row.iterations) &&
               isfinite(row.iterations) && row.iterations > 0,
        rows,
    )
    rename!(eligible, :iterations => :sequential_iterations)
    return select(eligible, :pair_id, :k, :sequential_iterations)
end

function with_iteration_ratios(selected::DataFrame, sequential::DataFrame, block)
    selected.pair_id = string.(selected[!, block.pair_column])
    selected = innerjoin(selected, sequential, on = [:pair_id, :k], validate = (false, true))
    selected.iteration_ratio = fill(Inf, nrow(selected))
    converged = coalesce.(selected.converged .== 1, false) .&
                .!ismissing.(selected.iterations) .&
                isfinite.(coalesce.(selected.iterations, Inf)) .&
                (coalesce.(selected.iterations, 0) .> 0)
    selected.iteration_ratio[converged] .=
        selected.iterations[converged] ./ selected.sequential_iterations[converged]
    return selected
end

function asynchronous_rows(
    data::DataFrame,
    sequential::DataFrame,
    block,
    method::String,
    threads::Int,
    sweeps::Int,
)
    selected = filter(
        row -> row.method == method && row.variant == "async" &&
               row.threads == threads && row.sweeps == sweeps,
        data,
    )
    selected = with_iteration_ratios(selected, sequential, block)

    repeat_counts = combine(
        groupby(selected, [:pair_id, :k]),
        nrow => :count,
        :repeat => (values -> sort(unique(values)) == [1, 2, 3]) => :valid_repeats,
    )
    if nrow(repeat_counts) != nrow(sequential) || any(repeat_counts.count .!= 3) ||
       !all(repeat_counts.valid_repeats)
        error(
            "$(block.name) $method, $threads threads, $sweeps sweeps is incomplete: " *
            "$(nrow(repeat_counts))/$(nrow(sequential)) pairs",
        )
    end
    return selected
end

function aggregate_geometric_repeats(rows::DataFrame)
    aggregated = combine(
        groupby(rows, [:pair_id, :k]),
        :iteration_ratio => function(values)
            all(value -> isfinite(value) && value > 0, values) || return Inf
            return exp(mean(log.(values)))
        end => :iteration_ratio,
    )
    aggregated.weight = ones(nrow(aggregated))
    return aggregated
end

function synchronous_rows(
    data::DataFrame,
    sequential::DataFrame,
    block,
    method_config,
    sweeps::Int,
)
    isnothing(method_config.sync_variant) && return nothing
    selected = filter(
        row -> row.method == method_config.method &&
               row.variant == method_config.sync_variant && row.sweeps == sweeps,
        data,
    )
    selected = with_iteration_ratios(selected, sequential, block)
    nrow(selected) == nrow(sequential) ||
        error("$(method_config.method) synchronous, $sweeps sweeps is incomplete")
    nrow(unique(selected, [:pair_id, :k])) == nrow(sequential) ||
        error("$(method_config.method) synchronous rows are not unique")
    selected.weight = ones(nrow(selected))
    return selected
end

function profile_coordinates(rows::DataFrame, xlimits)
    xmin, xmax = xlimits
    denominator = sum(rows.weight)
    finite_rows = filter(row -> isfinite(row.iteration_ratio), rows)
    weights_by_ratio = combine(
        groupby(finite_rows, :iteration_ratio),
        :weight => sum => :weight,
    )
    sort!(weights_by_ratio, :iteration_ratio)

    current = sum(
        row.weight for row in eachrow(weights_by_ratio) if row.iteration_ratio <= xmin;
        init = 0.0,
    ) / denominator
    xs = Float64[xmin]
    ys = Float64[current]
    for row in eachrow(weights_by_ratio)
        ratio = Float64(row.iteration_ratio)
        xmin < ratio <= xmax || continue
        push!(xs, ratio)
        push!(ys, current)
        current += row.weight / denominator
        push!(xs, ratio)
        push!(ys, current)
    end
    push!(xs, xmax)
    push!(ys, current)
    return xs, ys
end

function summarize_configuration(rows::DataFrame, block, method, threads, sweeps, variant)
    denominator = sum(rows.weight)
    finite_rows = filter(row -> isfinite(row.iteration_ratio), rows)
    return (
        dataset = block.name,
        method = method,
        variant = variant,
        threads = threads,
        sweeps = sweeps,
        matrix_fill_pairs = nrow(rows),
        finite_pairs = nrow(finite_rows),
        terminal_profile_fraction = nrow(finite_rows) / denominator,
        fraction_at_parity = sum(rows.iteration_ratio .<= 1.0) / denominator,
        median_finite_ratio = isempty(finite_rows.iteration_ratio) ? missing :
                              median(finite_rows.iteration_ratio),
    )
end

function draw_thread_key!(axis, palette, method_label::String, xlimits; include_sync::Bool)
    xmin, xmax = xlimits
    data_x(relative_x) = exp(log(xmin) + relative_x * (log(xmax) - log(xmin)))
    relative_edges = collect(range(0.37, 0.67, length = length(THREADS) + 1))
    relative_centers = (relative_edges[1:end-1] .+ relative_edges[2:end]) ./ 2
    x_edges = data_x.(relative_edges)
    x_centers = data_x.(relative_centers)
    y_bottom, y_top = 0.180, 0.230
    tick_bottom, label_baseline = 0.160, 0.105

    for index in eachindex(palette)
        poly!(
            axis,
            Point2f[
                (x_edges[index], y_bottom),
                (x_edges[index + 1], y_bottom),
                (x_edges[index + 1], y_top),
                (x_edges[index], y_top),
            ],
            color = palette[index],
            strokewidth = 0,
        )
    end
    lines!(
        axis,
        [x_edges[1], x_edges[end], x_edges[end], x_edges[1], x_edges[1]],
        [y_bottom, y_bottom, y_top, y_top, y_bottom],
        color = :black,
        linewidth = 1.0,
    )
    text!(
        axis,
        0.5,
        y_top + 0.025,
        text = method_label,
        space = :relative,
        align = (:center, :bottom),
        fontsize = TITLE_SIZE,
    )
    for x_center in x_centers
        lines!(
            axis,
            [x_center, x_center],
            [y_bottom, tick_bottom],
            color = :black,
            linewidth = 1.0,
        )
    end
    for index in (1, 4, 7)
        text!(
            axis,
            x_centers[index],
            label_baseline,
            text = string(THREADS[index]),
            align = (:center, :baseline),
            fontsize = TICK_LABEL_SIZE,
        )
    end
    text!(
        axis,
        0.355,
        label_baseline + 0.005,
        text = "async.",
        space = :relative,
        align = (:right, :baseline),
        fontsize = TICK_LABEL_SIZE,
    )
    text!(
        axis,
        0.73,
        label_baseline + 0.005,
        text = "threads",
        space = :relative,
        align = (:left, :baseline),
        fontsize = TICK_LABEL_SIZE,
    )
    if include_sync
        for (left, right) in ((0.390, 0.435), (0.465, 0.510))
            lines!(
                axis,
                data_x.([left, right]),
                [0.055, 0.055],
                color = palette[end],
                linewidth = SYNC_LINE_WIDTH,
            )
        end
        text!(
            axis,
            0.530,
            0.055,
            text = "sync.",
            space = :relative,
            align = (:left, :center),
            fontsize = TICK_LABEL_SIZE,
        )
    end
end

function draw_method_label!(axis, method_label::String)
    text!(
        axis,
        0.5,
        0.18,
        text = method_label,
        space = :relative,
        align = (:center, :center),
        fontsize = TITLE_SIZE,
    )
end

function panel_x_ticklabels(logical_row::Int, column::Int, ratio_max::Real)
    logical_row in (2, 4) || return fill("", length(PROFILE_XTICKLABELS))
    labels = copy(PROFILE_XTICKLABELS)
    max_tick_index = findfirst(tick -> tick == ratio_max, PROFILE_XTICKS)
    column == 1 && !isnothing(max_tick_index) && (labels[max_tick_index] = "")
    column == 2 && (labels[1] = "")
    column == 2 && !isnothing(max_tick_index) && (labels[max_tick_index] = "")
    column == 3 && (labels[1] = "")
    return labels
end

function panel_y_ticklabels(logical_row::Int, column::Int)
    column == 1 || return fill("", length(PROFILE_YTICKLABELS))
    labels = copy(PROFILE_YTICKLABELS)
    logical_row in (2, 4) && (labels[1] = labels[end] = "")
    return labels
end

function build_figure(block_data; ratio_max::Real = DEFAULT_RATIO_MAX)
    apply_paper_theme!()
    xlimits = (PROFILE_XLIMS[1], Float64(ratio_max))
    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 457, padding = 1)
    panel_width = paper_length(149, 530)
    panel_height = 95
    panel_gap = paper_length(8, 530)
    summaries = NamedTuple[]

    Label(
        figure[1:5, 0],
        "Fraction of cases with iteration ratio ≤ τ",
        rotation = pi / 2,
        fontsize = AXIS_LABEL_SIZE,
        tellheight = false,
    )
    Label(
        figure[3, 1:3],
        rich(BLOCKS[1].ratio_label, subscript(BLOCKS[1].ratio_subscript)),
        fontsize = AXIS_LABEL_SIZE,
        tellheight = true,
    )
    Label(
        figure[6, 1:3],
        rich(BLOCKS[2].ratio_label, subscript(BLOCKS[2].ratio_subscript)),
        fontsize = AXIS_LABEL_SIZE,
        tellheight = true,
    )

    for (block_index, item) in enumerate(block_data)
        block, data, sequential = item.block, item.data, item.sequential
        for (method_index, method_config) in enumerate(block.methods)
            logical_row = 2 * (block_index - 1) + method_index
            layout_row = LOGICAL_TO_LAYOUT_ROW[logical_row]
            palette = thread_palette(method_index == 1 ? "ats_ilu" : "parilu")
            include_sync = !isnothing(method_config.sync_variant)

            for (column, sweeps) in enumerate(SWEEPS)
                axis = profile_axis(
                    figure[layout_row, column];
                    x_ticklabels = panel_x_ticklabels(logical_row, column, ratio_max),
                    y_ticklabels = panel_y_ticklabels(logical_row, column),
                )
                xlims!(axis, xlimits...)
                if logical_row == 1
                    axis.title = sweeps == 1 ? "1 sweep" : "$sweeps sweeps"
                    axis.titlesize = TITLE_SIZE
                    axis.titlefont = TITLE_FONT
                end
                lines!(
                    axis,
                    [1.0, 1.0],
                    [0.0, 1.0],
                    color = SEQUENTIAL_COLOR,
                    linewidth = 1.0,
                    linestyle = :dot,
                )

                for (thread_index, threads) in enumerate(THREADS)
                    raw_rows = asynchronous_rows(
                        data,
                        sequential,
                        block,
                        method_config.method,
                        threads,
                        sweeps,
                    )
                    rows = aggregate_geometric_repeats(raw_rows)
                    xs, ys = profile_coordinates(rows, xlimits)
                    lines!(axis, xs, ys, color = palette[thread_index], linewidth = PLOT_LINE_WIDTH)
                    push!(
                        summaries,
                        summarize_configuration(
                            rows,
                            block,
                            method_config.method,
                            threads,
                            sweeps,
                            "async",
                        ),
                    )
                end

                if include_sync
                    rows = synchronous_rows(data, sequential, block, method_config, sweeps)
                    xs, ys = profile_coordinates(rows, xlimits)
                    lines!(
                        axis,
                        xs,
                        ys,
                        color = palette[end],
                        linewidth = SYNC_LINE_WIDTH,
                        linestyle = SYNC_LINESTYLE,
                    )
                    push!(
                        summaries,
                        summarize_configuration(
                            rows,
                            block,
                            method_config.method,
                            0,
                            sweeps,
                            "sync",
                        ),
                    )
                end
                if column == 2
                    if block_index == 1
                        draw_thread_key!(
                            axis,
                            palette,
                            method_config.label,
                            xlimits;
                            include_sync = include_sync,
                        )
                    else
                        draw_method_label!(axis, method_config.label)
                    end
                end
            end
        end
    end

    for column in 1:3
        colsize!(figure.layout, column, Fixed(panel_width))
    end
    for row in values(LOGICAL_TO_LAYOUT_ROW)
        rowsize!(figure.layout, row, Fixed(panel_height))
    end
    colgap!(figure.layout, panel_gap)
    rowgap!(figure.layout, 1, panel_gap)
    rowgap!(figure.layout, 2, 1)
    rowgap!(figure.layout, 3, 10)
    rowgap!(figure.layout, 4, panel_gap)
    rowgap!(figure.layout, 5, 1)
    return figure, DataFrame(summaries)
end

function parse_ratio_max(args)
    isempty(args) && return DEFAULT_RATIO_MAX
    length(args) == 2 && args[1] == "--ratio-max" ||
        error("usage: $(PROGRAM_FILE) [--ratio-max 4|8|64]")
    ratio_max = parse(Float64, args[2])
    ratio_max > PROFILE_XLIMS[1] || error("ratio maximum must exceed $(PROFILE_XLIMS[1])")
    return ratio_max
end

function output_suffix(ratio_max)
    ratio_max == DEFAULT_RATIO_MAX && return ""
    rounded = round(Int, ratio_max)
    suffix = ratio_max == rounded ? string(rounded) : replace(string(ratio_max), "." => "p")
    return "_ratio_max_$(suffix)"
end

function main()
    ratio_max = parse_ratio_max(ARGS)
    block_data = NamedTuple[]
    for block in BLOCKS
        data = CSV.read(block.input, DataFrame; stringtype = String)
        validate_columns(data, block.pair_column)
        sequential = sequential_reference(data, block)
        @printf("%s sequential-relative population: %d pairs\n", block.name, nrow(sequential))
        push!(block_data, (block = block, data = data, sequential = sequential))
    end

    figure, summary = build_figure(block_data; ratio_max = ratio_max)
    stem = joinpath(
        DEFAULT_OUTPUT_DIR,
        "solver_quality_relative_sequential_combined$(output_suffix(ratio_max))",
    )
    mkpath(DEFAULT_OUTPUT_DIR)
    CSV.write("$(stem)_summary.csv", summary)
    for path in save_publication_figure(stem, figure)
        println(path)
    end
end

main()
