#!/usr/bin/env julia

using CSV
using CairoMakie
using DataFrames
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "plotting", "ATSILUPaperStyle.jl"))
using .ATSILUPaperStyle

const OUTPUT_DIR = @__DIR__
const TIMING_THREADS = THREADS
const STRONG_SCALING_THREADS = vcat(1, THREADS)

const SPEEDUP_MIN = 0.5
const SPEEDUP_MAX = 80.0
const SPEEDUP_TICKS = [0.5, 1, 2, 4, 8, 16, 32, 64]
const SPEEDUP_TICK_LABELS = ["0.5", "1", "2", "4", "8", "16", "32", "64"]

const ATS_TIMING_COLORS = thread_palette("ats_ilu")
const PAR_TIMING_COLORS = thread_palette("parilu")

const DATASETS = [
    (
        name = "unsymmetric",
        timing_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-03",
            "unsymmetric_timing",
            "selected_timing_rows.csv",
        ),
        coverage_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-03",
            "unsymmetric_timing",
            "pair_coverage.csv",
        ),
    ),
    (
        name = "spd",
        timing_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-03",
            "spd_timing",
            "selected_timing_rows.csv",
        ),
        coverage_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-03",
            "spd_timing",
            "pair_coverage.csv",
        ),
    ),
]

function fixed_pairs(path::String)
    coverage = CSV.read(path, DataFrame; stringtype = String)
    return Set(
        (String(row.matrix), Int(row.k))
        for row in eachrow(filter(:fully_complete => ==(1), coverage))
    )
end

function timing_medians(specification)
    pairs = fixed_pairs(specification.coverage_path)
    rows = CSV.read(specification.timing_path, DataFrame; stringtype = String)
    rows = filter(
        row -> (String(row.canonical_matrix), Int(row.k)) in pairs,
        rows,
    )
    medians = combine(
        groupby(rows, [:canonical_matrix, :k, :family, :threads]),
        :seconds_per_sweep => median => :seconds_per_sweep,
    )
    medians.dataset .= specification.name
    return medians, pairs
end

function speedup_rows(medians::DataFrame)
    sequential = select(
        filter(:family => ==("sequential"), medians),
        :canonical_matrix,
        :k,
        :seconds_per_sweep => :sequential_seconds,
    )
    parallel = filter(
        row -> row.family != "sequential" && row.threads in TIMING_THREADS,
        medians,
    )
    parallel = innerjoin(
        parallel,
        sequential,
        on = [:canonical_matrix, :k],
        validate = (false, true),
    )
    parallel.speedup = parallel.sequential_seconds ./ parallel.seconds_per_sweep
    return parallel
end

function method_relative_speedup_rows(medians::DataFrame)
    iterative = filter(
        row -> row.family != "sequential" && row.threads in STRONG_SCALING_THREADS,
        medians,
    )
    one_thread = select(
        filter(row -> row.family != "sequential" && row.threads == 1, medians),
        :canonical_matrix,
        :k,
        :family,
        :seconds_per_sweep => :one_thread_seconds,
    )
    iterative = innerjoin(
        iterative,
        one_thread,
        on = [:canonical_matrix, :k, :family],
        validate = (false, true),
    )
    iterative.speedup = iterative.one_thread_seconds ./ iterative.seconds_per_sweep
    return iterative
end

function load_data()
    pairs = Dict{String,Set{Tuple{String,Int}}}()
    speedup_frames = DataFrame[]
    method_relative_frames = DataFrame[]
    for specification in DATASETS
        values, fixed = timing_medians(specification)
        pairs[specification.name] = fixed
        push!(speedup_frames, speedup_rows(values))
        push!(method_relative_frames, method_relative_speedup_rows(values))
    end

    panels = [
        (
            title = "sync. ILU",
            dataset = "unsymmetric",
            mode = "sync",
            pairs = length(pairs["unsymmetric"]),
            speedups = speedup_frames[1],
            ats_family = "ats_sync",
            par_family = "par_sync",
        ),
        (
            title = "async. ILU",
            dataset = "unsymmetric",
            mode = "async",
            pairs = length(pairs["unsymmetric"]),
            speedups = speedup_frames[1],
            ats_family = "ats_async",
            par_family = "par_async",
        ),
        (
            title = "IC",
            dataset = "spd",
            mode = "async",
            pairs = length(pairs["spd"]),
            speedups = speedup_frames[2],
            ats_family = "ats_async",
            par_family = "par_async",
        ),
    ]
    return (
        panels,
        vcat(speedup_frames...; cols = :union),
        vcat(method_relative_frames...; cols = :union),
    )
end

function survival_coordinates(values)
    finite_values = sort(Float64[value for value in values if isfinite(value)])
    denominator = length(values)
    current = count(value -> value >= SPEEDUP_MIN, finite_values)
    xs = Float64[SPEEDUP_MIN]
    ys = Float64[current / denominator]
    for value in unique(finite_values)
        SPEEDUP_MIN <= value <= SPEEDUP_MAX || continue
        push!(xs, value)
        push!(ys, current / denominator)
        current -= count(==(value), finite_values)
        push!(xs, value)
        push!(ys, current / denominator)
    end
    push!(xs, SPEEDUP_MAX)
    push!(ys, current / denominator)
    return xs, ys
end

function speedup_axis(parent; title = "", x_ticklabels, y_ticklabels)
    yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    axis = Axis(
        parent,
        title = title,
        xscale = log2,
        xreversed = true,
        xticks = (SPEEDUP_TICKS, x_ticklabels),
        yticks = (yticks, y_ticklabels),
        xgridvisible = true,
        ygridvisible = true,
    )
    xlims!(axis, SPEEDUP_MIN, SPEEDUP_MAX)
    axis.xreversed = true
    ylims!(axis, -0.005, 1.005)
    vlines!(
        axis,
        [1.0],
        color = SEQUENTIAL_COLOR,
        linestyle = :dot,
        linewidth = 0.9,
    )
    hidespines!(axis)
    axis.xticksvisible = false
    axis.yticksvisible = false
    return axis
end

function draw_speedups!(axis, panel, family::String, colors)
    for (index, threads) in enumerate(TIMING_THREADS)
        selected = filter(
            row -> row.family == family && row.threads == threads,
            panel.speedups,
        )
        nrow(selected) == panel.pairs || error(
            "$(panel.title), $family, $threads threads: " *
            "$(nrow(selected)) rows; expected $(panel.pairs)",
        )
        xs, ys = survival_coordinates(selected.speedup)
        lines!(axis, xs, ys; color = colors[index], linewidth = PLOT_LINE_WIDTH)
    end
end

function panel_x_ticklabels(row::Int, column::Int)
    row == 2 || return fill("", length(SPEEDUP_TICK_LABELS))
    labels = copy(SPEEDUP_TICK_LABELS)
    column == 1 && (labels[1] = "")
    column == 2 && (labels[1] = labels[end] = "")
    column == 3 && (labels[end] = "")
    return labels
end

function panel_y_ticklabels(row::Int, column::Int)
    column == 1 || return fill("", 5)
    labels = ["0%", "25%", "50%", "75%", "100%"]
    row == 2 && (labels[1] = labels[end] = "")
    return labels
end

function draw_thread_key!(axis, palette, method_label::String)
    log_span = log(SPEEDUP_MAX) - log(SPEEDUP_MIN)
    data_x(relative_x) = exp(log(SPEEDUP_MAX) - relative_x * log_span)
    relative_edges = collect(range(0.05, 0.31, length = length(TIMING_THREADS) + 1))
    relative_centers = (relative_edges[1:end-1] .+ relative_edges[2:end]) ./ 2
    x_edges = data_x.(relative_edges)
    x_centers = data_x.(relative_centers)
    y_bottom, y_top = 0.650, 0.700
    tick_bottom, label_baseline, thread_baseline = 0.630, 0.580, 0.523

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
        0.18,
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
            text = string(TIMING_THREADS[index]),
            align = (:center, :baseline),
            fontsize = TICK_LABEL_SIZE,
        )
    end
    text!(
        axis,
        0.18,
        thread_baseline,
        text = "threads",
        space = :relative,
        align = (:center, :baseline),
        fontsize = TICK_LABEL_SIZE,
    )
end

function build_figure(panels)
    apply_paper_theme!()
    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 213, padding = 1)
    panel_width = paper_length(149, 530)
    panel_height = 91
    panel_gap = paper_length(8, 530)
    for (column, panel) in enumerate(panels)
        ats_axis = speedup_axis(
            figure[1, column];
            title = "",
            x_ticklabels = panel_x_ticklabels(1, column),
            y_ticklabels = panel_y_ticklabels(1, column),
        )
        par_axis = speedup_axis(
            figure[2, column];
            x_ticklabels = panel_x_ticklabels(2, column),
            y_ticklabels = panel_y_ticklabels(2, column),
        )
        draw_speedups!(ats_axis, panel, panel.ats_family, ATS_TIMING_COLORS)
        draw_speedups!(par_axis, panel, panel.par_family, PAR_TIMING_COLORS)
        text!(
            ats_axis,
            0.04,
            0.816,
            text = panel.title,
            space = :relative,
            align = (:left, :bottom),
            fontsize = TITLE_SIZE,
            font = :bold,
        )
        column == 2 && draw_thread_key!(ats_axis, ATS_TIMING_COLORS, "ATS")
        column == 2 && draw_thread_key!(par_axis, PAR_TIMING_COLORS, "Par")
    end

    Label(
        figure[1:2, 0],
        "Fraction of cases attaining at least this speedup",
        rotation = pi / 2,
        fontsize = AXIS_LABEL_SIZE,
        tellheight = false,
    )
    Label(
        figure[3, 1:3],
        "Per-sweep speedup relative to sequential ILU/IC",
        fontsize = AXIS_LABEL_SIZE,
    )

    for column in 1:3
        colsize!(figure.layout, column, Fixed(panel_width))
    end
    rowsize!(figure.layout, 1, Fixed(panel_height))
    rowsize!(figure.layout, 2, Fixed(panel_height))
    colgap!(figure.layout, panel_gap)
    rowgap!(figure.layout, 1, panel_gap)
    rowgap!(figure.layout, 2, 1)
    return figure
end

function quantile_columns(values)
    quantiles = quantile(Float64.(values), [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return (
        p01 = quantiles[1],
        p05 = quantiles[2],
        p25 = quantiles[3],
        p50 = quantiles[4],
        p75 = quantiles[5],
        p95 = quantiles[6],
        p99 = quantiles[7],
    )
end

function summarize_speedups(rows::DataFrame)
    output = NamedTuple[]
    for group in groupby(rows, [:dataset, :family, :threads])
        statistics = quantile_columns(group.speedup)
        push!(output, (
            dataset = String(first(group.dataset)),
            family = String(first(group.family)),
            threads = Int(first(group.threads)),
            pairs = nrow(group),
            statistics...,
        ))
    end
    return DataFrame(output)
end

function traditional_axis(
    parent;
    title::String,
    threads,
    speedup_ticks,
    x_ticklabels,
    y_ticklabels,
)
    axis = Axis(
        parent,
        xscale = log2,
        yscale = log2,
        xticks = (Float64.(threads), x_ticklabels),
        yticks = (Float64.(speedup_ticks), y_ticklabels),
        xgridvisible = true,
        ygridvisible = true,
    )
    hidespines!(axis)
    axis.xticksvisible = false
    axis.yticksvisible = false
    xlims!(axis, first(threads), last(threads))
    ylims!(axis, first(speedup_ticks), last(speedup_ticks))
    return axis
end

function draw_quantile_trajectory!(axis, summary, family::String, color)
    rows = filter(:family => ==(family), summary)
    sort!(rows, :threads)
    band!(
        axis,
        Float64.(rows.threads),
        rows.p05,
        rows.p95;
        color = (color, 0.065),
    )
    band!(
        axis,
        Float64.(rows.threads),
        rows.p25,
        rows.p75;
        color = (color, 0.16),
    )
    lines!(
        axis,
        Float64.(rows.threads),
        rows.p50;
        color = color,
        linewidth = 1.35,
    )
    scatter!(
        axis,
        Float64.(rows.threads),
        rows.p50;
        color = color,
        markersize = 4.0,
        strokewidth = 0,
    )
end

function build_traditional_figure(
    panels,
    summary;
    threads = TIMING_THREADS,
    speedup_ticks = [1, 2, 4, 8, 16, 32, 64],
    reference_label = "thread-proportional",
)
    apply_paper_theme!()
    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 148, padding = 1)
    panel_width = paper_length(150, 530)
    panel_height = 116
    panel_gap = paper_length(8, 530)
    axes = Any[]
    thread_labels = string.(threads)
    speedup_labels = string.(speedup_ticks)

    for (column, panel) in enumerate(panels)
        panel_summary = filter(:dataset => ==(panel.dataset), summary)
        panel_thread_labels = copy(thread_labels)
        column < length(panels) && (panel_thread_labels[end] = "")
        column > 1 && (panel_thread_labels[1] = "")
        axis = traditional_axis(
            figure[1, column];
            title = panel.title,
            threads = threads,
            speedup_ticks = speedup_ticks,
            x_ticklabels = panel_thread_labels,
            y_ticklabels = column == 1 ? speedup_labels : fill("", length(speedup_labels)),
        )
        lines!(
            axis,
            [Float64(first(threads)), Float64(min(last(threads), last(speedup_ticks)))],
            [Float64(first(threads)), Float64(min(last(threads), last(speedup_ticks)))];
            color = RGBAf(SEQUENTIAL_COLOR, 0.65),
            linewidth = 0.9,
            linestyle = :dot,
        )
        draw_quantile_trajectory!(axis, panel_summary, panel.ats_family, ATS_COLOR)
        draw_quantile_trajectory!(axis, panel_summary, panel.par_family, PAR_COLOR)
        text!(
            axis,
            0.5,
            0.96,
            text = panel.title,
            space = :relative,
            align = (:center, :top),
            fontsize = TITLE_SIZE,
            font = TITLE_FONT,
        )
        push!(axes, axis)
    end

    axislegend(
        axes[2],
        [
            LineElement(color = ATS_COLOR, linewidth = 1.35, marker = :circle,
                        markercolor = ATS_COLOR, markersize = 4.0),
            LineElement(color = PAR_COLOR, linewidth = 1.35, marker = :circle,
                        markercolor = PAR_COLOR, markersize = 4.0),
            LineElement(color = RGBAf(SEQUENTIAL_COLOR, 0.65), linewidth = 0.9,
                        linestyle = :dot),
        ],
        ["ATS median", "Par median", reference_label];
        position = :lt,
        margin = (6, 6, 6, 10),
        backgroundcolor = (:white, 0.86),
        labelsize = TICK_LABEL_SIZE,
        patchsize = (16, 7),
        rowgap = 1,
    )

    Label(
        figure[1, 0],
        "Per-sweep speedup",
        rotation = pi / 2,
        fontsize = AXIS_LABEL_SIZE,
        tellheight = false,
    )
    Label(
        figure[2, 1:3],
        "OpenMP threads",
        fontsize = AXIS_LABEL_SIZE,
    )

    for column in 1:3
        colsize!(figure.layout, column, Fixed(panel_width))
    end
    rowsize!(figure.layout, 1, Fixed(panel_height))
    colgap!(figure.layout, panel_gap)
    rowgap!(figure.layout, 1, 1)
    return figure
end

function main()
    panels, speedups, method_relative_speedups = load_data()
    all(isfinite, speedups.speedup) && all(>(0), speedups.speedup) ||
        error("speedup values must be finite and positive")
    CSV.write(joinpath(OUTPUT_DIR, "speedup_values.csv"), speedups)
    summary = summarize_speedups(speedups)
    CSV.write(joinpath(OUTPUT_DIR, "speedup_quantiles.csv"), summary)

    println("fixed cohorts: unsymmetric=$(panels[1].pairs), SPD=$(panels[3].pairs)")

    all(isfinite, method_relative_speedups.speedup) &&
        all(>(0), method_relative_speedups.speedup) ||
        error("method-relative speedup values must be finite and positive")
    method_relative_summary = summarize_speedups(method_relative_speedups)
    CSV.write(
        joinpath(OUTPUT_DIR, "method_relative_speedup_quantiles.csv"),
        method_relative_summary,
    )
    method_relative_figure = build_traditional_figure(
        panels,
        method_relative_summary;
        threads = STRONG_SCALING_THREADS,
        speedup_ticks = [1, 2, 4, 8, 16, 32, 64, 128],
        reference_label = "ideal scaling",
    )
    method_relative_outputs = save_publication_figure(
        joinpath(OUTPUT_DIR, "strong_scaling_method_relative_median_iqr"),
        method_relative_figure,
    )
    println("wrote:")
    foreach(path -> println("  ", path), method_relative_outputs)
end

main()
