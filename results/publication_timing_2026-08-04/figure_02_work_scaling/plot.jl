#!/usr/bin/env julia

using CSV
using CairoMakie
using DataFrames
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "plotting", "ATSILUPaperStyle.jl"))
using .ATSILUPaperStyle

const OUTPUT_DIR = @__DIR__
const THREAD_COUNT = 64
const X_LIMITS = (3.5, 520.0)
const Y_LIMITS = (1.4, 76.0)
const X_TICKS = [4.0, 16.0, 64.0, 256.0]
const X_TICK_LABELS = ["4", "16", "64", "256"]
const Y_TICKS = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
const Y_TICK_LABELS = ["2", "4", "8", "16", "32", "64"]

const SPEEDUP_PATH = joinpath(
    ROOT,
    "results",
    "publication_timing_2026-08-04",
    "figure_01_scaling",
    "speedup_values.csv",
)

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
    ),
]

function pair_metadata(specification, keys)
    rows = CSV.read(specification.timing_path, DataFrame; stringtype = String)
    rows = filter(
        row -> (String(row.canonical_matrix), Int(row.k)) in keys,
        rows,
    )

    output = NamedTuple[]
    for group in groupby(rows, [:canonical_matrix, :k])
        for column in (:n, :nnz, :S_nnz)
            length(unique(group[!, column])) == 1 || error(
                "$(first(group.canonical_matrix)), k=$(first(group.k)): " *
                "$column is not constant",
            )
        end
        push!(output, (
            dataset = specification.name,
            canonical_matrix = String(first(group.canonical_matrix)),
            k = Int(first(group.k)),
            n = Int(first(group.n)),
            nnz = Int(first(group.nnz)),
            S_nnz = Int(first(group.S_nnz)),
        ))
    end
    length(output) == length(keys) || error(
        "$(specification.name): found $(length(output)) metadata rows; " *
        "expected $(length(keys))",
    )
    return DataFrame(output)
end

function load_data()
    speedups = CSV.read(SPEEDUP_PATH, DataFrame; stringtype = String)
    speedups = filter(:threads => ==(THREAD_COUNT), speedups)

    metadata = DataFrame[]
    for specification in DATASETS
        selected = filter(:dataset => ==(specification.name), speedups)
        keys = Set(
            (String(row.canonical_matrix), Int(row.k))
            for row in eachrow(selected)
        )
        push!(metadata, pair_metadata(specification, keys))
    end

    values = innerjoin(
        speedups,
        vcat(metadata...; cols = :union),
        on = [:dataset, :canonical_matrix, :k],
        validate = (false, true),
    )
    values.pattern_width = values.S_nnz ./ values.n
    values.method = ifelse.(startswith.(values.family, "ats"), "ATS", "Par")

    all(isfinite, values.pattern_width) && all(>(0), values.pattern_width) ||
        error("pattern widths must be finite and positive")
    all(isfinite, values.speedup) && all(>(0), values.speedup) ||
        error("speedups must be finite and positive")
    return values
end

function panel_specs(values)
    return [
        (
            title = "sync. ILU",
            pairs = 52,
            rows = filter(
                row -> row.dataset == "unsymmetric" &&
                       row.family in ("ats_sync", "par_sync"),
                values,
            ),
        ),
        (
            title = "async. ILU",
            pairs = 52,
            rows = filter(
                row -> row.dataset == "unsymmetric" &&
                       row.family in ("ats_async", "par_async"),
                values,
            ),
        ),
        (
            title = "IC",
            pairs = 43,
            rows = filter(
                row -> row.dataset == "spd" &&
                       row.family in ("ats_async", "par_async"),
                values,
            ),
        ),
    ]
end

function regression(rows)
    log_width = log2.(Float64.(rows.pattern_width))
    log_speedup = log2.(Float64.(rows.speedup))
    coefficients = hcat(ones(length(log_width)), log_width) \ log_speedup
    return (
        intercept = coefficients[1],
        slope = coefficients[2],
        correlation = cor(log_width, log_speedup),
    )
end

function metric_correlations(panels)
    metrics = [
        (label = "dimension", column = :n),
        (label = "matrix_nonzeros", column = :nnz),
        (label = "pattern_size", column = :S_nnz),
        (label = "pattern_width", column = :pattern_width),
    ]
    output = NamedTuple[]
    for panel in panels
        for method in ("ATS", "Par")
            rows = filter(:method => ==(method), panel.rows)
            log_speedup = log2.(Float64.(rows.speedup))
            for metric in metrics
                log_metric = log2.(Float64.(rows[!, metric.column]))
                push!(output, (
                    panel = panel.title,
                    method = method,
                    pairs = nrow(rows),
                    metric = metric.label,
                    correlation = cor(log_metric, log_speedup),
                ))
            end
        end
    end
    return DataFrame(output)
end

function work_axis(parent; show_ylabels::Bool)
    axis = Axis(
        parent,
        xscale = log2,
        yscale = log2,
        xticks = (X_TICKS, X_TICK_LABELS),
        yticks = (
            Y_TICKS,
            show_ylabels ? Y_TICK_LABELS : fill("", length(Y_TICKS)),
        ),
        xgridvisible = true,
        ygridvisible = true,
    )
    xlims!(axis, X_LIMITS...)
    ylims!(axis, Y_LIMITS...)
    hidespines!(axis)
    axis.xticksvisible = false
    axis.yticksvisible = false
    return axis
end

function draw_panel!(axis, panel, correlations)
    for method in ("ATS", "Par")
        rows = filter(:method => ==(method), panel.rows)
        nrow(rows) == panel.pairs || error(
            "$(panel.title), $method: $(nrow(rows)) rows; expected $(panel.pairs)",
        )

        color = method == "ATS" ? ATS_COLOR : PAR_COLOR
        scatter!(
            axis,
            rows.pattern_width,
            rows.speedup,
            color = (color, 0.62),
            marker = :circle,
            markersize = 4.2,
            strokewidth = 0,
        )

        fit = regression(rows)
        line_widths = exp2.(range(
            log2(minimum(rows.pattern_width)),
            log2(maximum(rows.pattern_width)),
            length = 160,
        ))
        line_speedups = exp2.(fit.intercept .+ fit.slope .* log2.(line_widths))
        lines!(
            axis,
            line_widths,
            line_speedups,
            color = color,
            linewidth = PLOT_LINE_WIDTH,
        )
        text!(
            axis,
            0.05,
            method == "ATS" ? 0.81 : 0.72,
            text = "r = $(round(fit.correlation, digits = 2))",
            space = :relative,
            align = (:left, :top),
            color = color,
            fontsize = TICK_LABEL_SIZE,
        )

        push!(correlations, (
            panel = panel.title,
            method = method,
            pairs = nrow(rows),
            correlation = fit.correlation,
            slope = fit.slope,
            intercept = fit.intercept,
        ))
    end

    text!(
        axis,
        0.04,
        0.96,
        text = panel.title,
        space = :relative,
        align = (:left, :top),
        fontsize = TITLE_SIZE,
        font = TITLE_FONT,
    )
end

function build_figure(panels)
    apply_paper_theme!()
    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 123, padding = 1)
    panel_width = 108
    panel_height = 101
    panel_gap = paper_length(8, 530)
    correlations = NamedTuple[]
    axes = Axis[]

    for (column, panel) in enumerate(panels)
        axis = work_axis(figure[1, column]; show_ylabels = column == 1)
        draw_panel!(axis, panel, correlations)
        push!(axes, axis)
        colsize!(figure.layout, column, Fixed(panel_width))
    end

    Label(
        figure[1, 0],
        "64-thread speedup per sweep",
        rotation = pi / 2,
        fontsize = AXIS_LABEL_SIZE,
        tellheight = false,
    )
    axislegend(
        axes[2],
        [
            LineElement(color = ATS_COLOR, linewidth = PLOT_LINE_WIDTH),
            LineElement(color = PAR_COLOR, linewidth = PLOT_LINE_WIDTH),
        ],
        ["ATS", "Par"],
        position = :rb,
        orientation = :horizontal,
        nbanks = 2,
        framevisible = false,
        backgroundcolor = (:white, 0.82),
        labelsize = TICK_LABEL_SIZE,
        patchsize = (14, 6),
        colgap = 6,
    )
    Label(
        figure[2, 1:3],
        "Average factor-pattern width",
        fontsize = AXIS_LABEL_SIZE,
    )

    rowsize!(figure.layout, 1, Fixed(panel_height))
    rowgap!(figure.layout, 1, 0)
    colgap!(figure.layout, panel_gap)
    return figure, DataFrame(correlations)
end

function main()
    values = load_data()
    panels = panel_specs(values)
    figure, correlations = build_figure(panels)
    all_metric_correlations = metric_correlations(panels)

    CSV.write(
        joinpath(OUTPUT_DIR, "work_scaling_values.csv"),
        select(
            values,
            :dataset,
            :canonical_matrix,
            :k,
            :family,
            :method,
            :n,
            :nnz,
            :S_nnz,
            :pattern_width,
            :speedup,
        ),
    )
    CSV.write(joinpath(OUTPUT_DIR, "work_scaling_correlations.csv"), correlations)
    CSV.write(
        joinpath(OUTPUT_DIR, "work_scaling_metric_correlations.csv"),
        all_metric_correlations,
    )

    outputs = save_publication_figure(
        joinpath(OUTPUT_DIR, "speedup_vs_pattern_width_64_threads"),
        figure,
    )
    println("wrote:")
    foreach(path -> println("  ", path), outputs)
    println(correlations)
end

main()
