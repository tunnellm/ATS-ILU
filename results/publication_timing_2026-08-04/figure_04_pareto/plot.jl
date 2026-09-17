#!/usr/bin/env julia

using CSV
using CairoMakie
using DataFrames
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
include(joinpath(ROOT, "plotting", "ATSILUPaperStyle.jl"))
using .ATSILUPaperStyle

const OUTPUT_DIR = @__DIR__
const SWEEPS = [1, 2, 3]

const PANELS = [
    (
        name = "async_ilu",
        row_label = "async. ILU",
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
        quality_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-07-31",
            "unsymmetric_gmres",
            "combined_solver_outcomes.csv",
        ),
        quality_pair_column = :matrix,
        ats_method = "ats_ilu",
        par_method = "parilu",
        ats_variant = "async",
        par_variant = "async",
        ats_family = "ats_async",
        par_family = "par_async",
        quality_threads_from_timing = true,
        repeats = 3,
    ),
    (
        name = "sync_ilu",
        row_label = "sync. ILU",
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
        quality_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-07-31",
            "unsymmetric_gmres",
            "combined_solver_outcomes.csv",
        ),
        quality_pair_column = :matrix,
        ats_method = "ats_ilu",
        par_method = "parilu",
        ats_variant = "sync_folded",
        par_variant = "sync",
        ats_family = "ats_sync",
        par_family = "par_sync",
        quality_threads_from_timing = false,
        repeats = 1,
    ),
    (
        name = "async_ic",
        row_label = "IC",
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
        quality_path = joinpath(
            ROOT,
            "results",
            "analysis_2026-08-02",
            "spd_cg",
            "combined_solver_outcomes.csv",
        ),
        quality_pair_column = :matrix_id,
        ats_method = "ats_ic",
        par_method = "par_ic",
        ats_variant = "async",
        par_variant = "async",
        ats_family = "ats_async",
        par_family = "par_async",
        quality_threads_from_timing = true,
        repeats = 3,
    ),
]

const POINT_COLOR = colorant"#71468F"
const X_LIMITS = (0.25, 2.0)
const Y_LIMITS = (0.25, 4.0)

const ATS_REGION = RGBAf(ATS_COLOR, 0.055)
const PAR_REGION = RGBAf(PAR_COLOR, 0.055)
const REFERENCE_COLOR = RGBAf(SEQUENTIAL_COLOR, 0.82)

geomean(values) = exp(mean(log.(values)))
pair_key(row) = (String(row.canonical_matrix), Int(row.k))

function fixed_pairs(path::String)
    coverage = CSV.read(path, DataFrame; stringtype = String)
    return Set(
        (String(row.matrix), Int(row.k))
        for row in eachrow(filter(:fully_complete => ==(1), coverage))
    )
end

function add_quality_identity!(quality::DataFrame, timing::DataFrame, specification)
    if specification.quality_pair_column == :matrix_id
        quality.canonical_matrix = String.(quality.matrix_id)
        return quality
    end

    canonical_by_name = Dict{String,String}()
    for row in eachrow(timing)
        matrix_name = first(splitext(String(row.matrix)))
        canonical = String(row.canonical_matrix)
        if haskey(canonical_by_name, matrix_name) &&
           canonical_by_name[matrix_name] != canonical
            error("ambiguous matrix name: $matrix_name")
        end
        canonical_by_name[matrix_name] = canonical
    end
    quality.canonical_matrix = [
        get(canonical_by_name, String(matrix), "")
        for matrix in quality[!, specification.quality_pair_column]
    ]
    any(isempty, quality.canonical_matrix) &&
        error("quality matrix is missing a timing identity")
    return quality
end

function timing_summary(timing::DataFrame, cohort, specification)
    selected = filter(
        row -> pair_key(row) in cohort && row.threads in THREADS &&
               row.family in (specification.ats_family, specification.par_family),
        timing,
    )
    summary = combine(
        groupby(selected, [:canonical_matrix, :k, :family, :threads]),
        :seconds_per_sweep => median => :seconds_per_sweep,
        :t_algorithm_setup => median => :setup_seconds,
    )
    expected = length(cohort) * length(THREADS) * 2
    nrow(summary) == expected || error(
        "$(specification.name) timing summary has $(nrow(summary)) rows; " *
        "expected $expected",
    )
    return summary
end

function quality_is_valid(row)
    solver_ok = hasproperty(row, :solver_status) ?
                coalesce(row.solver_status == "converged", false) :
                coalesce(row.converged == 1, false)
    factor_ok = hasproperty(row, :factor_status) ?
                coalesce(row.factor_status == "ok", false) : true
    iteration_ok = !ismissing(row.iterations) && isfinite(row.iterations) &&
                   row.iterations > 0
    return solver_ok && factor_ok && iteration_ok
end

function summarize_quality_group(group::DataFrame, expected_repeats::Int)
    repeat_ids = sort(unique(Int.(group.repeat)))
    complete = nrow(group) == expected_repeats &&
               repeat_ids == collect(1:expected_repeats)
    valid = complete && all(quality_is_valid(row) for row in eachrow(group))
    iterations = valid ? geomean(Float64.(group.iterations)) : missing
    return valid, iterations
end

function quality_summary(quality::DataFrame, cohort, specification)
    selected = filter(
        row -> pair_key(row) in cohort && row.sweeps in SWEEPS &&
               ((row.method == specification.ats_method &&
                 row.variant == specification.ats_variant) ||
                (row.method == specification.par_method &&
                 row.variant == specification.par_variant)),
        quality,
    )

    output = NamedTuple[]
    quality_threads = specification.quality_threads_from_timing ? THREADS : [0]
    for (canonical_matrix, k) in sort(collect(cohort))
        for method in (specification.ats_method, specification.par_method)
            for sweeps in SWEEPS, threads in quality_threads
                group = filter(
                    row -> row.canonical_matrix == canonical_matrix && row.k == k &&
                           row.method == method && row.sweeps == sweeps &&
                           row.threads == threads,
                    selected,
                )
                valid, iterations = summarize_quality_group(
                    group,
                    specification.repeats,
                )
                push!(output, (
                    canonical_matrix = canonical_matrix,
                    k = k,
                    method = method,
                    sweeps = sweeps,
                    threads = threads,
                    valid = valid,
                    iterations = iterations,
                ))
            end
        end
    end
    return DataFrame(output)
end

function method_quality(summary::DataFrame, method::String, prefix::String)
    selected = filter(:method => ==(method), summary)
    return select(
        selected,
        :canonical_matrix,
        :k,
        :sweeps,
        :threads,
        :valid => Symbol(prefix, "_valid"),
        :iterations => Symbol(prefix, "_iterations"),
    )
end

function method_timing(summary::DataFrame, family::String, prefix::String)
    selected = filter(:family => ==(family), summary)
    return select(
        selected,
        :canonical_matrix,
        :k,
        :threads,
        :seconds_per_sweep => Symbol(prefix, "_seconds_per_sweep"),
        :setup_seconds => Symbol(prefix, "_setup_seconds"),
    )
end

function classify(ats_valid::Bool, par_valid::Bool)
    ats_valid && par_valid && return "paired finite"
    ats_valid && return "ATS only"
    par_valid && return "Par only"
    return "neither"
end

function panel_rows(specification)
    cohort = fixed_pairs(specification.coverage_path)
    timing = CSV.read(specification.timing_path, DataFrame; stringtype = String)
    quality = CSV.read(specification.quality_path, DataFrame; stringtype = String)
    add_quality_identity!(quality, timing, specification)

    times = timing_summary(timing, cohort, specification)
    quality_values = quality_summary(quality, cohort, specification)
    ats_times = method_timing(times, specification.ats_family, "ats")
    par_times = method_timing(times, specification.par_family, "par")
    ats_quality = method_quality(quality_values, specification.ats_method, "ats")
    par_quality = method_quality(quality_values, specification.par_method, "par")

    if !specification.quality_threads_from_timing
        select!(ats_quality, Not(:threads))
        select!(par_quality, Not(:threads))
    end
    quality_join_keys = specification.quality_threads_from_timing ?
                        [:canonical_matrix, :k, :sweeps, :threads] :
                        [:canonical_matrix, :k, :sweeps]
    rows = innerjoin(ats_quality, par_quality; on = quality_join_keys)
    if !specification.quality_threads_from_timing
        rows = crossjoin(rows, DataFrame(threads = THREADS))
    end
    rows = innerjoin(rows, ats_times; on = [:canonical_matrix, :k, :threads])
    rows = innerjoin(rows, par_times; on = [:canonical_matrix, :k, :threads])

    rows.ats_construction_seconds = rows.ats_setup_seconds .+
                                    rows.sweeps .* rows.ats_seconds_per_sweep
    rows.par_construction_seconds = rows.par_setup_seconds .+
                                    rows.sweeps .* rows.par_seconds_per_sweep
    rows.construction_ratio = rows.ats_construction_seconds ./
                              rows.par_construction_seconds
    rows.iteration_ratio = [
        ats_valid && par_valid ? ats_iterations / par_iterations : missing
        for (ats_valid, par_valid, ats_iterations, par_iterations) in zip(
            rows.ats_valid,
            rows.par_valid,
            rows.ats_iterations,
            rows.par_iterations,
        )
    ]
    rows.outcome = classify.(rows.ats_valid, rows.par_valid)
    rows.panel .= specification.name
    rows.row_label .= specification.row_label
    rows.cohort_pairs .= length(cohort)
    return rows
end

function paired_counts(rows::DataFrame)
    counts = combine(
        groupby(rows, [:panel, :row_label, :sweeps, :threads, :cohort_pairs]),
        :outcome => (values -> count(==("paired finite"), values)) => :paired_pairs,
        :outcome => (values -> count(==("ATS only"), values)) => :ats_only,
        :outcome => (values -> count(==("Par only"), values)) => :par_only,
        :outcome => (values -> count(==("neither"), values)) => :neither,
    )
    return sort(counts, [:panel, :sweeps, :threads])
end

function summarize_pareto(rows::DataFrame)
    output = NamedTuple[]
    for group in groupby(rows, [:panel, :row_label, :sweeps, :cohort_pairs])
        finite = filter(:outcome => ==("paired finite"), group)
        construction = Float64.(finite.construction_ratio)
        iterations = Float64.(finite.iteration_ratio)
        ats_dominates = (construction .<= 1.0) .& (iterations .<= 1.0) .&
                        ((construction .< 1.0) .| (iterations .< 1.0))
        push!(output, (
            panel = String(first(group.panel)),
            row_label = String(first(group.row_label)),
            sweeps = Int(first(group.sweeps)),
            cohort_pairs = Int(first(group.cohort_pairs)),
            total_thread_configurations = nrow(group),
            paired_configurations = nrow(finite),
            ats_only_configurations = count(==("ATS only"), group.outcome),
            par_only_configurations = count(==("Par only"), group.outcome),
            neither_configurations = count(==("neither"), group.outcome),
            construction_geomean = geomean(construction),
            construction_p25 = quantile(construction, 0.25),
            construction_median = median(construction),
            construction_p75 = quantile(construction, 0.75),
            fraction_ats_faster = mean(construction .< 1.0),
            iteration_geomean = geomean(iterations),
            iteration_p25 = quantile(iterations, 0.25),
            iteration_median = median(iterations),
            iteration_p75 = quantile(iterations, 0.75),
            fraction_ats_dominates = mean(ats_dominates),
        ))
    end
    return DataFrame(output)
end

function power_limits(values)
    finite = Float64[value for value in values if !ismissing(value) && isfinite(value) && value > 0]
    isempty(finite) && error("cannot determine limits from an empty collection")
    lower = 2.0^floor(log2(minimum(finite)))
    upper = 2.0^ceil(log2(maximum(finite)))
    return lower, upper
end

function power_ticks(lower::Real, upper::Real)
    exponents = Int(round(log2(lower))):Int(round(log2(upper)))
    values = 2.0 .^ collect(exponents)
    labels = [
        value < 1 ? string(round(value; digits = 4)) : string(Int(round(value)))
        for value in values
    ]
    return values, labels
end

function shaded_quadrants!(axis, xlimits, ylimits)
    poly!(
        axis,
        Rect2f(xlimits[1], ylimits[1], 1.0 - xlimits[1], 1.0 - ylimits[1]),
        color = ATS_REGION,
        strokewidth = 0,
    )
    poly!(
        axis,
        Rect2f(1.0, 1.0, xlimits[2] - 1.0, ylimits[2] - 1.0),
        color = PAR_REGION,
        strokewidth = 0,
    )
end

function panel_ticklabels(labels::Vector{String}, row::Int, column::Int, axis::Symbol)
    output = fill("", length(labels))
    if axis == :x && row == length(PANELS)
        output .= labels
        column == 1 && (output[end] = "")
        column == 2 && (output[1] = output[end] = "")
        column == 3 && (output[1] = "")
    elseif axis == :y && column == 1
        output .= labels
        row > 1 && (output[end] = "")
        row < length(PANELS) && (output[1] = "")
    end
    return output
end

function build_figure(rows::DataFrame)
    apply_paper_theme!()
    xlimits = X_LIMITS
    ylimits = Y_LIMITS
    xvalues, xlabels = power_ticks(xlimits...)
    yvalues, ylabels = power_ticks(ylimits...)

    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 277, padding = 1)
    panel_width = paper_length(145, 530)
    panel_height = 76
    panel_gap = paper_length(8, 530)

    for (row_index, specification) in enumerate(PANELS)
        for (column, sweeps) in enumerate(SWEEPS)
            axis = Axis(
                figure[row_index, column],
                xscale = log2,
                yscale = log2,
                xticks = (
                    xvalues,
                    panel_ticklabels(xlabels, row_index, column, :x),
                ),
                yticks = (
                    yvalues,
                    panel_ticklabels(ylabels, row_index, column, :y),
                ),
                xgridvisible = true,
                ygridvisible = true,
            )
            shaded_quadrants!(axis, xlimits, ylimits)
            vlines!(
                axis,
                [1.0],
                color = REFERENCE_COLOR,
                linestyle = :dot,
                linewidth = 0.9,
            )
            hlines!(
                axis,
                [1.0],
                color = REFERENCE_COLOR,
                linestyle = :dot,
                linewidth = 0.9,
            )

            panel = filter(
                row -> row.panel == specification.name && row.sweeps == sweeps,
                rows,
            )
            points = filter(:outcome => ==("paired finite"), panel)
            scatter!(
                axis,
                points.construction_ratio,
                collect(skipmissing(points.iteration_ratio));
                color = POINT_COLOR,
                markersize = 2.7,
                strokewidth = 0,
            )
            column == 1 && text!(
                axis,
                0.035,
                0.965,
                text = specification.row_label,
                space = :relative,
                align = (:left, :top),
                fontsize = TITLE_SIZE,
                font = TITLE_FONT,
            )
            if row_index == 1
                axis.title = sweeps == 1 ? "1 sweep" : "$sweeps sweeps"
                axis.titlesize = TITLE_SIZE
                axis.titlefont = TITLE_FONT
            end
            hidespines!(axis)
            axis.xticksvisible = false
            axis.yticksvisible = false
            xlims!(axis, xlimits...)
            ylims!(axis, ylimits...)
        end
    end

    Label(
        figure[1:3, 0],
        "ATS / Par Krylov iterations",
        rotation = pi / 2,
        fontsize = AXIS_LABEL_SIZE,
        tellheight = false,
    )
    Label(
        figure[4, 1:3],
        "ATS / Par method-specific construction time",
        fontsize = AXIS_LABEL_SIZE,
    )

    for column in 1:3
        colsize!(figure.layout, column, Fixed(panel_width))
    end
    for row in 1:3
        rowsize!(figure.layout, row, Fixed(panel_height))
    end
    colgap!(figure.layout, 1, panel_gap)
    colgap!(figure.layout, 2, panel_gap)
    rowgap!(figure.layout, 1, panel_gap)
    rowgap!(figure.layout, 2, panel_gap)
    rowgap!(figure.layout, 3, 1)
    return figure, xlimits, ylimits
end

function main()
    frames = [panel_rows(specification) for specification in PANELS]
    rows = vcat(frames...; cols = :union)
    all(isfinite, rows.construction_ratio) && all(>(0), rows.construction_ratio) ||
        error("construction ratios must be finite and positive")
    counts = paired_counts(rows)
    summary = summarize_pareto(rows)

    CSV.write(joinpath(OUTPUT_DIR, "pareto_rows.csv"), rows)
    CSV.write(joinpath(OUTPUT_DIR, "paired_counts.csv"), counts)
    CSV.write(joinpath(OUTPUT_DIR, "pareto_summary.csv"), summary)
    figure, xlimits, ylimits = build_figure(rows)
    outputs = save_publication_figure(
        joinpath(OUTPUT_DIR, "construction_quality_pareto"),
        figure,
    )

    println("cohorts:")
    for specification in PANELS
        panel = filter(:panel => ==(specification.name), rows)
        println("  $(specification.row_label): $(first(panel.cohort_pairs)) pairs")
    end
    println("axis limits: x=$xlimits, y=$ylimits")
    println("paired ranges:")
    for group in groupby(counts, [:row_label, :sweeps])
        println(
            "  $(first(group.row_label)), $(first(group.sweeps)) sweeps: ",
            extrema(group.paired_pairs),
        )
    end
    println("wrote:")
    foreach(path -> println("  ", path), outputs)
end

main()
