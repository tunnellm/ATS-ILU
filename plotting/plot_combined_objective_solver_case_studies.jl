#!/usr/bin/env julia

using CSV
using CairoMakie
using DataFrames
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
const DIAGNOSTIC_OBJECTIVE_FLOOR = 1e-7
const SYMLOG_OBJECTIVE_THRESHOLD = 1e-7
const SYMLOG_LINEAR_SCALE = 2.0

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
        solver_label = "GMRES iterations",
        sweeps = 1:3,
        methods = [
            (
                method = "ats_ilu",
                label = "ATS-ILU",
                color = ATS_COLOR,
                variants = [(variant = "async", mode = "async"),
                            (variant = "sync_folded", mode = "sync")],
            ),
            (
                method = "parilu",
                label = "ParILU",
                color = PAR_COLOR,
                variants = [(variant = "async", mode = "async"),
                            (variant = "sync", mode = "sync")],
            ),
        ],
        cases = [
            (
                pair = "ML_Geer",
                k = 3,
                title = "(a) ML_Geer, k = 3",
                xlimits = (1e-16, 1e5),
                xticks = ([1e-15, 1e-9, 1e-3, 1e3],
                          ["10^-15", "10^-9", "10^-3", "10^3"]),
                ylimits = (35.0, 430.0),
                yticks = [40, 60, 100, 200, 400],
            ),
            (
                pair = "Transport",
                k = 3,
                title = "(b) Transport, k = 3",
                xlimits = (0.5, 5e3),
                xticks = ([1.0, 10.0, 100.0, 1e3], ["1", "10", "100", "10^3"]),
                ylimits = (58.0, 200.0),
                yticks = [60, 80, 100, 140, 180],
            ),
            (
                pair = "CoupCons3D",
                k = 3,
                title = "(c) CoupCons3D, k = 3",
                xlimits = (0.5, 1e4),
                xticks = ([1.0, 10.0, 100.0, 1e3], ["1", "10", "100", "10^3"]),
                ylimits = (6.5, 52.0),
                yticks = [7, 10, 20, 40],
            ),
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
        solver_label = "CG iterations",
        sweeps = 1:3,
        methods = [
            (
                method = "ats_ic",
                label = "ATS-IC",
                color = ATS_COLOR,
                variants = [(variant = "async", mode = "async")],
            ),
            (
                method = "par_ic",
                label = "ParIC",
                color = PAR_COLOR,
                variants = [(variant = "async", mode = "async")],
            ),
        ],
        cases = [
            (
                pair = "Janna/Geo_1438",
                k = 1,
                title = "(d) Geo_1438, k = 1",
                xlimits = (0.8, 1e3),
                xticks = ([1.0, 10.0, 100.0, 1e3], ["1", "10", "100", "10^3"]),
                ylimits = (102.0, 148.0),
                yticks = [105, 120, 140],
            ),
            (
                pair = "Oberwolfach/bone010",
                k = 3,
                title = "(e) bone010, k = 3",
                xlimits = (0.2, 1e3),
                xticks = ([1.0, 10.0, 100.0, 1e3], ["1", "10", "100", "10^3"]),
                ylimits = (200.0, 700.0),
                yticks = [200, 300, 500, 700],
            ),
            (
                pair = "Janna/Flan_1565",
                k = 0,
                title = "(f) Flan_1565, k = 0",
                xlimits = (1.0, 2e3),
                xticks = ([1.0, 10.0, 100.0, 1e3], ["1", "10", "100", "10^3"]),
                ylimits = (695.0, 760.0),
                yticks = [700, 725, 750],
            ),
        ],
    ),
]

function successful_row(row)
    return coalesce(row.converged == 1, false) && !ismissing(row.objective) &&
           isfinite(row.objective) && row.objective > 0 && !ismissing(row.iterations) &&
           isfinite(row.iterations) && row.iterations > 0
end

function sequential_reference(data::DataFrame, block, case)
    rows = filter(
        row -> string(row[block.pair_column]) == case.pair && row.k == case.k &&
               row.method == block.baseline_method,
        data,
    )
    nrow(rows) == 1 || error("expected one sequential row for $(case.pair), k=$(case.k)")
    row = rows[1, :]
    successful_row(row) || error("invalid sequential row for $(case.pair), k=$(case.k)")
    return (objective = Float64(row.objective), iterations = Int(row.iterations))
end

function case_summary(data::DataFrame, block, case)
    selected = filter(
        row -> string(row[block.pair_column]) == case.pair && row.k == case.k &&
               row.method in getproperty.(block.methods, :method),
        data,
    )
    rows = NamedTuple[]

    for method_config in block.methods
        for variant_config in method_config.variants
            for sweeps in block.sweeps
                group = filter(
                    row -> row.method == method_config.method &&
                           row.variant == variant_config.variant && row.sweeps == sweeps,
                    selected,
                )
                expected = variant_config.mode == "async" ? 21 : 1
                nrow(group) == expected || error(
                    "$(case.pair) $(method_config.method) $(variant_config.mode) " *
                    "sweep $sweeps has $(nrow(group)) rows; expected $expected",
                )
                finite = filter(successful_row, group)
                objectives = Float64.(finite.objective)
                iterations = Float64.(finite.iterations)
                has_finite = !isempty(objectives)
                push!(
                    rows,
                    (
                        dataset = block.name,
                        pair = case.pair,
                        k = case.k,
                        method = method_config.method,
                        method_label = method_config.label,
                        mode = variant_config.mode,
                        sweeps = sweeps,
                        requested_observations = nrow(group),
                        finite_observations = nrow(finite),
                        objective_min = has_finite ? minimum(objectives) : missing,
                        objective_median = has_finite ? median(objectives) : missing,
                        objective_max = has_finite ? maximum(objectives) : missing,
                        iterations_min = has_finite ? minimum(iterations) : missing,
                        iterations_median = has_finite ? median(iterations) : missing,
                        iterations_max = has_finite ? maximum(iterations) : missing,
                    ),
                )
            end
        end
    end
    return DataFrame(rows)
end

function draw_trajectories!(axis, summary::DataFrame, block)
    for method_config in block.methods
        for variant_config in method_config.variants
            rows = filter(
                row -> row.method == method_config.method &&
                       row.mode == variant_config.mode && row.finite_observations > 0,
                summary,
            )
            isempty(rows) && continue
            sort!(rows, :sweeps)
            mode = variant_config.mode
            linestyle = mode == "async" ? ASYNC_LINESTYLE : SYNC_LINESTYLE
            marker = mode == "async" ? :circle : :rect

            lines!(
                axis,
                rows.objective_median,
                rows.iterations_median,
                color = method_config.color,
                linewidth = mode == "async" ? PLOT_LINE_WIDTH : SYNC_LINE_WIDTH,
                linestyle = linestyle,
            )
            scatter!(
                axis,
                rows.objective_median,
                rows.iterations_median,
                color = mode == "async" ? method_config.color : :white,
                marker = marker,
                markersize = MARKER_SIZE,
                strokecolor = method_config.color,
                strokewidth = 1.0,
            )
        end
    end
end

function legend_elements(block)
    elements = Any[]
    labels = String[]
    for method_config in block.methods
        for variant_config in method_config.variants
            mode = variant_config.mode
            push!(
                elements,
                LineElement(
                    color = method_config.color,
                    linewidth = mode == "async" ? PLOT_LINE_WIDTH : SYNC_LINE_WIDTH,
                    linestyle = mode == "async" ? ASYNC_LINESTYLE : SYNC_LINESTYLE,
                    marker = mode == "async" ? :circle : :rect,
                    markercolor = mode == "async" ? method_config.color : :white,
                    markerstrokecolor = method_config.color,
                    markersize = MARKER_SIZE,
                ),
            )
            push!(labels, "$(method_config.label) $(mode).")
        end
    end
    return elements, labels
end

function objective_axis(case; objective_floor = nothing, symlog_threshold = nothing)
    if !isnothing(symlog_threshold)
        upper = case.xlimits[2]
        lower_exponent = round(Int, log10(symlog_threshold))
        upper_exponent = floor(Int, log10(upper))
        exponents = unique(round.(Int, range(lower_exponent, upper_exponent; length = 4)))
        ticks = vcat(0.0, 10.0 .^ exponents)
        labels = vcat("0", map(exponents) do exponent
            exponent == 0 ? "1" : exponent == 1 ? "10" : "10^$exponent"
        end)
        return (0.0, upper), (ticks, labels)
    end

    if isnothing(objective_floor)
        return case.xlimits, case.xticks
    end

    lower = max(case.xlimits[1], objective_floor)
    upper = case.xlimits[2]
    lower_exponent = round(Int, log10(lower)) + 1
    upper_exponent = round(Int, log10(upper)) - 1
    exponents = unique(round.(Int, range(lower_exponent, upper_exponent; length = 4)))
    ticks = 10.0 .^ exponents
    labels = map(exponents) do exponent
        exponent == 0 ? "1" : exponent == 1 ? "10" : "10^$exponent"
    end
    return (lower, upper), (ticks, labels)
end

function build_figure(
    block_data;
    objective_floor = nothing,
    symlog_threshold = nothing,
)
    apply_paper_theme!()
    figure = paper_figure(aspect_ratio = SIAM_LINEWIDTH_BP / 285, padding = 1)
    summaries = DataFrame[]
    axes = Matrix{Any}(undef, 2, 3)
    xscale = isnothing(symlog_threshold) ?
             log10 :
             Makie.Symlog10(symlog_threshold; linscale = SYMLOG_LINEAR_SCALE)

    Label(
        figure[3, 1:3],
        rich("Pattern-restricted objective, f", subscript("S")),
        fontsize = AXIS_LABEL_SIZE,
        tellheight = true,
    )

    for (block_index, item) in enumerate(block_data)
        block, data = item.block, item.data
        for (column, case) in enumerate(block.cases)
            summary = case_summary(data, block, case)
            sequential = sequential_reference(data, block, case)
            xlimits, xticks = objective_axis(
                case;
                objective_floor = objective_floor,
                symlog_threshold = symlog_threshold,
            )
            summary.sequential_objective = fill(sequential.objective, nrow(summary))
            summary.sequential_iterations = fill(sequential.iterations, nrow(summary))
            push!(summaries, summary)

            axis = Axis(
                figure[block_index, column],
                xscale = xscale,
                xreversed = true,
                yscale = log10,
                xticks = xticks,
                yticks = case.yticks,
                ylabel = column == 1 ? block.solver_label : "",
                xgridwidth = 1.0,
                ygridwidth = 1.0,
            )
            hidespines!(axis)
            axis.xticksvisible = false
            axis.yticksvisible = false
            xlims!(axis, xlimits...)
            ylims!(axis, case.ylimits...)
            axis.xreversed[] = true
            draw_trajectories!(axis, summary, block)
            text!(
                axis,
                0.5,
                0.95,
                text = case.title,
                space = :relative,
                align = (:center, :top),
                fontsize = TITLE_SIZE,
            )
            axes[block_index, column] = axis
        end
    end

    top_elements, top_labels = legend_elements(BLOCKS[1])
    axislegend(
        axes[1, 1],
        top_elements,
        top_labels,
        position = (0.76, 0.72),
        orientation = :vertical,
        nbanks = 1,
        backgroundcolor = (:white, 0.84),
        labelsize = TICK_LABEL_SIZE,
        patchsize = (17, 7),
        rowgap = 1,
    )

    bottom_elements, bottom_labels = legend_elements(BLOCKS[2])
    axislegend(
        axes[2, 1],
        bottom_elements,
        bottom_labels,
        position = (0.76, 0.76),
        orientation = :vertical,
        nbanks = 1,
        backgroundcolor = (:white, 0.84),
        labelsize = TICK_LABEL_SIZE,
        patchsize = (17, 7),
        rowgap = 1,
    )
    text!(
        axes[2, 3],
        0.5,
        0.04,
        text = "ParIC: no finite factors",
        space = :relative,
        align = (:center, :bottom),
        color = PAR_COLOR,
        fontsize = TICK_LABEL_SIZE,
    )

    colgap!(figure.layout, paper_length(14, 940))
    rowgap!(figure.layout, 6)
    return figure, vcat(summaries...)
end

function main()
    block_data = NamedTuple[]
    for block in BLOCKS
        data = CSV.read(block.input, DataFrame; stringtype = String)
        push!(block_data, (block = block, data = data))
    end
    figure, summary = build_figure(block_data)
    mkpath(DEFAULT_OUTPUT_DIR)
    stem = joinpath(DEFAULT_OUTPUT_DIR, "objective_vs_solver_case_studies_combined")
    CSV.write("$stem.csv", summary)
    for path in save_publication_figure(stem, figure)
        println(path)
    end

end

main()
