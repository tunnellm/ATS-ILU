#!/usr/bin/env julia

using CSV
using DataFrames
using Printf
using Statistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const INPUT_PATH = joinpath(
    ROOT,
    "results",
    "publication_timing_2026-08-04",
    "figure_04_pareto",
    "pareto_rows.csv",
)
const OUTPUT_DIR = @__DIR__
const TRANSITIONS = [(1, 2), (2, 3)]
const PANEL_ORDER = ["async_ilu", "sync_ilu", "async_ic"]
const PANEL_METADATA = Dict(
    "async_ilu" => (
        ats_label = "ATS-ILU",
        par_label = "ParILU",
        schedule = "Asynchronous",
    ),
    "sync_ilu" => (
        ats_label = "ATS-ILU",
        par_label = "ParILU",
        schedule = "Synchronous",
    ),
    "async_ic" => (
        ats_label = "ATS-IC",
        par_label = "ParIC",
        schedule = "Asynchronous",
    ),
)

geomean(values) = exp(mean(log.(values)))

function endpoint_rows(rows::DataFrame, sweeps::Int, suffix::String)
    selected = filter(:sweeps => ==(sweeps), rows)
    return select(
        selected,
        :canonical_matrix,
        :k,
        :threads,
        :ats_valid => Symbol("ats_valid_", suffix),
        :par_valid => Symbol("par_valid_", suffix),
        :ats_iterations => Symbol("ats_iterations_", suffix),
        :par_iterations => Symbol("par_iterations_", suffix),
        :ats_construction_seconds => Symbol("ats_construction_", suffix),
        :par_construction_seconds => Symbol("par_construction_", suffix),
    )
end

function summarize_method(joined::DataFrame, method::String)
    construction_before = joined[!, Symbol(method, "_construction_before")]
    construction_after = joined[!, Symbol(method, "_construction_after")]
    iterations_before = joined[!, Symbol(method, "_iterations_before")]
    iterations_after = joined[!, Symbol(method, "_iterations_after")]
    construction_ratio = Float64.(construction_after ./ construction_before)
    iteration_ratio = Float64.(iterations_after ./ iterations_before)
    return (
        configurations = nrow(joined),
        construction_multiplier = geomean(construction_ratio),
        iteration_multiplier = geomean(iteration_ratio),
        fraction_improved = mean(iteration_ratio .< 1.0),
        fraction_unchanged = mean(iteration_ratio .== 1.0),
    )
end

function summarize(rows::DataFrame)
    output = NamedTuple[]
    for panel in PANEL_ORDER
        panel_rows = filter(:panel => ==(panel), rows)
        metadata = PANEL_METADATA[panel]
        for (before, after) in TRANSITIONS
            left = endpoint_rows(panel_rows, before, "before")
            right = endpoint_rows(panel_rows, after, "after")
            joined = innerjoin(
                left,
                right;
                on = [:canonical_matrix, :k, :threads],
                validate = (true, true),
            )
            valid = joined.ats_valid_before .& joined.ats_valid_after .&
                    joined.par_valid_before .& joined.par_valid_after
            joined = joined[valid, :]
            for (method, method_label) in
                (("ats", metadata.ats_label), ("par", metadata.par_label))
                values = summarize_method(joined, method)
                push!(output, (
                    panel = panel,
                    method = method,
                    method_label = method_label,
                    schedule = metadata.schedule,
                    sweep_before = before,
                    sweep_after = after,
                    values...,
                ))
            end
        end
    end
    return DataFrame(output)
end

function result(summary::DataFrame, panel::String, method::String, before::Int)
    rows = filter(
        row -> row.panel == panel && row.method == method &&
               row.sweep_before == before,
        summary,
    )
    nrow(rows) == 1 || error("expected one marginal-sweep row")
    return first(eachrow(rows))
end

function write_table(summary::DataFrame)
    path = joinpath(OUTPUT_DIR, "table.tex")
    open(path, "w") do stream
        println(stream, "\\begin{table}[htbp]")
        println(stream, "\\centering")
        println(stream, "\\small")
        println(stream, "\\setlength{\\tabcolsep}{4pt}")
        println(stream, "\\begin{tabular}{@{}llrrrrrr@{}}")
        println(stream, "\\toprule")
        println(stream, "\\multirow{2}{*}{Method} &")
        println(stream, "\\multirow{2}{*}{Schedule} &")
        println(stream, "\\multicolumn{3}{c}{\$1\\!\\to\\!2\$ sweeps} &")
        println(stream, "\\multicolumn{3}{c}{\$2\\!\\to\\!3\$ sweeps} \\\\")
        println(stream, "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}")
        println(stream, "& & \$T_2/T_1\$ & \$K_2/K_1\$ & Improved &")
        println(stream, "\$T_3/T_2\$ & \$K_3/K_2\$ & Improved \\\\")
        println(stream, "\\midrule")
        for (panel_index, panel) in enumerate(PANEL_ORDER)
            metadata = PANEL_METADATA[panel]
            for (method, method_label) in
                (("ats", metadata.ats_label), ("par", metadata.par_label))
                first_transition = result(summary, panel, method, 1)
                second_transition = result(summary, panel, method, 2)
                @printf(
                    stream,
                    "%s & %s & %.2f & %.2f & %.0f\\%% & %.2f & %.2f & %.0f\\%% \\\\\n",
                    method_label,
                    metadata.schedule,
                    first_transition.construction_multiplier,
                    first_transition.iteration_multiplier,
                    100 * first_transition.fraction_improved,
                    second_transition.construction_multiplier,
                    second_transition.iteration_multiplier,
                    100 * second_transition.fraction_improved,
                )
            end
            panel_index < length(PANEL_ORDER) && println(stream, "\\addlinespace")
        end
        println(stream, "\\bottomrule")
        println(stream, "\\end{tabular}")
        println(stream, "\\caption{Marginal construction and Krylov effects of an additional sweep.}")
        println(stream, "\\label{tab:marginal-sweeps}")
        println(stream, "\\end{table}")
    end
    return path
end

function main()
    rows = CSV.read(INPUT_PATH, DataFrame; stringtype = String)
    summary = summarize(rows)
    CSV.write(joinpath(OUTPUT_DIR, "marginal_sweep_summary.csv"), summary)
    table_path = write_table(summary)
    println("wrote:")
    println("  ", joinpath(OUTPUT_DIR, "marginal_sweep_summary.csv"))
    println("  ", table_path)
end

main()
