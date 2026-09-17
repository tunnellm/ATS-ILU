module ATSILUPaperStyle

using CairoMakie

export ATS_COLOR,
    ASYNC_LINESTYLE,
    AXIS_LABEL_SIZE,
    MARKER_SIZE,
    PAR_COLOR,
    PLOT_LINE_WIDTH,
    SEQUENTIAL_COLOR,
    SIAM_LINEWIDTH_BP,
    SIAM_LINEWIDTH_IN,
    SIAM_LINEWIDTH_TEX_PT,
    SYNC_LINE_WIDTH,
    SYNC_LINESTYLE,
    TICK_LABEL_SIZE,
    TITLE_FONT,
    TITLE_SIZE,
    PROFILE_XLIMS,
    PROFILE_XTICKS,
    PROFILE_XTICKLABELS,
    PROFILE_YTICKS,
    PROFILE_YTICKLABELS,
    THREADS,
    apply_paper_theme!,
    paper_figure,
    paper_length,
    profile_axis,
    save_publication_figure,
    thread_palette

const ATS_COLOR = colorant"#762125"
const PAR_COLOR = colorant"#1D4167"
const SEQUENTIAL_COLOR = colorant"#5F6368"
const ASYNC_LINESTYLE = :solid
const SYNC_LINESTYLE = :dash

# siamart251216 uses a 5.125 in text block. See the standard class linked from
# https://epubs.siam.org/journal-authors. PDF dimensions are expressed in big
# points (72 bp/in), while TeX uses 72.27 pt/in.
const SIAM_LINEWIDTH_IN = 5.125
const SIAM_LINEWIDTH_BP = 369
const SIAM_LINEWIDTH_TEX_PT = 370.38375

# These values are final printed point sizes because publication PDFs are saved
# at one PDF point per Makie unit and are already exactly one \linewidth wide.
const TITLE_SIZE = 8.0
const TITLE_FONT = :bold
const AXIS_LABEL_SIZE = 7.0
const TICK_LABEL_SIZE = 6.0
const PLOT_LINE_WIDTH = 1.1
const SYNC_LINE_WIDTH = 1.35
const MARKER_SIZE = 5.0

const THREADS = [2, 4, 8, 16, 32, 64, 128]
const PROFILE_XLIMS = (0.5, 64.0)
const PROFILE_XTICKS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
const PROFILE_XTICKLABELS = ["0.5", "1", "2", "4", "8", "16", "32", "64"]
const PROFILE_YTICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
const PROFILE_YTICKLABELS = ["0%", "25%", "50%", "75%", "100%"]

const ATS_THREAD_COLORS = [
    colorant"#E5B9B8",
    colorant"#D99594",
    colorant"#CD7473",
    colorant"#BC5555",
    colorant"#A83D3F",
    colorant"#922D30",
    colorant"#762125",
]

const PAR_THREAD_COLORS = [
    colorant"#BBCDE0",
    colorant"#98B5D2",
    colorant"#759DC2",
    colorant"#5684AE",
    colorant"#3D6D99",
    colorant"#2C5781",
    colorant"#1D4167",
]

function thread_palette(method::AbstractString)
    method == "ats_ilu" && return ATS_THREAD_COLORS
    method == "parilu" && return PAR_THREAD_COLORS
    error("unsupported method: $method")
end

function apply_paper_theme!()
    set_theme!(
        Theme(
            fontsize = AXIS_LABEL_SIZE,
            fonts = (
                regular = "TeX Gyre Heros Makie",
                bold = "TeX Gyre Heros Makie Bold",
            ),
            backgroundcolor = :white,
            figure_padding = 5,
            Axis = (
                backgroundcolor = :white,
                xgridcolor = (:black, 0.10),
                ygridcolor = (:black, 0.10),
                xgridwidth = 1.0,
                ygridwidth = 1.0,
                xticklabelsize = TICK_LABEL_SIZE,
                yticklabelsize = TICK_LABEL_SIZE,
                xlabelsize = AXIS_LABEL_SIZE,
                ylabelsize = AXIS_LABEL_SIZE,
                titlesize = TITLE_SIZE,
                titlefont = TITLE_FONT,
                titlegap = 3,
            ),
            Legend = (
                backgroundcolor = :transparent,
                framevisible = false,
                labelsize = TICK_LABEL_SIZE,
                patchsize = (17, 7),
                rowgap = 1,
                colgap = 8,
            ),
        ),
    )
end

"""Create a full-width SIAM figure with a specified width-to-height ratio."""
function paper_figure(; aspect_ratio::Real, padding::Real = 4)
    aspect_ratio > 0 || throw(ArgumentError("aspect_ratio must be positive"))
    padding >= 0 || throw(ArgumentError("padding must be nonnegative"))
    height = round(Int, SIAM_LINEWIDTH_BP / aspect_ratio)
    return Figure(
        size = (SIAM_LINEWIDTH_BP, height),
        figure_padding = padding,
        backgroundcolor = :white,
    )
end

"""Scale a layout length from a reference canvas to the SIAM line width."""
function paper_length(value::Real, reference_width::Real)
    reference_width > 0 || throw(ArgumentError("reference_width must be positive"))
    return value * SIAM_LINEWIDTH_BP / reference_width
end

function profile_axis(
    parent;
    x_ticklabels::Vector{String} = fill("", length(PROFILE_XTICKLABELS)),
    y_ticklabels::Vector{String} = fill("", length(PROFILE_YTICKLABELS)),
    x_ticklabelrotation::Real = 0,
)
    axis = Axis(
        parent,
        xscale = log2,
        xticks = (PROFILE_XTICKS, x_ticklabels),
        yticks = (PROFILE_YTICKS, y_ticklabels),
        xticklabelrotation = x_ticklabelrotation,
    )

    hidespines!(axis)
    axis.xticksvisible = false
    axis.yticksvisible = false
    xlims!(axis, PROFILE_XLIMS...)
    ylims!(axis, -0.005, 1.005)
    return axis
end

function save_publication_figure(stem::AbstractString, figure; png_scale::Real = 2)
    figure_width = first(size(figure.scene))
    figure_width == SIAM_LINEWIDTH_BP || error(
        "publication figure width is $figure_width, expected $SIAM_LINEWIDTH_BP",
    )
    mkpath(dirname(stem))
    outputs = String[]
    for extension in ("pdf", "svg")
        path = "$stem.$extension"
        save(path, figure, pt_per_unit = 1)
        push!(outputs, path)
    end
    png_path = "$stem.png"
    save(png_path, figure, px_per_unit = png_scale)
    push!(outputs, png_path)
    return outputs
end

end
