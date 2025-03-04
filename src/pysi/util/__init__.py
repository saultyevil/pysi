from .array import check_array_is_ascending_order, find_where_target_in_array, get_subset_in_second_array, smooth_array
from .files import read_file_with_header, remove_suffix_from_string, split_root_and_directory
from .plot import (
    finish_figure,
    get_subplot_dims,
    plot_pcolor,
    plot_scatter,
    prepare_fig_and_ax,
    remove_extra_axes,
    set_axes_scales,
    set_figure_style,
)
from .run import run_py_optical_depth, run_py_wind, run_windsave2table
from .shell import find_file_with_pattern, run_shell_command

__all__ = [
    "run_shell_command",
    "find_file_with_pattern",
    "check_array_is_ascending_order",
    "find_where_target_in_array",
    "get_subset_in_second_array",
    "smooth_array",
    "read_file_with_header",
    "remove_suffix_from_string",
    "split_root_and_directory",
    "plot_pcolor",
    "plot_scatter",
    "finish_figure",
    "set_figure_style",
    "remove_extra_axes",
    "set_axes_scales",
    "get_subplot_dims",
    "prepare_fig_and_ax",
    "run_windsave2table",
    "run_py_optical_depth",
    "run_py_wind",
]
