"""Analyse or modify simulation conditions.

This module contains functions for analysing the quality of a
simulation, or for modifying the atomic data or parameter files of a
simulation.
"""

import re
from copy import copy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import pysi.sim.grid

END_SUMMARY_LINES = 9


def model_convergence(
    root: str,
    path: str | Path = Path(),
    return_per_cycle: bool = False,  # noqa: FBT001, FBT002
    return_converging: bool = False,  # noqa: FBT001, FBT002
) -> float | list[float]:
    """Check the convergence of a SIROCCO simulation.

    This function looks for !!Check_convergence in the 00 diag file.

    TODO(EP): This should be cleaned up, but it works for now.

    Parameters
    ----------
    root: str
        The root name of the SIROCCO simulation
    path: str [optional]
        The working directory of the simulation
    return_per_cycle: bool [optional]
        Return the convergence fraction for each cycle
    return_converging: bool [optional]
        Return the number of cells which are still converging instead of the
        number of cells which have converged

    Returns
    -------
    convergence: float or int or list
        The convergence fraction in the final cycle of the simulation. If this
        is -1, then a convergence fraction was not found.

    """
    converging = -1
    convergence = -1
    prev_line = ""

    convergence_per_cycle = []
    converging_per_cycle = []
    with Path(f"{path}/diag_{root}/{root}_00.diag").open() as f:
        diag_lines = f.readlines()

    for i in range(len(diag_lines)):
        line = diag_lines[i]
        if line.find("converged") != -1 and line.find("converging") != -1:
            # Skip if the convergence statistic is from the brief run summary
            if prev_line.find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += END_SUMMARY_LINES  # noqa: PLW2901
                continue
            line = line.split()
            try:
                convergence = float(line[2].replace("(", "").replace(")", ""))
                convergence_per_cycle.append(convergence)
            except ValueError:
                convergence_per_cycle.append(-1)
                continue
            try:
                converging = float(line[6].replace("(", "").replace(")", ""))
                converging_per_cycle.append(converging)
            except ValueError:
                converging_per_cycle.append(-1)
                continue
        prev_line = copy(line)

    if return_converging:
        if return_per_cycle:
            return converging_per_cycle
        return converging

    if return_per_cycle:
        return convergence_per_cycle

    return convergence


def model_convergence_components(
    root: str, path: str | Path = Path()
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Get the convergence statistics for a SIROCCO simulation.

    Returns a break down in terms of the number of cells which have passed

    the convergence checks on radiation temperature, electron temperature and
    heating and cooling balance.

    Parameters
    ----------
    root: str
        The root name of the SIROCCO simulation
    path: str [optional]
        The working directory of the SIROCCO simulation
    Returns
    -------
    n_tr: List[float]
        The fraction of cells which have passed the radiation temperature
        convergence test.
    n_te: List[float]
        The fraction of cells which have passed the electron temperature
        convergence test.
    n_te_max: List[float]
        The fraction of cells which have reached the maximum electron
        temperature.
    n_hc: List[float]
        The fraction of cells which have passed the heating/cooling
        convergence test.

    """
    brief_summary_len = 7
    n_tr = []
    n_te = []
    n_hc = []
    n_te_max = []
    file_found = False

    diag_path = f"{path}/diag_{root}/{root}_00.diag"

    try:
        with Path(diag_path).open() as f:
            diag = f.readlines()
        file_found = True
    except OSError:
        pass

    if not file_found:
        raise OSError(f"unable to find {root}_0.diag file")

    for i in range(len(diag)):
        line = diag[i]
        if line.find("t_r") != -1 and line.find("t_e(real)") != -1 and line.find("hc(real)") != -1:
            if diag[i - 2].find("Convergence statistics for the wind after the ionization calculation:") != -1:
                i += brief_summary_len  # noqa: PLW2901
                continue

            line = line.split()

            try:
                n_cells = int(diag[i - 1].split()[9])
                n_tr.append(int(line[2]) / n_cells)
                n_te.append(int(line[4]) / n_cells)
                n_te_max.append(int(line[6]) / n_cells)
                n_hc.append(int(line[8]) / n_cells)
            except ValueError:
                n_tr.append(-1)
                n_te.append(-1)
                n_te_max.append(-1)
                n_hc.append(-1)

    return n_tr, n_te, n_te_max, n_hc


def model_errors(root: str, path: str | Path = Path(), n_cores: int = -1, print_errors: bool = False) -> dict:
    """Get the error summary for a SIROCCO simulation.

    TODO(EP): this is too complex
    TODO(EP): return array-like/dict for each MPI process

    Parameters
    ----------
    root: str
        The root name of the SIROCCO simulation
    path: str [optional]
        The working directory of the SIROCCO simulation

    n_cores: int [optional]
        If this is provided, then only the first n_cores processes will be
        checked for errors
    print_errors: bool [optional]
        Print the error summary to screen

    Returns
    -------
    total_errors: dict
        A dictionary containing the total errors over all processors. The keys
        are the error messages and the values are the number of times that
        error occurred.

    """
    total_errors = {}
    diag_files = Path(f"{path}/diag_{root}").glob("*.diag")
    diag_files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])
    n_diag_files = n_cores if n_cores > 0 else len(diag_files)
    if n_diag_files == 0:
        raise ValueError(f"No .diag files were found in {path}/diag_{root} so cannot find any errors")

    broken_diag_files = []
    exit_msg = "ended for unknown reasons"

    for diag in diag_files:
        try:
            with Path(diag).open() as f:
                lines = f.readlines()
        except OSError:
            broken_diag_files.append(diag)
            continue

        # Find the final error summary: look over the lines list in reverse

        error_start = -1

        for k, start_line in enumerate(lines):
            # Find start of error summary
            if start_line.find("Error summary:") != -1:
                error_start = k
                exit_msg = "exited successfully" if start_line.find("End of program") != -1 else "was aborted"
            else:
                continue

            error_end = -1
            for kk, end_line in enumerate(lines[error_start + 3 :]):
                end_line = end_line.split()  # noqa: PLW2901
                if len(end_line) and end_line[0].isdigit() is False:
                    error_end = error_start + kk + 3
                    break

            if error_end == -1:
                broken_diag_files.append(diag)
                break

            # Extract the errors from the diag file

            errors = lines[error_start:error_end]

            for error_line in errors:
                error_words = error_line.split()
                if len(error_words) == 0:
                    continue
                try:
                    error_count = error_words[0]
                except IndexError:
                    broken_diag_files.append(diag)
                    break

                if error_count.isdigit():
                    try:
                        error_count = int(error_words[0])
                    except ValueError:
                        continue
                    error_message = " ".join(error_words[2:])
                    try:
                        total_errors[error_message] += error_count
                    except KeyError:
                        total_errors[error_message] = error_count

        if error_start == -1:
            broken_diag_files.append(diag)

    if len(broken_diag_files) == len(diag_files):
        raise OSError(f"Unable to find any error summaries for {root + path}")

    if print_errors:
        n_reported = len(diag_files) - len(broken_diag_files)
        print(  # noqa: T201
            f"Total errors reported from {n_reported} of {len(diag_files)} processes for {path + root}, which {exit_msg}:"
        )
        for key in total_errors:
            n_error = int(total_errors[key])
            n_error = max(n_error, 1)
            print(f"  {n_error:6d} -- {key}")  # noqa: T201

    return total_errors


def plot_model_convergence(
    root: str, path: str | Path = Path(), *, display: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the convergence of the model.

    Parameters
    ----------
    root : str
        The root name of the simulation.
    path : str | Path, optional
        The directory containing the simulation, by default Path()
    display : bool, optional
        Whether to display the figure, by default False

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        A tuple of the figure and axes objects containing the plot of the model convergence.

    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    convergence = model_convergence(root, path, return_per_cycle=True)
    converging = model_convergence(root, path, return_per_cycle=True, return_converging=True)
    tr, te, te_max, hc = model_convergence_components(root, path)

    cycles = np.arange(1, len(convergence) + 1, 1)
    ax.set_ylim(0, 1.05)

    # As the bare minimum, we need the convergence per cycle but if the other
    # convergence stats are passed as well, plot those too

    ax.plot(cycles, convergence, label="Convergence")
    # ax.plot(cycles, converging, label="Converging")
    ax.plot(cycles, tr, "--", label="Radiation temperature", alpha=0.65)
    ax.plot(cycles, te, "--", label="Electron temperature", alpha=0.65)
    ax.plot(cycles, hc, "--", label="Heating and cooling", alpha=0.65)
    for value in te_max:
        if value > 0:
            ax.plot(cycles, te_max, "--", label="Electron temperature max", alpha=0.65)
            break

    ax.legend()
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Fraction of cells passed")
    fig = pysi.plot.finish_figure(fig, title=f"Final convergence = {float(convergence[-1]) * 100:4.2f}%")

    if display:
        plt.show()
    else:
        plt.close()

    return fig, ax
