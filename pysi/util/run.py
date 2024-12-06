"""Run related SIROCCO programs.

This gives an interface to running swind or windsave2table or etc.
"""

import time
from os import listdir
from pathlib import Path
from shutil import which
from subprocess import run

from pysi import error
from pysi.util.shell import run_shell_command


def run_windsave2table(  # noqa: PLR0913
    root: str,
    file_path: Path | str = Path(),
    *,
    ion_density: bool = False,
    cell_spec: bool = False,
    version: str | None = None,
    verbose: bool = False,
) -> None:
    """Run windsave2table in a directory to create the standard data tables.

    The function can also create a root.all.complete.txt file which merges all
    the data tables together into one (a little big) file.

    Parameters
    ----------
    root: str
        The root name of the Python simulation
    file_path: str
        The directory where windsave2table will run
    ion_density: bool [optional]
        Use windsave2table in the ion density version instead of ion fractions
    cell_spec: bool [optional]
        Use windsave2table to get the cell spectra.
    version: str [optional]
        The version number of windsave2table to use
    verbose: bool [optional]
        Enable verbose output

    """
    name = "windsave2table"
    if version:
        name += version
        with Path.open(f"{file_path}/.sirocco-version", "w") as file_out:
            file_out.write(f"{version}\n")

    in_path = which(name)
    if not in_path:
        raise OSError(f"{name} not in $PATH and executable")

    files_before = listdir(file_path)

    if not Path(f"{file_path}/data").exists():
        run_shell_command("Setup_Py_Dir", file_path)

    command = [name]
    if ion_density:
        command.append("-d")
    if cell_spec:
        command.append("-xall")
    command.append(root)

    cmd = run_shell_command(command, file_path, verbose)
    if cmd.returncode != 0:
        raise error.RunError(
            f"windsave2table has failed to run, possibly due to an incompatible version\n{cmd.stdout.decode('utf-8')}"
        )

    files_after = listdir(file_path)

    # Move the new files in fp/tables
    s = set(files_before)
    new_files = [x for x in files_after if x not in s]
    Path(f"{file_path}/tables").mkdir(exist_ok=True)
    for new in new_files:
        try:
            Path(f"{file_path}/{new}").rename(f"{file_path}/tables/{new}")
        except PermissionError:  # noqa: PERF203
            time.sleep(1.5)
            Path(f"{file_path}/{new}").rename(f"{file_path}/tables/{new}")

    return cmd.returncode


def run_py_optical_depth(
    root: str, file_path: Path | str = Path(), *, scatter_surface: float | None = None, verbose: bool = False
) -> None:
    """Run `py_optical_depth` with the provided parameters.

    Parameters
    ----------
    root: str
        The root name of the model.
    file_path: [optional] str
        The directory containing the model.
    scatter_surface: [optional] float
        The scattering optical depth to find the surface of.
    verbose: bool
        Print stdout to the screen.

    """
    command = ["py_optical_depth"]
    if scatter_surface:
        command.append(f"-p {float(scatter_surface)}")
    command.append(root)

    cmd = run_shell_command(command, file_path)
    stdout, stderr = cmd.stdout, cmd.stderr

    if verbose:
        print(stdout.decode("utf-8"))  # noqa: T201
    if stderr:
        print(stderr.decode("utf-8"))  # noqa: T201


def run_py_wind(root: str, commands: list[str], file_path: Path | str = Path()) -> None:
    """Run py_wind with the provided commands.

    Parameters
    ----------
    root : str
        The root name of the model.
    commands : list[str]
        The commands to pass to py_wind.
    file_path : [optional] str
        The directory containing the model.

    Returns
    -------
    output: list[str]
        The stdout output from py_wind.

    """
    cmd_file = f"{file_path}/.tmpcmds.txt"

    with Path.open(cmd_file, "w", encoding="utf-8") as file_out:
        for command in commands:
            file_out.write(f"{command}\n")

    # This isn't using `run_shell_command` because we also need to pass stdin,
    # which is not what we want to do with that function.
    with Path.open(cmd_file, "r", encoding="utf-8") as stdin:
        sh_out = run(["py_wind", root], stdin=stdin, capture_output=True, cwd=file_path, check=True)  # noqa: S603, S607

    Path(cmd_file).unlink()

    stdout, stderr = sh_out.stdout, sh_out.stderr
    if stderr:
        print(stderr.decode("utf-8"))  # noqa: T201

    return stdout.decode("utf-8").split("\n")
