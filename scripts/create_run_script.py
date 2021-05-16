#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv
from typing import List

from pypython import util
"""
#!/bin/bash

declare -a directories=(
    "Directory1"
    "Directory2"
    "Directory3"
    "Directory4"
    "Directory5"
)

cwd=$(pwd)
for i in "${directories[@]}"
do
    cd $i
    pwd
    # commands
    cd ..
done
"""


def create_run_script(commands):
    """Create the run script given a list of commands."""

    # Find any python parameter file in the directory and subdirectories
    directories = []
    pfs = util.get_parameter_files()
    for pf in pfs:
        root, directory = util.get_root_from_filepath(pf)
        directories.append(directory)

    # Construct the file as shown in the doc string of the script
    file = "#!/bin/bash\n\ndeclare -a directories=(\n"
    for d in directories:
        file += "\t\"{}\"\n".format(d)
    file += ")\n\ncwd=$(pwd)\nfor i in \"${directories[@]}\"\ndo\n\tcd $i\n\tpwd\n"
    if len(commands) > 1:
        for k in range(len(commands) - 1):
            file += "\t{}\n".format(commands[k + 1])
    else:
        file += "\t# commands\n"
    file += "\tcd $cwd\ndone\n"

    # Print the file and write it to the current directory
    print(file)
    with open("commands.sh", "w") as f:
        f.write(file)

    return


if __name__ == "__main__":
    create_run_script(argv)
