#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The purpose of this is to mass delete a bunch of wind_save files using a
standard name, i.e. python_01.wind_save.
"""

import argparse as ap
from subprocess import Popen, PIPE
from typing import List
from PyPython import PythonUtils as Utils


def delete_windsaves(wdpf: List[str], root) -> None:
    """
    Deletes the wind_saves. Possibly not portable because of the use of
    subprocess to use rm, instead of using a "proper" library to do it.
    """

    for i in range(len(wdpf)):
        pf, wd = Utils.split_root_directory(wdpf[i])
        cmd = "cd {}; rm {}*.wind_save".format(wd, root)
        sh = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = sh.communicate()
        if stderr:
            print(stderr.decode("utf-8"))

    return


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("root",
                   help="The root name of the wind_save files.")
    args = p.parse_args()
    delete_windsaves(Utils.get_pfs(args.root), args.root)
