import numpy as np
from subprocess import Popen, PIPE

sh = "squeue -u $USER -o '%A'"
stdout, stderr = Popen(sh, stdout=PIPE, stderr=PIPE, shell=True).communicate()
job_ids = stdout.decode("utf-8").split()[1:]
print(job_ids)

for job in job_ids:
    sh = "scancel " + job
    stdout, stderr = Popen(sh, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    if stderr:
        print(stderr.decode("utf-8")

