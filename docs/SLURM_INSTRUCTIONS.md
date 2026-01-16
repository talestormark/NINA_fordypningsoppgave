# Running jobs

When you want to run your large project on the cluster you must submit it to SLURM as a job. It is then queued along with other submitted jobs, and will be run when the computer resources are available. You can get e-mail notifications for when the job is started, when it is completed or if it fails. More in-depth documentation can be found at the Sigma2 webpage.

The first step is to create a SLURM job script. It is essentially a bash script with some extra information for SLURM. An example of a functioning script:

```bash
#!/bin/sh
#SBATCH --account=share-ie-idi    # Account for IE-IDI department
#SBATCH --job-name=example_job
#SBATCH --time=0-02:30:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1              # Setting the number of GPUs to 1
#SBATCH --constraint=gpu80g       # Requesting 80GB GPU memory
#SBATCH --mem=100G                # Asking for 100GB RAM
#SBATCH --nodes=1
#SBATCH --output=outputs/slurm_outputs/%j_output.txt  # %j = job ID
#SBATCH --error=outputs/slurm_outputs/%j_error.err

#SBATCH --mail-user=tmstorma@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2024.02-1
source activate <your_conda_env>
python /cluster/home/tmstorma/<project_folder>/<your_script>.py
```

Let's break down this example. The first line, `#!/bin/sh` tells the computer that this is a bash script. The next lines are all commented out, but they are still read by SLURM and change the environment variables.

| Item | Description |
|------|-------------|
| partition | Whether your script needs CPU or GPU. |
| account | The billing account connected to running the script (e.g., `share-ie-idi` for IE-IDI department) |
| time | The amount of time allocated for your job. The task fails if your program runs for longer than this time limit. |
| nodes | The number of nodes. |
| ntasks-per-node | The number of tasks per node. |
| mem | The amount of memory allocated for your job. The task fails if your program exceeds this limit. |
| gres | Generic resource scheduling - used to request GPUs (e.g., `gpu:1` for one GPU) |
| constraint | Additional hardware constraints (e.g., `gpu80g` for 80GB GPU memory) |
| job-name | The name of the job. |
| output | The file in which to save all printed outputs. Use `%j` for job ID in path. |
| error | The file in which to save all printed errors. Use `%j` for job ID in path. |
| mail-user | The email to send notifications to. |
| mail-type | Which emails to send. |

## Running your SLURM script

Once you have a script that you want to run, e.g. `example_script.slurm`, you can run the script by typing `sbatch example_script.slurm`. This will start your script. This is an example of what this looks like:

```bash
[jssaethe@idun-login1 jssaethe]$ sbatch example_script.slurm
Submitted batch job 18915035
```

## Checking the SLURM queue

Once you have started your script, you can check whether it has started or not and/or how long it has been running by using the command `squeue`. This will show the entire queue. Therefore, if you only want to see your personal entries you can add the `-u` flag, followed by your username. An example is:

```bash
[jssaethe@idun-login1 jssaethe]$ squeue -u jssaethe
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          18915029      GPUQ example_ jssaethe PD       0:00      1 (Priority)
```

## Canceling your SLURM job

If you want to cancel your SLURM job, you can do this using the `scancel` command, followed by the JOBID of your job. Following the example above, you would type `scancel 18915029` to stop the listed job.