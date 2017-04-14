#!/bin/bash
#SBATCH -A xpress
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -C quad,flat
#SBATCH -c 64
#SBATCH -p normal
#SBATCH -J octotiger_node_level_experiments
#SBATCH -t 02:00:00

#
# Scaling experiment on a single node
# Meant to be used to run different thread values within a single job
#

name=$1
level=$2
threads=$3
threads_increment=$4

# Add these in case of crashes: --hpx:ini=hpx.stacks.small_size=0xC0000 -Ihpx.stacks.use_guard_pages=0

while [[ ${threads} -le 64 ]]; do
    # srun ./mic-knl-gcc-build/octotiger-gb17-tcmalloc-Release/octotiger -Disableoutput -Problem=dwd -Max_level=${level} -Xscale=36.0 -Eos=wd -Angcon=1 -Stopstep=30 --hpx:threads=${threads} -Restart=restart${level}.chk > results/${name}_N${SLURM_NNODES}_t${threads}_l${level} 2>&1
    srun ./mic-knl-gcc-build/octotiger-gb17-tcmalloc-Release/octotiger --hpx:threads=${threads} -Ihpx.parcel.message_handlers=0 -Ihpx.stacks.use_guard_pages=0 -Ihpx.max_busy_loop_count=500 -Ihpx.max_background_threads=16 -Ihpx.parcel.max_connections=20000 -Problem=dwd -Max_level=${level} -Xscale=4.0 -Eos=wd -Angcon=1 -Disableoutput -Stopstep=9 -Restart=restart${level}_gb17.chk > results/${name}_N${SLURM_NNODES}_t${threads}_l${level}_m0 2>&1

    echo ${threads}
    let threads=threads+${threads_increment}
done
