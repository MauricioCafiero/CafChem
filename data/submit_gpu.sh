#!/bin/bash

if [ -z "$1" ]; then
    echo No input filname provided!
else    
    input=$1
    basename=${input%.*}
    slurmname="$basename.slurm"
    outname="$basename.out"
    
    if [ -z "$2" ]; then
        echo No memory requirment provided, using 10G
	mem_in=10
    else
        mem_in=$2
    fi
    
    cp ~/scripts/gpu.slurm $slurmname
    sed -i "s/mem=/mem="$mem_in"G/" $slurmname
    sed -i "s/job-name=/job-name=$basename/" $slurmname

    echo "python3.12 $input > $outname " >> $slurmname
    sbatch $slurmname
    echo "$basename submitted"
fi
