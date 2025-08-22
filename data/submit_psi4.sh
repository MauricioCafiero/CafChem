#!/bin/bash

if [ -z "$1" ]; then
    echo No input filname provided!
else    
    input=$1
    basename=${input%.*}
    slurmname="$basename.slurm"
    output="$basename.out"
    
    if [ -z "$2" ]; then
        echo No number of cores provided, using 32
	procs_in=32
    else
        procs_in=$2
    fi
    procs=$(($procs_in / 2))
    mem=${procs_in}
    
    cp ~/scripts/template.slurm $slurmname
    sed -i "s/tasks=/tasks=$procs/" $slurmname
    sed -i "s/mem=/mem="$mem"GB/" $slurmname
    sed -i "s/name=/name=$basename/" $slurmname
    sed -i "s/cores = [0-9][0-9]*[0-9]*/cores = $mem/" $input
    sed -i "s/p4_mem = [0-9][0-9]*[0-9]*/p4_mem = "$mem"/" $input

    echo "python3.11 $input > $output" >> $slurmname
    sbatch $slurmname
    echo "$basename submitted"
fi
