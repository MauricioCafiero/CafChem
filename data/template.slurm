#!/bin/bash
#SBATCH --job-name=    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=m.cafiero@reading.ac.uk     # Where to send mail	
#SBATCH --ntasks=                   # Run on a single CPU
#SBATCH --mem=                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --nodes=1               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
pwd; hostname; date

echo "Running plot script on a single CPU core"


