#!/bin/bash
#SBATCH --job-name=prometheus_genie
#SBATCH --output=prometheus_output_file_%j.out
#SBATCH --error=prometheus_err_file_%j.err
#SBATCH --partition=icecube
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=10G

echo "Starting the benchmark job..."


module load astro
module load hdf5/intel/1.10.4
module load intel/20.0.4


if [ -f /software/astro/anaconda/anaconda3-2020.11/etc/profile.d/conda.sh ]; then
    echo "Sourcing conda.sh..."
    source /software/astro/anaconda/anaconda3-2020.11/etc/profile.d/conda.sh
else
    echo "Could not find conda.sh!" >&2
    exit 1
fi

echo "Activating conda environment..."
conda activate myenv || { echo "Failed to activate conda environment"; exit 1; }

echo "Sourcing setup script..."
source /groups/icecube/jackp/setup_prometheus_and_genie.sh || { echo "Failed to source setup script"; exit 1; }


#cd /groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/examples
echo "Running benchmark script..."

python example_genie_external_root.py --simset 22 --rootfile /groups/icecube/jackp/genie_test_outputs/output_gheps/gntp_icecube_numu_100.gtac.root
echo "Benchmark job completed successfully!"

