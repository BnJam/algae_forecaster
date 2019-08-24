#SBATCH --account=rpp-ycoady
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=4000M
#SBATCH --time=00:10:00

python3 framework/test.py