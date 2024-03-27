#SBATCH --job-name=train_loras_nllb_langs
#SBATCH --account=project_462000447
#SBATCH --time=00:01:00
#SBATCH --mem-per-cpu=2G
#SBATCH --partition=small-g
##SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --gpus-per-node=1

module use /appl/local/csc/modulefiles/
module load pytorch
pip install -r training/requirements.txt

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

srun python training/train_loras_nllb_langs.sh
