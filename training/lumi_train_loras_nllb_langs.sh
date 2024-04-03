#!/bin/bash
#SBATCH --job-name=train_loras_nllb_langs
##SBATCH --account=project_2001194 #PUHTI version
#SBATCH --account=project_462000447
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=small-g
##SBATCH --partition=gpu #PUHTI version
#SBATCH --gpus-per-node=4


module use /appl/local/csc/modulefiles/ #LUMI version
module load pytorch #LUMI version
##module load pytorch/2.1 #PUHTI version

rm -rf ~/.cache
pip uninstall -y adaptor
pip install -e ./adaptor[generative]
pip install -r training/requirements.txt

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/training"

export WANDB_API_KEY=XXX
export WANDB_PROJECT=mammoth-lora

# puhti data config:
export DATA_DIR=/scratch/project_2001194/tiedeman/Tatoeba-Challenge/data/release/v2023-09-26
export HF_MODEL_ID=google/mt5-base
export TARGET_LANGS=epo,est,eus,ewe,fao,fij,fin,fur
##export TARGET_LANGS=fur

## To run in parallel, single-node:
## torchrun --nnodes 1 --nproc-per-node=4 --rdzv-endpoint=localhost:0 --rdzv-backend=c10d training/train_loras_nllb_langs.py --base_data_dir ${DATA_DIR} --base_model ${HF_MODEL_ID} --target_langs ${TARGET_LANGS}
srun python training/train_loras_nllb_langs.py --base_data_dir ${DATA_DIR} --base_model ${HF_MODEL_ID} --target_langs ${TARGET_LANGS}