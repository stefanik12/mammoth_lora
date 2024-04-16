#!/bin/bash
#SBATCH --job-name=train_loras_nllb_langs
#SBATCH --account=project_2001194
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpusmall
##SBATCH --out=gpu.%J.log
##SBATCH --err=gpu.%J.log
##SBATCH --mail-type=BEGIN #uncomment to enable mail
#SBATCH --gres=gpu:a100:1

module load pytorch/2.1

# rm -rf ~/.cache
# pip uninstall -y adaptor
pip install -e ./adaptor[generative]
pip install -r training/requirements.txt

##export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/training"

export WANDB_API_KEY=66661710847db589f9d9dce6fab050f8804d8a6e
export WANDB_PROJECT=mammoth-lora

# puhti data config:
export DATA_DIR=/scratch/project_2001194/mstefani/pieceof-Tatoeba-Challenge/data/release/v2023-09-26
export BASE_MODEL_ID=facebook/nllb-200-distilled-600M
## export TARGET_LANGS=epo,est,eus,ewe,fao,fij,fin,fon,fra,fur,gla,gle,glg,grn,guj,hat,hau,heb,hin,hne

export CUDA_VISIBLE_DEVICES=0
srun python training/train_loras_nllb_langs.py --eval_run True --eval_batches 500 --use_language_prefixes True --base_data_dir ${DATA_DIR} --base_model ${BASE_MODEL_ID} --target_langs ${TARGET_LANGS} --pair_evaluation_langs 'fin,fra;est,gla' --baseline_training True --checkpoint_dir /scratch/project_2001194/mstefani

