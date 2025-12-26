#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=server2
#SBATCH --output=logs/memit_gpt_xl_mcf_step2000_%j.out
#SBATCH --error=logs/memit_gpt_xl_mcf_step2000_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd /home/leeg/Editing-Code/AlphaEdit

# UV 가상환경 활성화
source .venv/bin/activate

echo "시작: $(date) memit_gpt_xl_mcf_step2000"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m experiments.evaluate \
    --alg_name=MEMIT \
    --model_name="gpt2-xl" \
    --hparams_fname="gpt2-xl.json" \
    --ds_name="mcf" \
    --dataset_size_limit="2000" \
    --num_edits="100" \

echo ""
echo "=== 실험 완료 ==="
echo "완료: $(date)"
