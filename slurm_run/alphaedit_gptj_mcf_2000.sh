#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=server2
#SBATCH --output=logs/alphaedit_gptj_mcf_step2000_%j.out
#SBATCH --error=logs/alphaedit_gptj_mcf_step2000_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd /home/leeg/Editing-Code/AlphaEdit

# UV 가상환경 활성화
source .venv/bin/activate

echo "시작: $(date) alphaedit_gptj_mcf_step2000"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name="EleutherAI/gpt-j-6B" \
    --hparams_fname="EleutherAI_gpt-j-6B.json" \
    --ds_name="mcf" \
    --dataset_size_limit="2000" \
    --num_edits="100" \


echo ""
echo "=== 실험 완료 ==="
echo "완료: $(date)"
