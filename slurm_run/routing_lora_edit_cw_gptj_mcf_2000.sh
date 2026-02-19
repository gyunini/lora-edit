#!/bin/bash
#SBATCH --job-name=routing_lora_cw
#SBATCH --nodelist=ubuntu
#SBATCH --output=logs/routing_lora_edit_cw_gptj_mcf_step2000_%j.out
#SBATCH --error=logs/routing_lora_edit_cw_gptj_mcf_step2000_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

cd /home/leeg/Experiments/multi-lora-editing

# UV 가상환경 활성화
source .venv/bin/activate

echo "시작: $(date) routing_lora_edit_cw_gptj_mcf_step2000"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m experiments.evaluate \
    --alg_name=Routing_LoRA_Edit_CW \
    --model_name="EleutherAI/gpt-j-6B" \
    --hparams_fname="EleutherAI_gpt-j-6B.json" \
    --ds_name="mcf" \
    --dataset_size_limit="2000" \
    --num_edits="100" \


echo ""
echo "=== 실험 완료 ==="
echo "완료: $(date)"
