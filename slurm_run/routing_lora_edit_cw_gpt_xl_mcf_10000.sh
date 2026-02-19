#!/bin/bash
#SBATCH --job-name=routing_lora_cw
#SBATCH --nodelist=ubuntu
#SBATCH --output=logs/routing_lora_edit_cw_gpt_xl_mcf_step10000_%j.out
#SBATCH --error=logs/routing_lora_edit_cw_gpt_xl_mcf_step10000_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

cd /home/leeg/Experiments/multi-lora-editing

# UV 가상환경 활성화
source .venv/bin/activate

echo "시작: $(date) routing_lora_edit_cw_gpt_xl_mcf_step10000"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m experiments.evaluate \
    --alg_name=Routing_LoRA_Edit_CW \
    --model_name="gpt2-xl" \
    --hparams_fname="gpt2-xl.json" \
    --ds_name="mcf" \
    --dataset_size_limit="10000" \
    --num_edits="100" \


echo ""
echo "=== 실험 완료 ==="
echo "완료: $(date)"
