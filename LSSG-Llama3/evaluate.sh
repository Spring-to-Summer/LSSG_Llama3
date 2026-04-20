MODEL=$1
DEVICE=${2:-0}

#CONDA_BASE=$(conda info --base)
#source $CONDA_BASE/etc/profile.d/conda.sh
#conda activate eval

lm_eval --model hf \
    --model_args pretrained="${MODEL}" \
    --tasks logiqa2 \
    --device "cuda:${DEVICE}" \
    --batch_size auto:8 \
    --output_path eval \
    --log_samples \
    --limit 100000


lm_eval --model hf \
    --model_args pretrained="${MODEL}" \
    --tasks ai2_arc \
    --device "cuda:${DEVICE}" \
    --batch_size auto:8 \
    --output_path eval \
    --log_samples \
    --limit 100000

lm_eval --model hf \
    --model_args pretrained="${MODEL}" \
    --tasks mmlu \
    --device "cuda:${DEVICE}" \
    --num_fewshot 5 \
    --batch_size auto:8 \
    --output_path eval \
    --log_samples \
    --limit 100000
