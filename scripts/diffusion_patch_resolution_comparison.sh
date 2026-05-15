#!/bin/bash
# Compare diffusion patch attack performance across patch resolutions.
# Only SS model attacked, no patch projection, 1 epoch, small run.
# Uses image init (dog.jpg) for coherent patch generation.

cd "$(dirname "$0")/.."

PATCH_SIZES=(300)

BASE_IMAGE="test_assets/dog.jpg"
PROMPT="a dog"

# Diffusion settings: reverse_steps = source_steps so denoising starts
# from full noise — required for coherent output with random init, and
# gives the optimizer maximum latent freedom with image init.
SOURCE_STEPS=50
REVERSE_STEPS=50

for SIZE in "${PATCH_SIZES[@]}"; do
    EXPERIMENT="diffusion_ss_patch_size_${SIZE}"
    echo "============================================"
    echo "Running patch size: ${SIZE}x${SIZE}  experiment: ${EXPERIMENT}"
    echo "============================================"

    python3 run_patch_attack.py \
        --patch_parametrization diffusion \
        --attack_ss \
        --loss_weights 0 0 1 \
        --patch_size "${SIZE}" \
        --n 1 \
        --small_run \
        --experiment_name "${EXPERIMENT}" \
        --diffusion_init_mode image \
        --diffusion_base_image "${BASE_IMAGE}" \
        --diffusion_prompt "${PROMPT}" \
        --diffusion_latent_eps 1.0 \
        --diffusion_guidance_scale 3.0 \
        --ss_model pspnet_cityscapes \
        --save_artifacts
done

echo "============================================"
echo "All runs complete. Compare results in MLflow:"
echo "  mlflow ui --backend-store-uri mlruns"
echo "============================================"
