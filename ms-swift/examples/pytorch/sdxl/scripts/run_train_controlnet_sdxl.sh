PYTHONPATH=../../.. \
accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path="AI-ModelScope/stable-diffusion-xl-base-1.0" \
 --output_dir="train_controlnet_sdxl" \
 --dataset_name="AI-ModelScope/controlnet_dataset_condition_fill50k" \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --max_train_steps=15000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --validation_steps=100 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --report_to="tensorboard" \
 --seed=42 \
