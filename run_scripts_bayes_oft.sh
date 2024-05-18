export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/${selected_subject}"
export CLASS_DIR="./data/class_data/${class_token}"
export OUTPUT_DIR="checkpoints/bayes_oft/${name}"
export instance_prompt="a photo of ${unique_token} ${class_token}"
export class_prompt="a photo of ${class_token}"
export path="logs/bayes_oft/${name}"

seed=0
sigma=0.01
testsigma=0.0
step=1200
selected_subject="backpack"
class_token="backpack"
name="${selected_subject}"
unique_token="qwe"


# Training models
accelerate launch train_dreambooth_bayes_oft.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=$INSTANCE_DIR \
    --class_data_dir="$CLASS_DIR" \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="$instance_prompt" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --class_prompt="$class_prompt" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=${step} \
    --learning_rate=1e-4 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=${step} \
    --use_8bit_adam \
    --seed=${seed} \
    --name="$name" \
    --num_class_images=200 \
    --convert_up_block \
    --convert_subblock_0 \
    --convert_subblock_1 \
    --convert_subblock_2 \
    --convert_subblock_3 \
    --convert_ff \
    --convert_attn2 \
    --convert_linear \
    --init_sigma ${sigma} \
    --prior_sigma ${sigma} \
    --lam 0.0 \
    --eps=6e-5 \
    --r=4 \



# Generating images based on trained model
accelerate launch test_dreambooth.py \
    --model_path="${OUTPUT_DIR}/checkpoint-${step}" \
    --output_path="./${path}/${name}src/"  \
    --class_token="${class_token}" \
    --oft \
    --test_sigma=${testsigma} \


# Evaluate the generation result
python utils/process.py --data_path="./${path}/${name}"
python eval_updated_v2.py --image_dir=${path}" --json_name jsons/metadata_1_${selected_subject}.json --subject_name ${selected_subject}

