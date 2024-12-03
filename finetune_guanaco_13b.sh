python qlora/qlora.py \
    --model_name_or_path huggyllama/llama-13b \
    --output_dir ./output/guanaco-13b \
    --logging_steps 10 \
    --save_strategy steps \
    --data_seed 42 \
    --save_steps 100 \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --early_stopping_patience 3 \
    --eval_dataset_size 1024 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 16 \
    --max_new_tokens 32 \
    --dataloader_num_workers 16 \
    --group_by_length \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bf16 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset oasst1 \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 1875 \
    --eval_steps 100 \
    --learning_rate 0.0001 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.05 \
    --weight_decay 0.0 \
    --seed 0
