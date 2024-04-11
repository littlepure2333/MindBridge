batch_size=50
val_batch_size=50
num_epochs=600
mse_mult=10000
subj=1
model_name="mindbridge_subj"$subj

cd src/

# CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 --main_process_port 29501 \
main.py \
--model_name $model_name --subj_list $subj \
--num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--h_size 2048 --n_blocks 4 --pool_type max --pool_num 8192 \
--mse_mult $mse_mult \
--eval_interval 10 --ckpt_interval 10 \
--max_lr 3e-4 --num_workers 4