batch_size=50
val_batch_size=50
mse_mult=10000
rec_mult=1
cyc_mult=1
num_epochs=200
subj_source="1 2 5"
subj_target=7
length=4000
model_name="mindbridge_subj125_a"$subj_target"_l"$length
load_from="../train_logs/mindbridge_subj125/last.pth"
gpu_id=0

cd src/

# accelerate launch --multi_gpu --num_processes 4 --gpu_ids 0,1,2,3 --main_process_port 29503 \
CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
main.py \
--model_name $model_name --subj_target $subj_target --subj_source $subj_source \
--num_epochs $num_epochs --batch_size $batch_size --val_batch_size $val_batch_size \
--h_size 2048 --n_blocks 4 --pool_type max --pool_num 8192 \
--mse_mult $mse_mult --rec_mult $rec_mult --cyc_mult $cyc_mult \
--eval_interval 10 --ckpt_interval 10 \
--load_from $load_from --num_workers 8 \
--max_lr 1.5e-4 --adapting --length $length
