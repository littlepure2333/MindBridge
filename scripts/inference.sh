subj_load=1
subj_test=1
model_name="mindbridge_subj125"
ckpt_from="last"
text_image_ratio=0.5
guidance=5
gpu_id=0

cd src/

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
recon.py \
--model_name $model_name --ckpt_from $ckpt_from \
--h_size 2048 --n_blocks 4 --pool_type max \
--subj_load $subj_load --subj_test $subj_test \
--text_image_ratio $text_image_ratio --guidance $guidance \
--recons_per_sample 8 
# --test_end 2 \
# --test_start 0 \


results_path="../train_logs/"$model_name"/recon_on_subj"$subj_test

CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore \
eval.py --results_path $results_path
