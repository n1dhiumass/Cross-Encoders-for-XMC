#!/usr/bin/env bash

set +exu


sleep 3600
sleep 3600

squeue -u nishantyadav,nmonath,rangell > jobs_running.txt
date > launch_time.txt

for data in american_football doctor_who fallout final_fantasy military pro_wrestling starwars world_of_warcraft coronation_street elder_scrolls ice_hockey muppets
do
sbatch -p 2080ti-long --gres gpu:1 --mem 32GB --job-name e2e-ece0-$data bin/run.sh  \
bin/run.sh \
python eval/run_cross_encoder_for_ent_ent_matrix.py \
--data_name $data \
--n_ent_x -1 \
--n_ent_y -1 \
--topk 100 \
--embed_type bienc \
--token_opt m2e \
--batch_size 150 \
--res_dir          ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_e2e_graph/score_mats_0-last.ckpt \
--cross_model_file ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_e2e_graph/model/0-last.ckpt \
--bi_model_file    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt

done



unset exu