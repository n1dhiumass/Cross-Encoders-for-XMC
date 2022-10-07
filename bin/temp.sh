#!/usr/bin/env bash

set +exu

##### Exact CrossEnc Eval
##for data in lego pro_wrestling forgotten_realms star_trek yugioh #coronation_street elder_scrolls ice_hockey muppets
#for data in lego pro_wrestling
##for data in doctor_who star_trek
#do
##    sbatch -p gpu --mem 32GB --job-name exact_1_${data} bin/run.sh \
##    python eval/run_exact_cross_encoder.py \
##    --top_k 64 \
##    --data_name $data \
##    --res_dir ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats_2-last.ckpt
#
##    sbatch -p gpu --mem 64GB --job-name nsw_1_${data} bin/run.sh \
##    python eval/analyze_nsw_graph.py --res_dir ../../results/4_Zeshel/hnsw_debug/score_mats --data_name $data --embed_type tfidf
##
##    sbatch -p gpu --mem 32GB --job-name nsw_2_${data} bin/run.sh \
##    python eval/analyze_nsw_graph.py --res_dir ../../results/4_Zeshel/hnsw_debug/score_mats --data_name $data --embed_type bienc
#
#    for res_dir_val in "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_5_bs_10_max_nbrs_500_budget_w_ddp/score_mats_model-1-11959.0-1.22.ckpt" "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_2_bs_10_max_nbrs_250_budget_w_ddp/score_mats_model-1-11159.0-1.15.ckpt" "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp/score_mats"
##    for res_dir_val in "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_63_negs_5_bs_10_max_nbrs_500_budget_w_ddp/score_mats_model-1-11959.0-1.22.ckpt"
#    do
#        sbatch -p gpu --mem 32GB --job-name nsw_3_${data} bin/run.sh \
#        python eval/analyze_nsw_graph.py --res_dir ${res_dir_val} --data_name $data --embed_type bienc
#
#        sbatch -p gpu --mem 32GB --job-name nsw_4_${data} bin/run.sh \
#        python eval/analyze_nsw_graph.py --res_dir ${res_dir_val} --data_name $data --embed_type tfidf
#    done
#
#done

##### Graph Analysis Eval
##for data in lego pro_wrestling forgotten_realms star_trek yugioh #coronation_street elder_scrolls ice_hockey muppets
#for data in lego pro_wrestling american_football doctor_who
##for data in american_football doctor_who
#do
#    for embed in tfidf
#    do
#    for res_dir_val in "../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_0-last.ckpt"
#    do
#
#    sbatch -p gpu --gres gpu:1 --mem 32GB --job-name answ_1_${data} --exclude gpu-0-0 bin/run.sh \
#    python eval/analyze_nsw_graph.py \
#    --res_dir ${res_dir_val} \
#    --data_name $data \
#    --embed_type ${embed} \
#    --bi_model_file ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt \
#    --misc std_bienc --plot_only 1
#
#    sbatch -p gpu --gres gpu:1 --mem 32GB --job-name answ_2_${data} --exclude gpu-0-0 bin/run.sh \
#    python eval/analyze_nsw_graph.py \
#    --res_dir ${res_dir_val} \
#    --data_name $data \
#    --embed_type ${embed} \
#    --bi_model_file ../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0/model/model-2-3010.0-1.94.ckpt  \
#    --misc small_distill_from_std_bienc --plot_only 1
#
#    sbatch -p gpu --gres gpu:1 --mem 32GB --job-name answ_2b_${data} --exclude gpu-0-0 bin/run.sh \
#    python eval/analyze_nsw_graph.py \
#    --res_dir ${res_dir_val} \
#    --data_name $data \
#    --embed_type ${embed} \
#    --bi_model_file ../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0/model/3-last.ckpt  \
#    --misc last_of_small_distill_from_std_bienc --plot_only 1
#
#    sbatch -p gpu --gres gpu:1 --mem 32GB --job-name answ_3_${data} --exclude gpu-0-0 bin/run.sh \
#    python eval/analyze_nsw_graph.py \
#    --res_dir ${res_dir_val} \
#    --data_name $data \
#    --embed_type ${embed} \
#    --bi_model_file    ../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_from_scratch/model/model-3-3412.0-2.04.ckpt \
#    --misc small_distill_from_scratch --plot_only 1
#
#    #        sbatch -p gpu --mem 32GB --job-name nsw_4_${data} bin/run.sh \
#    #        python eval/analyze_nsw_graph.py --res_dir ${res_dir_val} --data_name $data --embed_type tfidf
#    done
#    done
#done


##gpu --gres gpu:1 --mem 64GB
## NSW Eval
##for data in lego pro_wrestling
#for data in lego pro_wrestling # american_football doctor_who
##for data in american_football doctor_who
##for data in doctor_who
#do
#    for embed in bienc anchor
#    do
##        for entry_method in bienc random
#        for entry_method in bienc
#        do
#            for gtype in nsw knn
#            do
##                sbatch -p gpu --gres gpu:1 --mem 32GB --job-name ${gtype}_${data}_${embed}_100 --exclude gpu-0-0 bin/run.sh \
##                python eval/nsw_eval_zeshel.py \
##                --data_name $data \
##                --embed_type $embed \
##                --graph_metric l2 \
##                --bi_model_file    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt  \
##                --res_dir          ../../results/6_ReprCrossEnc/_analyze_graph_debug \
##                --graph_type $gtype \
##                --n_ment 100 \
##                --entry_method ${entry_method} \
##                --misc 100_ments
#
#
#                sbatch -p gpu --gres gpu:1 --mem 32GB --job-name ${gtype}_${data}_${embed}_all --exclude gpu-0-0 bin/run.sh \
#                python eval/nsw_eval_zeshel.py \
#                --data_name $data \
#                --embed_type $embed \
#                --graph_metric l2 \
#                --bi_model_file    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt  \
#                --res_dir          ../../results/6_ReprCrossEnc/_analyze_graph_debug \
#                --graph_type $gtype \
#                --n_ment -1 \
#                --entry_method ${entry_method} \
#                --misc all_ments --plot_only 1
#
#            done
#        done
#	done
#done


# Cross-Enc Matrix Computation
#for data in lego forgotten_realms star_trek yugioh #coronation_street elder_scrolls ice_hockey muppets
#for data in forgotten_realms star_trek yugioh #coronation_street elder_scrolls ice_hockey muppets
#for data in lego pro_wrestling forgotten_realms star_trek yugioh
#for data in lego pro_wrestling forgotten_realms star_trek yugioh american_football doctor_who
#for data in doctor_who star_wars
#for data in lego pro_wrestling american_football doctor_who
#for data in pro_wrestling
#for data in american_football fallout final_fantasy world_of_warcraft
#for data in doctor_who military starwars
#for data in lego pro_wrestling yugioh
#for data in yugioh
#do
##sbatch -p gpu --gres gpu:1 --mem 32GB --job-name mat1-ce-$data --exclude gpu-0-0 bin/run.sh \
#sbatch -p 2080ti-long --gres gpu:1 --mem 32GB --job-name mat1-ce-$data  bin/run.sh  \
#python eval/run_cross_encoder_for_ment_ent_matrix_zeshel.py \
#--n_ment 100 \
#--n_ent -1 \
#--batch_size 300 \
#--data_name $data \
#--layers final \
#--cross_model_ckpt ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/model/model-2-15999.0--79.46.ckpt \
#--res_dir          ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds/score_mats_model-2-15999.0--79.46.ckpt
#done
#

# Running cross-encoder eval with biencoder retrieval
#for data in lego forgotten_realms star_trek yugioh coronation_street elder_scrolls ice_hockey muppets
#for data in pro_wrestling
#for data in star_trek yugioh coronation_street
#for data in lego forgotten_realms star_trek yugioh
#for data in coronation_street elder_scrolls ice_hockey muppets american_football doctor_who fallout final_fantasy military pro_wrestling starwars world_of_warcraft
#for data in american_football doctor_who fallout final_fantasy military pro_wrestling starwars world_of_warcraft
for data in lego forgotten_realms star_trek yugioh coronation_street elder_scrolls ice_hockey muppets
do
        for ckpt in "model-2-15999.0--90.84.ckpt"
        do
#            sbatch -p 2080ti-long  --gres gpu:1 --mem 32GB --job-name cross_w_bi_$data bin/run.sh \
            sbatch -p gpu --gres gpu:1 --mem 32GB --job-name m2cross_w_bi_${data} --exclude gpu-0-0 bin/run.sh \
            python eval/run_cross_encoder_w_binenc_retriever_zeshel.py \
            --data_name $data \
            --n_ment -1 \
            --top_k 64 \
            --batch_size 1 \
            --bi_model_file    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt  \
            --res_dir          ../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_cls_w_lin_tfidf_hard_negs/eval \
            --cross_model_file ../../results/8_CUR_EMNLP/d=ent_link/m=cross_enc_l=ce_neg=precomp_s=1234_63_negs_w_cls_w_lin_tfidf_hard_negs/model/$ckpt \
            --run_exact_reranking_opt 0 \
            --run_rnr_opt 1 \
            --disable_wandb 1 \
            --misc ${ckpt}
        done
done


#
##            sbatch -p gpu --gres gpu:1 --mem 32GB --job-name mq0cross_self_${data} --exclude gpu-0-0 bin/run.sh \
#            sbatch -p 2080ti-long  --gres gpu:1 --mem 32GB --job-name mq0cross_w_bi_$data bin/run.sh \
#            python eval/run_e-crossencoder_eval.py \
#            --data_name $data \
#            --n_ment -1 \
#            --top_k 64 \
#            --batch_size 1 \
#            --res_dir    ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/eval \
#            --model_file ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_wo_lin_w_0.5_bi_cross_loss_0.5_mutual_distill_from_scratch/model/$ckpt \
#            --use_dummy_ment 0 \
#            --run_exact_reranking_opt 0 \
#            --misc ${ckpt}_w_self_retr

#            sbatch -p gpu --gres gpu:1 --mem 32GB --job-name m1cross_all_layer_w_bi_$data --exclude gpu-0-0 bin/run.sh \
#            sbatch -p 2080ti-long --gres gpu:1 --mem 32GB --job-name m2cross_all_layer_w_bi_$data bin/run.sh \
#            python eval/run_cross_encoder_w_binenc_retriever_zeshel.py \
#            --data_name $data \
#            --n_ment -1 \
#            --top_k 64 \
#            --batch_size 1 \
#            --bi_model_file    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/model-3-12039.0-2.17.ckpt  \
#            --res_dir          ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_all_layers/eval \
#            --cross_model_file ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_crossenc_w_embeds_all_layers/model/$ckpt \
#            --use_all_layers 1 \
#            --misc ${ckpt}_all_layers
#        done
#done

# m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_negs_w_cls_w_lin_4_epochs_reproduce_6_49/model/model-1-12279.0--80.14.ckpt
# m=cross_enc_l=ce_neg=precomp_s=1234_w_cls_w_lin_6_387/63_of_1000_reranked_bienc_negs/epoch_1/model-1-12279.0--77.46.ckpt
# m=cross_enc_l=ce_neg=precomp_s=1234_w_cls_w_lin_6_387/63_of_500_reranked_bienc_negs/epoch_1/eoe-1-last.ckpt



##
## Running bi-encoder eval
##for data in yugioh
#for data in lego forgotten_realms star_trek yugioh coronation_street elder_scrolls ice_hockey muppets
##for data in american_football doctor_who fallout final_fantasy military pro_wrestling starwars world_of_warcraft
#do
#    for ckpt in "model-3-12039.0-2.17.ckpt"
#    do
##        sbatch -p gpu --gres gpu:1 --mem 32GB --job-name bi_1_$data --exclude gpu-0-0 bin/run.sh \
#        sbatch -p 2080ti-short --gres gpu:1 --mem 32GB --job-name bi_1_${data}-500 bin/run.sh \
#        python eval/run_biencoder_eval_zeshel.py \
#        --data_name $data \
#        --n_ment -1 \
#        --top_k 1000 \
#        --batch_size 1 \
#        --model_ckpt ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/model/$ckpt \
#        --res_dir    ../../results/6_ReprCrossEnc/d=ent_link/m=bi_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_4_epochs_wp_0.01_w_ddp/eval \
#        --misc $ckpt
#    done
#
#
#done



##gpu --gres gpu:1 --mem 64GB
## Running nbrhood ranking eval
#for data in lego pro_wrestling forgotten_realms star_trek yugioh
##for data in lego pro_wrestling
#do
##    for embed in tfidf bienc
##    do
#        for scoremat in score_mats score_mats_wrt_final_model
#        do
##            for currdir in "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_wo_dp_bs_16_w_hard_bienc" "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc" "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_hard_bienc_small_train" "4_Zeshel/d=ent_link/m=cross_enc_l=ce_s=1234_hard_negs_63_bs_8_w_tfidf_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train_finetune_31" "5_CrossEnc/d=ent_link/m=cross_enc_l=margin_s=1234_nsw_negs_10_nbrs_4_paths_8_negs_w_tfidf_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_ce_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_rand_bienc" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_bienc" "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=hard_negs_w_rank_s=1234_10_maxnbrs_63_negs_w_bienc" "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_bienc_negs" "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_bienc_negs_small_train" "5_CrossEnc/d=ent_link/m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_256_bienc_negs_small_train_from_scratch "
#            for currdir in "5_CrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=nsw_graph_rank_s=1234_10_maxnbrs_3_distcut_63_negs_w_tfidf"
#            do
#                sbatch -p gpu --gres gpu:1 --mem 32GB --job-name ${data}_bienc_${scoremat} --exclude gpu-0-0 bin/run.sh \
#                python eval/run_exact_cross_encoder.py \
#                --data_name $data \
#                --embed_type tfidf \
#                --bi_model_config  ../../results/4_Zeshel/d=ent_link/m=bi_enc_l=ce_s=1234_hard_negs_wo_dp_bs_32/model/best_wrt_dev_9/wrapper_config.json  \
#                --res_dir ../../results/$currdir/${scoremat}
#
##                sbatch -p cpu --mem 32GB --job-name ${data}_tfidf_${scoremat} bin/run.sh \
##                python eval/run_exact_cross_encoder.py \
##                --data_name $data \
##                --embed_type tfidf \
##                --res_dir ../../results/$currdir/${scoremat}
#            done
#        done
##	done
#done


## Running cross-encoder eval using NSW
##for data in lego forgotten_realms star_trek yugioh coronation_street elder_scrolls ice_hockey muppets
#for data in coronation_street elder_scrolls ice_hockey muppets american_football doctor_who fallout final_fantasy pro_wrestling starwars world_of_warcraft military
##for data in american_football doctor_who fallout final_fantasy military pro_wrestling starwars world_of_warcraft
#do
#    sbatch -p gpu --gres gpu:1 --mem 32GB --job-name ${data}_500_ce_nsw --exclude gpu-0-0 bin/run.sh \
#    python eval/run_cross_encoder_w_nsw_eval.py \
#    --data_name $data \
#    --n_ment -1 \
#    --cross_model_file ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/model/model-2-17239.0-1.22.ckpt \
#    --res_dir          ../../results/6_ReprCrossEnc/d=ent_link/m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/eval \
#    --bi_model_file    ../../results/7_EntModel/d=ent_link/m=bi_enc_l=ce_neg=distill_s=1234_distill_w_64_negs_wrt_cross_id_6_82_0_all_data/model/model-3-12318.0-1.92.ckpt  \
#    --embed_type bienc \
#    --max_nbrs 10 \
#    --beamsize 5 \
#    --top_k 100 \
#    --comp_budget 500
#done
#

##### Run graph analysis
##for data in lego pro_wrestling
##for data in lego
#for data in pro_wrestling
#do
##    for curdir in   "m=cross_enc_l=ce_neg=bienc_hard_negs_w_rerank_s=1234_63_from_500_negs_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_model-1-12279.0--79.47.ckpt" \
##                    "m=cross_enc_l=ce_neg=bienc_nsw_search_s=1234_64_negs_5_bs_10_max_nbrs_500_budget_w_ddp_w_best_wrt_dev_mrr_cls_w_lin_rtx/score_mats_model-1-10959.0--78.85.ckpt" \
##                    "m=cross_enc_l=rank_margin_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs/score_mats_model-2-17239.0-1.22.ckpt" \
##                    "m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt" \
##                    "m=cross_enc_l=ce_neg=bienc_hard_negs_s=1234_63_hard_negs_w_bienc_w_ddp_w_best_wrt_dev_mrr_cls_w_lin/score_mats_model-1-11359.0--80.19.ckpt"
#
##    for curdir in  "m=cross_enc_l=rank_ce_neg=bienc_hard_negs_w_knn_rank_s=1234_63_negs_w_ddp_w_cls_w_lin_d2p_neg_lin/score_mats_model-1-12279.0-3.21.ckpt"
###    do
###        for embed in bienc anchor tfidf
###        do
###            sbatch -p gpu --gres=gpu:1 --mem 32GB --job-name answ_all_${data}_${embed} --exclude gpu-0-0 bin/run.sh \
###            python eval/analyze_nsw_graph.py \
###            --res_dir ../../results/6_ReprCrossEnc/d=ent_link/$curdir \
###            --data_name $data \
###            --embed_type $embed \
###            --n_ment -1 \
###            --misc all_ments
###
###            sbatch -p gpu --gres=gpu:1 --mem 32GB --job-name answ_100_${data}_${embed} --exclude gpu-0-0 bin/run.sh \
###            python eval/analyze_nsw_graph.py \
###            --res_dir ../../results/6_ReprCrossEnc/d=ent_link/$curdir \
###            --data_name $data \
###            --embed_type $embed
###        done
###    done
#    for embed in bienc anchor tfidf
#    do
##        for gtype in "nsw" "knn" "knn_e2e"
#        for gtype in "knn_e2e"
#        do
#            sbatch -p gpu --gres=gpu:1 --mem 32GB --job-name answ_ip_${data}_${embed}_${gtype} --exclude gpu-0-0 bin/run.sh \
#            python eval/analyze_nsw_graph.py \
#            --res_dir  ../../results/6_ReprCrossEnc/_knn_debug \
#            --data_name $data \
#            --embed_type $embed \
#            --graph_type $gtype \
#            --nsw_metric inner_prod \
#            --misc inner_prod
#        done
#
##        sbatch -p gpu --gres=gpu:1 --mem 32GB --job-name answ_l2_${data}_${embed}_${gtype} --exclude gpu-0-0 bin/run.sh \
##        python eval/analyze_nsw_graph.py \
##        --res_dir  ../../results/6_ReprCrossEnc/_knn_debug \
##        --data_name $data \
##        --embed_type $embed \
##        --graph_type nsw \
##        --nsw_metric l2 \
##        --misc l2
#    done
#done

unset exu