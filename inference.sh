CUDA_VISIBLE_DEVICES=1 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 2 --device cuda --checkpoint transformer_fusion_sentence_4_4_64_CCA.chkpt --inference 1 --dev 0 --loss CCA --num_layers=4 --num_heads=4