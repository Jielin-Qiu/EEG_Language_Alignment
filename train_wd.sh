CUDA_VISIBLE_DEVICES=1 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss WD --epochs 200 --device cuda
# CUDA_VISIBLE_DEVICES=1 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --checkpoint transformer_fusion_sentence_1_1_8_16_64_WD.chkpt --batch_size 1 --inference 1 --dev 0 --loss WD --epochs 200 --device cuda
