CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=1 --num_heads=1
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=1 --num_heads=2
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=1 --num_heads=3
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=1 --num_heads=4

CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=2 --num_heads=1
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=3 --num_heads=1
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=4 --num_heads=1

CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=2 --num_heads=2
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=3 --num_heads=3
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=4 --num_heads=4
CUDA_VISIBLE_DEVICES=3 python main_new.py --dataset ZuCo --task SA --level sentence --modality fusion --model transformer --batch_size 64 --inference 0 --dev 0 --loss CE --epochs 200 --device cuda --num_layers=5 --num_heads=5
