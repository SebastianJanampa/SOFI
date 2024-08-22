python benchmark.py --model mscc --batch_size 4 --dataset holicity

python benchmark.py --model mscc --batch_size 4 --dataset hlw

python benchmark.py --model mscc --batch_size 4 --dataset gsv

python benchmark.py --batch_size 4 --dec_n_points 8 --enc_n_points 32 --resolution high --num_feature_levels 2 --resume logs/gsv/sofi/checkpoint0028.pth --dataset holicity

python benchmark.py --batch_size 4 --dec_n_points 8 --enc_n_points 32 --resolution high --num_feature_levels 2 --resume logs/gsv/sofi/checkpoint0028.pth --dataset hlw

python benchmark.py --batch_size 4 --dec_n_points 8 --enc_n_points 32 --resolution high --num_feature_levels 2 --resume logs/gsv/sofi/checkpoint0028.pth --dataset gsv

python benchmark.py --model ctrlc --batch_size 4 --dataset holicity

python benchmark.py --model ctrlc --batch_size 4 --dataset hlw

python benchmark.py --model ctrlc --batch_size 4 --dataset gsv
