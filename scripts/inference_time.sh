set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

python benchmark.py --model sofi --batch_size 1 --dec_n_points 8 --enc_n_points 32 --resolution high --num_feature_levels 2 --resume logs/gsv/sofi/checkpoint0028.pth --dataset $dataset

