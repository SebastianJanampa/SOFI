set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

python main.py --output_dir logs/$dataset/sofi --num_feature_levels 2 --batch_size 8 --print_freq 400 --lr 2e-4 --lr_backbone 2e-5 --dec_n_points 8 --enc_n_points 32 --resolution high --num_workers 4 --dataset  $dataset
