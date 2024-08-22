set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

python main.py --output_dir logs/$dataset/mscc --num_feature_levels 2 --batch_size 8 --model mscc \
--lr 0.0001 --lr_backbone 0.0001 --print_freq 800  --frozen_weights logs/$dataset/ctrlc/checkpoint.pth --dataset $dataset
