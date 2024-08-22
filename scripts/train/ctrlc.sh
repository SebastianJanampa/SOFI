set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

python main.py --output_dir logs/$dataset/ctrlc --num_feature_levels 1 --batch_size 32 --model ctrlc \
--lr 0.0001 --lr_backbone 0.0001  --dataset $dataset


