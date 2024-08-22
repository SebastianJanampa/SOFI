set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

output=outputs/${dataset}/mscc
epoch=("029" "028")

for ((i=0;i<${#epoch[@]};++i)); do
	python main.py --output_dir $output --num_feature_levels 4 --model mscc  --resume logs/gsv/mscc/checkpoint0${epoch[i]}.pth --append_word ${epoch[i]} --eval  --dataset $dataset
done
