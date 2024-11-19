set -e

if [[ $# -eq 0 ]] ; then
    echo 'Require Experiment Name'
    exit 1
fi
dataset=$1

output=outputs/${dataset}/sofi
epoch=("028" "029")

for ((i=0;i<${#epoch[@]};++i)); do
	python main.py --output_dir $output --num_feature_levels 2 --resume  logs/gsv/sofi/checkpoint0${epoch[i]}.pth --append_word ${epoch[i]} --eval --dec_n_points 8 --enc_n_points 32 --resolution high --dataset $dataset
done
