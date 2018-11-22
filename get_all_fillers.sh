model_path=$1
ckpt=$2
vocab_path=$3
name=$4
model=$5
output_dir='saved_embeddings/1b'
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

size=200000
awk -v s=$((size+4)) 'NR>=5 && NR<=s {print $1}' $vocab_path > top${size}.wl
echo getting fillers for size $size
if [[ $model = 'baseline'  ]]; then
    python write_fillers.py  -mp $model_path -ckpt $ckpt -vp $vocab_path -op 1b-${name}-${size}-baseline -od 1b -m baseline -d 650  -wl top${size}.wl -ws $size -vs 210000
else
    python write_fillers.py  -mp $model_path -ckpt $ckpt -vp $vocab_path -op 1b-${name}-${size}-hrr-e -od 1b -m e -d 650  -wl top${size}.wl -ws $size -vs 210000 
    python write_fillers.py  -mp $model_path -ckpt $ckpt -vp $vocab_path -op 1b-${name}-${size}-hrr-f -od 1b -m f -d 650  -wl top${size}.wl -ws $size -vs 210000 
fi

