#!/bin/bash
### 多进程模型infer需改动信息

# funs
doMakeDir(){
    path=$1
    if [ ! -d $path ]
    then 
        mkdir -p $path
    fi
}

# pwd
root='~/LMM_caption'
cd "$root"

# 确保使用的是 bash shell
conda init bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: the script should be run under Bash." >&2
    exit 1
fi
# 初始化Conda环境
source /home/user/miniconda3/etc/profile.d/conda.sh || {
    echo "Failed to source conda profile script." >&2
    exit 1 
}
# conda
conda deactivate
conda activate modelscope
echo "Current conda env: $CONDA_DEFAULT_ENV"

# definition
work_dir='~/LMM_caption/split_file'
out_dir='~/LMM_caption/json_dir'
shard_num=10
scene_name='train_clip_pic_16'
work_dir=$work_dir/$scene_name
out_dir=$out_dir/$scene_name
doMakeDir $work_dir
doMakeDir $out_dir

# start
echo ["Do processing!"]
for ((i=0; i<$shard_num; i++))
    do
        # define name/path
        name=$(printf "slice%04d" $i) 
        input_path="$work_dir/$name"
        # output_path="$out_dir/$name"

        # gpu
        gpu=$(($i % $shard_num))
        echo $"Using GPU: $gpu"

        # log
        out_log=$out_dir$"/logs/"
        doMakeDir $out_log

        # command
        ### 根据任务切换--prompt1/--prompt2(后期可改造成串联关系) ###
        # CUDA_VISIBLE_DEVICES=$gpu unbuffer -p python qwen_vl_chat.py \

        CUDA_VISIBLE_DEVICES=$gpu python qwen_vl_chat.py \
            --input_txt $input_path \
            --prompt2 \
            > $out_log$name.log 2>&1 & 
            sleep 5

done
