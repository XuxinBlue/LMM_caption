## Introduction

  这个项目是多模态大模型来标注图片数据集的一次尝试，大模型使用的是来自阿里的Qwen_vl_chat(基于QWen-7B).



  本次大模型标注的思路主要有4部分，首先s0阶段是前置数据处理阶段，包括s0_0： img检索写入txt；s0_1： 大json拆分单个对应json；s0_2：  img分成n份来准备进行多进程并行推理；其次是简单处理s1阶段， 对输入json进行部分信息的简单复制，直接保存不必要更改的信息；然后是推理阶段s2， 多模态大模型标注需变更的信息，又分成2部分，时间信息和物体动作信息， 依据prompt1/2来区分；最后是数据后处理阶段，s3： 合并所有所需json.

## Structure



```
|-- caption_run
|   |-- error.txt
|   |-- qwen_vl_chat.py
|   |-- s0-s1-s3.ipynb
|   |-- s2_parallel_run.sh 
|-- json_dir 
|   |-- dataset's logs
|   |-- dataset's final json
|-- origin_input
|   |-- dataset.json
|   |-- dataset
|   |-- dataset.txt 
|-- split_file
|   |-- dataset's split
|-- others 
|   |-- check.ipynb
```

其中， origin_input： 数据集， 数据集的总json， 数据集的路径txt；split_file： 数据集分成的n份；json_dir： 运行log， 输出的json；caption_run： 运行文件；others：其他文件.

## Installation

- Python Env

```
conda create -n modelscope python=3.8
conda activate modelscope
```

- Pytorch

```
pip3 install torch torchvision torchaudio
```

- Pip

```
pip install modelscope
```

## Inference

项目整体运行步骤： 

第1步在jupyter中按顺序运行s0， s1文件，分别是：s0_0_img_retrival， s0_1_split_json， s0_2_split_img， s1_info_copy； 

第2步运行s2_parallel_run.sh脚本，多进程使用gpu进行模型infer，最终执行文件是qwen_vl_chat.py;

```
bash s2_parallel_run.sh
```

第3步在jupyter中运行s3_merge_json， 合并json.

## TODO

+ [ ] 更换更具推理性价比的多模态大模型
