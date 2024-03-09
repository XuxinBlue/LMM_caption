
'''
Qwen-vl-chat
'''
# Pytorch
import torch
torch.manual_seed(1234)

# Model
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

# Others
import os
from tqdm import tqdm
import json
from PIL import Image
import re
import argparse

parser = argparse.ArgumentParser(description='Qwen-vl-chat')
parser.add_argument('--input_txt', type=str, help='input txt')
parser.add_argument('--prompt1', action='store_true', help='whether to process statge1')
parser.add_argument('--prompt2', action='store_true', help='whether to process statge2')
args = parser.parse_args()

# funcs
def get_img_reso(img_path):
    with Image.open(img_path) as img:
        return img.size
    
def traversal_infer(infer_objects, infer_objects_name, model, tokenizer, history, output_path, width, height):
    # infer_objects是个多元素的dict
    for infer_object in infer_objects:
        if infer_object['value'] != '*':
            # get prompt2
            prompt = f"对{infer_object['value']}进行bbox标注."
            # infer_one
            response, history_out = model.chat(tokenizer, prompt, history)
            print(f'response: {response}')
            # response = "<ref> man</ref><box>(241,354),(707,999)</box>"

            # 使用正则表达式匹配坐标值
            matches = re.findall(r'\((\d+),(\d+)\)', response)

            # matches 是一个包含所有匹配的列表，每个匹配都是一个元组，元组的两个元素分别是 x 和 y 坐标
            # 例如，对于这个例子，matches 将会是：[('241', '354'), ('707', '999')]

            # 如果你想要将这些坐标转换为整数，你可以这样做：
            coordinates = [(float(x)/10, float(y)/10) for x, y in matches]
            # 现在，coordinates 是一个包含所有坐标的列表，每个坐标都是一个元组，元组的两个元素分别是 x 和 y 坐标
            # 例如，对于这个例子，coordinates 将会是：[(24, 35), (70, 99)]

            # 提取xy坐标值为array
            x_values, y_values = [], []
            for x, y in coordinates:
                x_values.append(x)
                y_values.append(y)
            # x_values是[24, 70], y_values是[35, 99]

            # save to json
            with open(output_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                # 同一属性有多个的情况
                for i in range(0, len(x_values), 2):
                    elem_dict = {
                        "x": x_values[i],
                        "y": y_values[i],
                        "width": x_values[i+1] - x_values[i],
                        "height": y_values[i+1] - y_values[i],
                        "rotation": 0,
                        "rectanglelabels": [
                            infer_object['value']
                        ],
                        "original_width": width,
                        "original_height": height
                    }
                    # 为nouns/verbs赋空值
                    if infer_objects_name not in data:
                        data[infer_objects_name] = []
                    # 得到新的data
                    data[infer_objects_name].append(elem_dict)
                # 将文件指针移动到文件的开始
                f.seek(0)
                # 将修改后的 data 写回到文件中
                json.dump(data, f, indent=4, ensure_ascii=False)
                # 清除文件的剩余部分（如果有的话）
                f.truncate()

if __name__ == '__main__':
    ### 1. 模型定义
    model_id = 'qwen/Qwen-VL-Chat'
    # model_id = 'qwen/Qwen-VL'
    revision = 'v1.1.0'
    # revision = 'v1.0.3'

    model_dir = snapshot_download(model_id, revision=revision)
    # 请注意：分词器默认行为已更改为默认关闭特殊token攻击防护。
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, bf16=True).eval()
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, fp16=True).eval()
    # 使用CPU进行推理，需要约32GB内存
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu", trust_remote_code=True).eval()
    # 默认使用自动模式，根据设备自动选择精度
    # model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()

    # 可指定不同的生成长度、top_p等相关超参
    model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)

    ### 2. 变量定义prompt/input
    # 2.1 得到input_txt及input_json
    input_jpg_paths, input_json_paths, output_json_paths = [], [], []
    with open(args.input_txt, 'r') as f:
        for line in f:
            input_jpg_paths.append(line.strip())
            input_json_paths.append(line.strip().replace('.jpg', '.json'))
            output_json_paths.append(line.strip().replace('.jpg', '_output.json'))

    # 2.2 得到prompt
    # 分阶段提问, 第1步提问时间,第2步提问目标检测结果
    prompt1 = "回答上面图片的时间,从这3个选项中选一个: Daytime, Evening, Night, 如不能确定则随机选择1个,记住,你的回答只能有1个单词"


    ### 3. 模型infer
    for path, input_json_path, output_json_path in tqdm(zip(input_jpg_paths, input_json_paths, output_json_paths)):
        try:
            # stage1
            if args.prompt1:
                # get history
                history1 = []
                info1 = (f'Picture 0:<img>{path}</img>\n', 'Got it.')
                history1.append(info1)
                # 输出
                response1, history1_out = model.chat(tokenizer, prompt1, history=history1)
                print(f'response1: {response1}')

                # 保存
                output_json = path.replace('.jpg', '_output.json')
                # 先读取
                with open(output_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 再修改
                for label in data.get('Label', []):
                    if 'rectanglelabels' in label:
                        label['rectanglelabels'] = [response1]
                # 最后写入
                with open(output_json, 'w') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

            # stage2
            if args.prompt2:
                # check flag
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'flag' in data and data['flag'] == 1:
                        continue
                # flag为false就删掉重标(important)
                with open(output_json_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'nouns' in data:
                        del data['nouns']
                    if 'verbs' in data:
                        del data['verbs']
                    f.seek(0)
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.truncate()

                # get img_reso
                img_width, img_height = get_img_reso(path)

                # get json's nouns/verbs
                with open(input_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    nouns = data['nouns']
                    verbs = data['verbs']

                # get history
                history2 = []
                info1 = (f'Picture 0:<img>{path}</img>\n', 'Got it.')
                history2.append(info1)

                # traversal infer            
                traversal_infer(nouns, 'nouns', model, tokenizer, history2, output_json_path, img_width, img_height)
                traversal_infer(verbs, 'verbs', model, tokenizer, history2, output_json_path, img_width, img_height)

                # flag
                with open(output_json_path, 'r+', encoding='utf-8') as f:
                    data = json.load(f)
                    data['flag'] = 1
                    f.seek(0)
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.truncate()

        except Exception as e:
            print(f'{e}')
            with open('./error.txt', 'a') as f:
                f.write(f'{path}\n')

# CUDA_VISIBLE_DEVICES=0 python qwen_vl_chat.py --input_txt '~/LMM_caption/split_file/train_clip_pic_16/slice0000' --prompt1
# CUDA_VISIBLE_DEVICES=0 python qwen_vl_chat.py --input_txt '~/LMM_caption/split_file/train_clip_pic_16/slice0000' --prompt2
# CUDA_VISIBLE_DEVICES=0 python qwen_vl_chat.py --input_txt '~/LMM_caption/caption_run/error.txt' --prompt2
        
        
