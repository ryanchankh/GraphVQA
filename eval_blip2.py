import argparse
import os
import time
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image


def main(args):

    ## CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(2)
    
    ## DATASET
    df_image_question_answer = pd.read_csv('./meta_info/val_image_question_answer.csv', index_col=0)
    
    ## BLIP-2 model
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipImageProcessor
#     image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#     model = Blip2ForConditionalGeneration.from_pretrained(
#         "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
#     )
    processor = Blip2Processor.from_pretrained(
        "/shared_data/p_vidalr/ryanckh/huggingface/models--Salesforce--blip2-flan-t5-xl/snapshots/cc2bb7bce2f7d4d1c37753c7e9c05a443a226614/"
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        "/shared_data/p_vidalr/ryanckh/huggingface/models--Salesforce--blip2-flan-t5-xl/snapshots/cc2bb7bce2f7d4d1c37753c7e9c05a443a226614/", torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    print('Model checkpoint loaded!')
    
    ## Gerate Attribute
    vqa_answer = {'image_id': [], 'answer_vqa': []}
    n_images = len(df_image_question_answer)
    for image_id, question, answer_short, answer_long in  tqdm(df_image_question_answer.iloc, total=n_images):
        image = Image.open(f'./images/{image_id}.jpg')
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                
                inputs_dict = processor(images=image, text=question, return_tensors="pt").to(device)
                
                # feedforward BLIP2
                outputs = model.generate(**inputs_dict)
                processor.decode(outputs[0], skip_special_tokens=True)
                vqa_answer['image_id'].append(image_id)
                vqa_answer['answer_vqa'].append(processor.decode(outputs[0], skip_special_tokens=True))
    pd.DataFrame(vqa_answer).to_csv('./meta_info/val_image_question_answer_blip2.csv')
        
def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='imagenet')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parseargs()
    main(args)

