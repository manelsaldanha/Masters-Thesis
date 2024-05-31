from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)
import json
import os

def write_to_json(filename,data):
    try:
        with open(filename, 'r') as json_file:
            existing_data = json.load(json_file)
            
            with open(filename, 'w') as json_file:
                if existing_data:
                    existing_data.append(data)
                    json.dump(existing_data, json_file)
                else:
                    json.dump([data], json_file)
    except:
        with open(filename, 'w') as json_file:
            json.dump([data], json_file)


start_idx=10659
questions=[]
json_file_path=r'train_Toloka.json'
# Open the file and load the JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

for d in data:
    questions.append(d['refer'])

questions=questions[start_idx:]
device = "cuda" # the device to load the model onto
file_path = "results/Toloka/toloka_paraphrase2.json"


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",padding_side='left')
model.to(device)

for i, q in tqdm(enumerate(questions), total=len(questions)):
#for i,q in enumerate(questions):
    count_it=0
    while 1:
        not_english_char=0
        buffer=f'Rewrite a question that I will give you into a single and short statement inside "", without losing the meaning/context of the question. It is mandantory that the statement can not be another question.'

        if '?' not in q:
            q=q+'?'

        message=[{"role": "user", "content": buffer},
        {"role": "assistant", "content": "What is the question to re-write?"},
        {"role": "user", "content": q},
        {"role": "assistant", "content": 'The paraphrase sentence is "'}]

        encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")

        model_inputs = encodeds.to(device)
    

        generated_ids = model.generate(model_inputs, max_new_tokens=40, do_sample=True,pad_token_id=tokenizer.eos_token_id)
        decoded = tokenizer.batch_decode(generated_ids)

        answer=decoded[0].split('[/INST]')[-1]
        sentence=answer.split('"')[1]
        n_line_count = sentence.count('\n')
        n_tab_count = sentence.count('\t')
        
        print(sentence)
        if ('?' not in sentence) and (len(sentence)>15) and (n_line_count<=1) and (n_tab_count<=2) and ('_' not in sentence) and ('~' not in sentence):
            for char in sentence:
                if ord(char) > 127:
                    not_english_char=1
                    break
            if not_english_char==0:
                break
        
        if count_it==15:
            sentence=sentence.lower()
            sentence=sentence.replace('?', '')
            sentence=sentence.replace('where', '')
            sentence=sentence.replace('when', '')
            sentence=sentence.replace('what', '')
            sentence=sentence.replace('how', '')
            sentence=sentence.replace('who', '')
            break
        
        count_it+=1

    sentence=sentence.replace('_', '')
    sentence=sentence.replace('~', '')
    print('SENTENCE: ',sentence)
    write_to_json(file_path,{'idx':start_idx+i,'original':q,'paraphrase':sentence})
