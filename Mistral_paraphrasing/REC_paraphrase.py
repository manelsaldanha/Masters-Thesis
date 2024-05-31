from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from tqdm import tqdm
from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)
import json

questions=[]
start_idx=201707
json_file_path=r'/cfs/home/u021542/Mistral/train_refcoco_mix_unc.json'
# Open the file and load the JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

for d in data:
    questions.append(d['refer'])

questions=questions[start_idx:]

device = "cuda" # the device to load the model onto
file_path = "results/unc_RefCOCO/RefCOCO4mix.txt"

with open(file_path, 'w') as file:

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",padding_side='left')
    model.to(device)

    for i, q in tqdm(enumerate(questions), total=len(questions)):
    #for i,q in enumerate(questions):
        count_it=0
        while 1:
            not_english_char=0
            buffer=f'Rewrite a descriptive statement that I will give you into a single and short question inside "", without losing the meaning/context of the original desciptive statement.'
    

            message=[{"role": "user", "content": buffer},
            {"role": "assistant", "content": "What is the descriptive statement to re-write?"},
            {"role": "user", "content": q},
            {"role": "assistant", "content": 'The paraphrase question is "'}]

            encodeds = tokenizer.apply_chat_template(message, return_tensors="pt")

            model_inputs = encodeds.to(device)
        

            generated_ids = model.generate(model_inputs, max_new_tokens=40, do_sample=True,pad_token_id=tokenizer.eos_token_id)
            decoded = tokenizer.batch_decode(generated_ids)
    
            answer=decoded[0].split('[/INST]')[-1]
            sentence=answer.split('"')[1]
            n_line_count = sentence.count('\n')
            n_tab_count = sentence.count('\t')
            
            print(sentence)
            if ('?' in sentence) and (len(sentence)>15) and (n_line_count<=1) and (n_tab_count<=2) and ('_' not in sentence) and ('~' not in sentence):
                for char in sentence:
                    if ord(char) > 127:
                        not_english_char=1
                        break
                if not_english_char==0:
                    break
            
            if count_it==15:
                sentence=sentence.lower()
                sentence=sentence.replace('?', '')
                sentence="{}?".format(sentence)
                
                if ('where' in sentence) or ('when' in sentence) or ('what' in sentence) or ('how' in sentence) or ('who' in sentence) or ('which' in sentence):
                    break

                sentence="Is this {}".format(sentence)

                break
            
            count_it+=1

        sentence=sentence.replace('_', '')
        sentence=sentence.replace('~', '')
        sentence=sentence.split('?')[0]+'?'
        print('QUESTION: ',sentence)
        file.write(f"idx:{start_idx+i}; original: {q}; paraphrase: {sentence}\n")

