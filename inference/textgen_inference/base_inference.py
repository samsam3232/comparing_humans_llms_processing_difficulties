import argparse
from global_utils import read_as_defaultdict
from inference.utils import *
import pandas as pd
from itertools import product
import json
from collections import defaultdict
import os
from tqdm import tqdm
import torch
from inference.textgen_inference.fastchat_inference import run_fastchat_preds
from inference.textgen_inference.openai_inference import run_openai_preds
from typing import List
from fastchat.model import load_model, get_conversation_template
from transformers import MllamaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM


def run_prediction(model, tokenizer, prompt: Dict, options: List, model_args: Dict, is_cot: bool = False, is_thinking: bool = False):

    if model_args.get('open_source', True):
        results, txt = run_fastchat_preds(model, tokenizer, model_args, prompt, options, is_cot, is_thinking)
    else:
        results, txt = run_openai_preds(model_args, prompt, options, is_cot, is_thinking)
    return results, txt


def find_is_done(results, sentence, model, question, compute_type):

    for i in range(len(results['model'])):
        if results['model'][i] == model and results['sentence'][i] == sentence and results['question'][i] == question and results['compute_type'][i] == compute_type:
            return True
    return False


def count_per_model(results, model):

    count = 0
    for mod in results['model']:
        if mod == model:
            count += 1
    return count


def load_models(model_args, rev):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if "llama-3.2" in model_args['model_name'].lower() and 'vision' in model_args['model_name'].lower():
        if 'num_gpus' in model_args.get('creation_args', {}):
            ngpus = model_args['creation_args'].pop('num_gpus')
            model_args['creation_args']['device_map'] = 'sequential'
            max_gpu_memory = {i: "60GiB" for i in range(ngpus)}
        else:
            max_gpu_memory = None
        model = MllamaForConditionalGeneration.from_pretrained(model_args['model_name'], device_map="balanced", max_memory=max_gpu_memory, torch_dtype = torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
    elif "gemma-2" in  model_args['model_name'].lower():
        rev = "main"
        tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
        model = AutoModelForCausalLM.from_pretrained(model_args['model_name'], device_map="auto",)
    elif "gemma-3" in model_args['model_name'].lower():
        rev = "main"
        tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
        model = Gemma3ForCausalLM.from_pretrained(model_args['model_name'], device_map="auto",)
    else:
        model, tokenizer = load_model(model_args['model_name'], device, revision=rev,
                                  debug=False, **model_args.get('creation_args', {}))
        if rev != "main":
            model_args['model_name'] = model_args['model_name'] + f"_{rev}"
    
    return model, tokenizer


def load_prefixes(config):

    if os.path.exists(config['prefix_path']):
        with open(config['prefix_path'], 'r') as f:
            prefix = json.load(f)

        with open(config['prefix_path'].replace('.json', '_rev.json'), 'r') as f:
            rev_prefix = json.load(f)
        
        with open(config['prefix_path'].replace('.json', '_cot.json'), 'r') as f:
            cot_prefix = json.load(f)

        with open(config['prefix_path'].replace('.json', '_cot_rev.json'), 'r') as f:
            cot_rev_prefix = json.load(f)
    else:
        prefix, cot_prefix, rev_prefix, cot_rev_prefix = get_prefix(config)
        with open(config['prefix_path'], 'w') as f:
            json.dump(prefix, f)
        with open(config['prefix_path'].replace('.json', '_rev.json'), 'w') as f:
            json.dump(rev_prefix, f)
        with open(config['prefix_path'].replace('.json', '_cot.json'), 'w') as f:
            json.dump(cot_prefix, f)
        with open(config['prefix_path'].replace('.json', '_cot_rev.json'), 'w') as f:
            json.dump(cot_rev_prefix, f)
    
    return prefix, cot_prefix, rev_prefix, cot_rev_prefix


def main(config_path):

    config = read_as_defaultdict(config_path)
    check_config_correctness(config)

    df = pd.read_csv(config['data_path'])
    results = get_results(config['results_path'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prefix, cot_prefix, rev_prefix, cot_rev_prefix = load_prefixes(config)

    keys2add = config.get('keys_to_add', ['sentence_type'])

    model, tokenizer = 1, 1
    previous_model_name = None
    for model_args in config['model_args']:
        curr_model_name = model_args['model_name']
        rev = "main" if 'revision' not in model_args.get('creation_args', {}) else model_args['creation_args'].pop('revision')
        if rev != "main":
            curr_model_name = model_args['model_name'] + f"_{rev}"

        if model_args.get('open_source', False):
            
            if curr_model_name != previous_model_name:
                del model
                model, tokenizer = load_models(model_args, rev)
                previous_model_name = curr_model_name

        
        print(model_args['model_name'])

        compute_type = model_args.get('compute_type', 'regular')
        is_cot = compute_type == 'cot'
        is_thinking = compute_type == 'thinking'

        for i in tqdm(range(df.shape[0])):

            curr_sent, curr_quest, curr_options, curr_ans = split_sample(df.iloc[i])

            if find_is_done(results, curr_sent, curr_model_name, curr_quest, compute_type):
                continue

            prefixes = [prefix, rev_prefix] if not is_cot else [cot_prefix, cot_rev_prefix]
            order = ['reg', 'rev']
            for k, pref in enumerate(prefixes):

                pref = pref if ("o4" not in curr_model_name and "o3" not in curr_model_name and "o1" not in curr_model_name and "gpt-5" not in curr_model_name) else [pref[0]]  # O4 models only use the first prefix
                for j, curr_prefix in enumerate(pref):
                    prompt = construct_prompt(curr_prefix, curr_quest, curr_sent)

                    try:
                        curr_results, special_txt = run_prediction(model, tokenizer, prompt, curr_options, model_args, is_cot, is_thinking)
                    except Exception as e:
                        print(f"Error for {model_args['model_name']} on {curr_sent} -- {str(e)}")
                        continue

                    results['model'].append(model_args['model_name'])
                    results['sentence'].append(curr_sent)
                    results['question'].append(curr_quest)
                    results['correct'].append(curr_results['probs']['correct'])
                    results['incorrect'].append(curr_results['probs']['incorrect'])
                    results['prompt_index'].append(j)
                    results['base_correct'].append(curr_results['unnormalized_probs']['correct'])
                    results['base_incorrect'].append(curr_results['unnormalized_probs']['incorrect'])     
                    results['compute_type'].append(compute_type)  
                    results['special_txt'].append(special_txt)
                    results['order'].append(order[k])

                    for key in keys2add:
                        results[key].append(df.iloc[i][key])
                    
                    if compute_type != "regular":
                        break
            
            if i % 5 == 0 and i > 0:
                df_res = pd.DataFrame.from_dict(results)
                df_res.to_csv(config['results_path'])

        df_res = pd.DataFrame.from_dict(results)
        df_res.to_csv(config['results_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Instruction style')
    parser.add_argument('-c', '--config_path', type=str, help="Path to where the configuration is kept")
    args = parser.parse_args()
    main(**vars(args))
