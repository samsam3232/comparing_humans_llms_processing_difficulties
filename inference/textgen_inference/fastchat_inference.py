from typing import Dict, List
from fastchat.model import get_conversation_template
from collections import defaultdict
from transformers import PreTrainedTokenizer
import numpy as np
import torch
import string as string_funcs


def clean_word(word):

    word = word.replace(' ', '').lower().strip()
    if len(word) == 0:
        return word
    if not word[0].isascii():
        word = word[1:]
    
    return word


def run_thinking_pred(model, tokenizer, prompt):

    curr_prompt = [{'role': 'user', 'content': prompt['system']}, {'role': 'user', 'content': prompt['question']}]
    text = tokenizer.apply_chat_template(
        curr_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=400
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    outputs = tokenizer.decode(output_ids[:], skip_special_tokens=True).strip("\n")
    
    pred = outputs.split('</think>')[0] + '</think>'
    return pred
    
    
def run_cot_pred(model, tokenizer, prompt, model_args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    curr_prompt = ""
    curr_prompt += prompt['system']
    curr_prompt += f"\n\n{prompt['question']}"
    curr_prompt += f"\n\nExplanation:"

    input_ids = tokenizer(curr_prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in input_ids:
        input_ids.pop("token_type_ids")

    pad_tok = tokenizer.unk_token_id
    
    with torch.no_grad():
        outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=400,
                            pad_token_id=pad_tok, **model_args.get('generation_args', {}))

    if model.config.is_encoder_decoder:
        outputs = outputs.sequences
    else:
        outputs = outputs.sequences[0, input_ids['input_ids'].shape[1]:]
        
    pred = tokenizer.decode(outputs, skip_special_tokens=True, skip_between_special_tokens = False).strip()
    new_pred = '.'.join(pred.lower().split('answer')[0].split('.')[:-1]).capitalize()
    explanation = string_funcs.capwords(new_pred, sep='. ')

    return explanation


def find_opt_tokens(tokenizer, options):

    if hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    else:
        vocab = tokenizer.get_vocab()
    results = defaultdict(lambda: list())
    results['correct'] = [k for i, k in vocab.items() if clean_word(i) == options[0].lower().replace('the ', '')]
    results['incorrect'] = [k for i, k in vocab.items() if clean_word(i) == options[1].lower().replace('the ', '')]
    return results


def parse_mc_generation_results(outputs: Dict, tokenizer: PreTrainedTokenizer, options: List = ['Yes', 'No']):

    """
    Parses the multichoice generation to get: the prediction themselves, and the probabilities of each of the possible
    choices.
    """

    results = dict()
    pred = tokenizer.decode(outputs['sequences'][0][-2:], skip_special_tokens=True).strip()
    
    opt_tokens = find_opt_tokens(tokenizer, options)

    tokens = opt_tokens['correct'] + opt_tokens['incorrect']

    found = False
    for j in range(len(outputs.scores)):
        unnormalized_scor = outputs.scores[j][0].softmax(dim=-1)[tokens]
        unnormalized_probs = defaultdict(lambda: 0)
        for i in range(len(opt_tokens['correct'])):
            if np.isnan(unnormalized_scor[i].item()):
                continue
            unnormalized_probs['correct'] += unnormalized_scor[i].item()
        for i in range(len(opt_tokens['incorrect'])):
            if np.isnan(unnormalized_scor[i].item()):
                continue
            unnormalized_probs['incorrect'] += unnormalized_scor[i + len(opt_tokens['correct'])].item()
        if unnormalized_probs['incorrect'] > 0 or unnormalized_probs['correct'] > 0:
            found = True
            break

    probs = defaultdict(lambda: 0)
    new_found = False
    for j in range(len(outputs.scores)):
        curr_scor = outputs.scores[j][0, tokens].softmax(dim=-1)
        for i in range(len(opt_tokens['correct'])):
            if np.isnan(curr_scor[i].item()):
                continue
            probs['correct'] += curr_scor[i].item()

        for i in range(len(opt_tokens['incorrect'])):
            if np.isnan(curr_scor[i].item()):
                continue
            probs['incorrect'] += curr_scor[i + len(opt_tokens['correct'])].item()

        if probs['correct'] > 0 or probs['incorrect'] > 0:
            new_found = True
            break

    results['probs'] = probs if new_found else {'correct': 0.0, 'incorrect': 0.0}
    results['unnormalized_probs'] = unnormalized_probs if found else {'correct': 0.0, 'incorrect': 0.0}
    results['text'] = pred
    return results


def get_prompt(model_name: str, prompt: Dict) -> str:

    """
    Retrieves the prompts for the answer generation
    """

    curr_prompt = ""
    curr_prompt += prompt['system']
    curr_prompt += f"\n\n{prompt['question']}"
    curr_prompt += f"\n\n{prompt['suffix']}"

    return curr_prompt


def normalize_probs(probs: Dict) -> Dict:

    """
    Given probabilities, normalizes them to sum to 1
    """

    new_probs = dict()
    sum_probs = sum(list(probs.values()))
    for key in probs:
        new_probs[key] = float(probs[key]) / sum_probs

    return new_probs


def run_fastchat_preds(model, tokenizer, model_args: Dict, prompt: str, options: List, is_cot: bool = False, is_thinking: bool = False) -> Dict:

    """
    Gets the probabilities for the correct and incorrect answers for open source models.
    """

    device = 'cuda'

    curr_prompt = get_prompt(model_args['model_name'], prompt)

    special_txt = ""
    if is_cot:
        cot_txt = run_cot_pred(model, tokenizer, prompt, model_args)
        curr_prompt = curr_prompt.replace(prompt['suffix'], '') + cot_txt + f"\n{prompt['suffix']}"
        special_txt = f"COT --- {cot_txt}"
    elif is_thinking:
        thinking_txt = run_thinking_pred(model, tokenizer, prompt)
        curr_prompt = curr_prompt.replace(prompt['suffix'], '') + thinking_txt + f"\n{prompt['suffix']}"
        special_txt = f"THINKING --- {thinking_txt}"

    input_ids = tokenizer(curr_prompt, return_tensors="pt").to(model.device)
    if "token_type_ids" in input_ids:
        input_ids.pop("token_type_ids")

    with torch.no_grad():
        outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=3,
                                pad_token_id=tokenizer.eos_token_id, **model_args.get('generation_args', {}))
    
    pred = tokenizer.decode(outputs['sequences'][0][:], skip_special_tokens=True).strip()

    curr_results = parse_mc_generation_results(outputs, tokenizer, options)

    return curr_results, special_txt
