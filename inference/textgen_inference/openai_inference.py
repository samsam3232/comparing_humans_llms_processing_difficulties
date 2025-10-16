from time import sleep
import random
from collections import defaultdict
import numpy as np
from openai import OpenAI
import os
from typing import List, Dict

client = OpenAI(
  organization=os.environ['OPENAI_ORG'],
  api_key=os.environ['OPENAI_API_KEY'],
)

COMPLETION_ARGS = {"max_tokens": 7, "top_p": 1}


def construct_openai_prompt(model_name: str, prompt: Dict) -> List:
    if "o1" in model_name:
        messages = [
            {"role": "user", "content": [{'type': 'text', 'text': prompt['system']}]},
            {"role": "user", "content": [{'type': 'text', 'text': prompt['question']}]}
        ]
    else:
        messages = [
            {"role": "system", "content": [{'type': 'text', 'text': prompt['system']}]},
            {"role": "user", "content": [{'type': 'text', 'text': prompt['question']}]}
        ]
    return messages


def construct_openai_args(prompt: Dict, generation_args: Dict, is_cot: bool, is_thinking: bool) -> Dict:

    messages = construct_openai_prompt(generation_args['model_name'], prompt)
    curr_completion_args = {**COMPLETION_ARGS, **generation_args, 'messages': messages}
    curr_completion_args["max_tokens"] = 400 if (is_cot or is_thinking) else 7
    curr_completion_args['model'] = curr_completion_args.pop('model_name')
    gen_args = curr_completion_args.pop('generation_args', {})
    curr_completion_args.update(gen_args)

    # Add logprobs parameter for the new API
    curr_completion_args['logprobs'] = True
    curr_completion_args['top_logprobs'] = 5
    if 'o3' in curr_completion_args['model'] or 'o4' in curr_completion_args['model'] or 'o1' in curr_completion_args['model'] or 'gpt-5' in curr_completion_args['model']:
        curr_completion_args.pop('max_tokens', None)
        curr_completion_args.pop('n', None)
        curr_completion_args.pop('logprobs', None)
        curr_completion_args.pop('top_logprobs', None)

    return curr_completion_args


def parse_prediction(answer: str, pot_answers: List):
    words = answer.lower().split(" ")
    for i, word in enumerate(words):
        if word in pot_answers:
            return word
    return None


def return_probs(results: Dict, options: List) -> Dict:
    correct = results[options[0].lower()]
    incorrect = results[options[1].lower()]
    total = max(1, correct + incorrect)
    return {'probs': {'correct': (float(correct) / total),
                      'incorrect': (float(incorrect) / total)},
            'unnormalized_probs': {'correct': correct / 5., 'incorrect': incorrect / 5.}}


def extract_logprobs(response_choice, options, is_cot) -> Dict:

    # Extract logprobs for the generated tokens
    logprobs = {options[0].split(' ')[-1]: 0.0, options[1].split(' ')[-1]: 0.0}
    logprob = None
    opposite_logprob = None

    correct_ans, incorrect_ans = options[0].split(' ')[-1], options[1].split(' ')[-1]

    find_my_answer = is_cot
    answer_found = False

    answer_tokens = ["my", "answer", "is"]
    previous = False
    ans_string = 0
    if hasattr(response_choice, "logprobs") and response_choice.logprobs is not None:
        tokens = response_choice.logprobs.content
        for token_info in tokens:
            token = token_info.token
            if token.lower().replace(" ", "") in answer_tokens:
                ans_string += 1
                if ans_string == 3:
                    answer_found = True
                if not previous:
                    previous = True
            elif previous and token.lower().replace(" ", "") not in answer_tokens:
                previous = False
                ans_string = 0

            if correct_ans.lower() in token.lower() or incorrect_ans.lower() in token.lower() and (answer_found or not find_my_answer):
                logprob = np.exp(token_info.logprob)
                found = correct_ans if correct_ans.lower() in token.lower() else incorrect_ans
                search_for = correct_ans if correct_ans.lower() not in token.lower() else incorrect_ans
                for top_logprob in token_info.top_logprobs:
                    if search_for.lower() in top_logprob.token.lower():
                        opposite_logprob = np.exp(top_logprob.logprob)
                        break
                if opposite_logprob is None:
                    opposite_logprob = 0.0
                if token and logprob is not None:
                    logprobs[found] = logprob
                    logprobs[search_for] = opposite_logprob
                break

    return logprobs


def find_explanation(text: str) -> str:
    """
    Extracts the explanation from the text if it exists.
    """
    if 'explanation' in text.lower():
        txt = text.split('\nMy answer is')
        return txt
    return text


def get_model_response(completion_args: Dict) -> str:

    try:
        response = client.chat.completions.create(**completion_args)
    except Exception as e:
        sleep(random.randint(2000, 3000) / 1000.)
        try:
            response = client.chat.completions.create(**completion_args)
        except Exception:
            sleep(random.randint(2000, 3000) / 1000.)
            response = client.chat.completions.create(**completion_args)
    
    return response


def retrieve_reasoning_predictions(completion_args: Dict, options: List = ['Yes', 'No'], is_cot: bool = False) -> Dict:

    options_lower = [i.lower().replace('the ', '') for i in options]

    results = defaultdict(lambda : 0)
    for i in range(5):
        response = get_model_response(completion_args)

        for i in range(len(response.choices)):
            curr_response = response.choices[i].message.content

            answer = parse_prediction(curr_response, options_lower)
            if answer is not None:
                results[answer] += 1
    
    results = return_probs(results, options_lower)

    return results, ""


def retrieve_non_thinking_predictions(completion_args: Dict, options: List = ['Yes', 'No'], is_cot: bool = False) -> Dict:

    response = get_model_response(completion_args)

    logprobs = extract_logprobs(response.choices[0], options, is_cot)

    returned_logprobs = {'probs': dict(), 'unnormalized_probs': dict()}

    returned_logprobs['unnormalized_probs']['correct'] = logprobs[options[0].split(' ')[-1]]
    returned_logprobs['unnormalized_probs']['incorrect'] = logprobs[options[1].split(' ')[-1]]

    if returned_logprobs['unnormalized_probs']['correct'] + returned_logprobs['unnormalized_probs']['incorrect'] == 0:
        returned_logprobs['probs']['correct'] = 0.0
        returned_logprobs['probs']['incorrect'] = 0.0
    else:
        returned_logprobs['probs']['correct'] = returned_logprobs['unnormalized_probs']['correct'] / (returned_logprobs['unnormalized_probs']['correct'] + returned_logprobs['unnormalized_probs']['incorrect'])
        returned_logprobs['probs']['incorrect'] = returned_logprobs['unnormalized_probs']['incorrect'] / (returned_logprobs['unnormalized_probs']['correct'] + returned_logprobs['unnormalized_probs']['incorrect'])

    txt = ""
    if is_cot:
        txt = find_explanation(response.choices[0].message.content)
    return returned_logprobs, txt


def retrieve_model_predictions(completion_args: Dict, options: List = ['Yes', 'No'], is_cot: bool = False, is_thinking: bool = False) -> Dict:

    if is_thinking:
        returned_logprobs, txt = retrieve_reasoning_predictions(completion_args, options, is_cot)
    else:
        returned_logprobs, txt = retrieve_non_thinking_predictions(completion_args, options, is_cot)
    return returned_logprobs, txt


def run_openai_preds(model_args: Dict, prompt: Dict, options: List, is_cot: bool = False, is_thinking: bool = False) -> Dict:
    completion_args = construct_openai_args(prompt, model_args, is_cot, is_thinking)
    completion_args.pop('open_source', None)
    completion_args.pop('compute_type', None)

    model_name = model_args['model_name']
    thinking = False
    if 'o3' in model_name or 'o4' in model_name or "o1" in model_name or "gpt-5" in model_name:
        thinking = True
    curr_results = retrieve_model_predictions(completion_args, options, is_cot, thinking)
    return curr_results[0], curr_results[1]