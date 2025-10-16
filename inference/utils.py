from typing import DefaultDict, Dict, List
from collections import defaultdict
import os
import pandas as pd
from copy import deepcopy
from global_utils import read_as_dict, read_file
from constants import *


def check_config_correctness(config: DefaultDict) -> None:

    """
    Checks if the config has the right parameters
    """

    assert "data_path" in config, "You need a path to the data you want to test"
    assert "model_args" in config, "You need some models you want to check"
    assert '.csv' in config['data_path'], "Your data needs to be in a csv format"
    assert "results_path" in config, "You need a path to the results"


def get_results(results_path: str) -> DefaultDict:

    """
    Given a path to the results_path, loads the already existing results if there are some
    """

    results = defaultdict(lambda: list())
    if os.path.exists(results_path):
        saved_res = pd.read_csv(results_path)
        for col in saved_res.columns:
            if "named" in col:
                continue
            results[col] += saved_res[col].tolist()

    return results


def get_prefix(config: DefaultDict) -> List:

    """
    Given a config returns the base prefix for the runs we are about to perform.
    """

    if "prompt_path" in config:
        base_prefix = read_as_dict(config["prompt_path"])
    else:
        quest = DEFAULT_QUESTION
        quest_cot = DEFAULT_QUESTION_COT
        base_prefix = [{'system': DEFAULT_SYSTEM, 'question': quest}, 
                       {'system': DEFAULT_SYSTEM_2, 'question': quest}]
        base_prefix_cot = [{'system': DEFAULT_SYSTEM, 'question': quest_cot},
                           {'system': DEFAULT_SYSTEM_2, 'question': quest_cot}]

        quest_rev = DEFAULT_QUESTION_REV
        quest_cot_rev = DEFAULT_QUESTION_COT_REV
        base_prefix_rev = [{'system': DEFAULT_SYSTEM, 'question': quest_rev}, 
                       {'system': DEFAULT_SYSTEM_2, 'question': quest_rev}]
        base_prefix_cot_rev = [{'system': DEFAULT_SYSTEM, 'question': quest_cot_rev},
                           {'system': DEFAULT_SYSTEM_2, 'question': quest_cot_rev}]


    if "examples_path" in config:
        examples = read_file(config["examples_path"])
    else:
        questiontype = config.get('question_type', 'yes_no')
        num_examples = config.get('num_examples', 4)
        examples = get_examples(questiontype, num_examples)
        examples_cot = get_examples(questiontype, num_examples, 8, 'cot')
        examples_rev = get_examples(questiontype, num_examples, 8, 'rev')
        examples_cot_rev = get_examples(questiontype, num_examples, 8, 'cot_rev')

    if config.get("question_type", 'yes_no') == "question":
        suffix = "My answer is: The "
    else:
        suffix = "My answer is: "

    prompts, prompts_cot, prompts_rev, prompts_cot_rev = list(), list(), list(), list()

    for j in range(8):
        new_prompt = deepcopy(base_prefix_cot[j%2])
        new_prompt['suffix'] = suffix
        curr_ex = examples_cot[j]
        for key in new_prompt:
            new_prompt[key] = new_prompt[key].replace('EXAMPLES', curr_ex)
        prompts_cot.append(new_prompt)
    
    for j in range(8):
        new_prompt = deepcopy(base_prefix[j%2])
        new_prompt['suffix'] = suffix
        curr_ex = examples[j]
        for key in new_prompt:
            new_prompt[key] = new_prompt[key].replace('EXAMPLES', curr_ex)
        prompts.append(new_prompt)


    for j in range(8):
        new_prompt = deepcopy(base_prefix_cot_rev[j%2])
        new_prompt['suffix'] = suffix
        curr_ex = examples_cot_rev[j]
        for key in new_prompt:
            new_prompt[key] = new_prompt[key].replace('EXAMPLES', curr_ex)
        prompts_cot_rev.append(new_prompt)
    
    for j in range(8):
        new_prompt = deepcopy(base_prefix_rev[j%2])
        new_prompt['suffix'] = suffix
        curr_ex = examples_rev[j]
        for key in new_prompt:
            new_prompt[key] = new_prompt[key].replace('EXAMPLES', curr_ex)
        prompts_rev.append(new_prompt)

    return prompts, prompts_cot, prompts_rev, prompts_cot_rev


def split_sample(line: pd.Series) -> List:

    """
    Given a line in a data frame returns the relevant elements
    """

    curr_sent = line['sentence']
    curr_quest = line['question']
    curr_options = [line['correct_answer'], line['incorrect_answer']]
    curr_ans = line['correct_answer'].lower()
    return [curr_sent, curr_quest, curr_options, curr_ans]


def construct_prompt(prefix: Dict, question: str, sent: str) -> Dict:

    """
    Given a sentence and a question, constructs the prompts
    """

    prompt = deepcopy(prefix)
    for key in prompt:
        prompt[key] = prompt[key].replace("QUESTION", question).replace("SENTENCE", sent)

    return prompt
