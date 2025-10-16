import argparse
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import json

DEFAULT_VALUES = ['sentence', 'question', 'set_id', 'sent_type', 'question_type', 'answer']


def find_indices(single_result: str, values_to_extract: list) -> dict:
    """
    Finds the indices of specified values in a single result string.
    
    Args:
        single_result (str): The result string to search.
        values_to_extract (list): List of keys to find in the result.
    
    Returns:
        dict: A dictionary with keys as values to extract and their corresponding indices.
    """
    indices = {}
    lines = single_result.split('\n')

    for line in lines:
        for value in values_to_extract:
            if f" {value}." in line:
                indices[value] = int(line.split('# ')[1].split('.')[0])
                break

    return indices


def single_results_parse(single_result: str, values_to_extract: list) -> dict:
    """
    Parses a single result string and extracts specified values.
    
    Args:
        single_result (str): The result string to parse.
        values_to_extract (list): List of keys to extract from the result.
    
    Returns:
        dict: A dictionary containing the extracted values.
    """

    indices = find_indices(single_result, values_to_extract)
    lines = single_result.split('\n')

    result_dict = {}
    start_time = None
    for line in lines:
        if "was non-random" in line:
            elements = line.split('= ')[1].split(',')[0]
            result_dict['group_id'] = elements
        if "demographics_form,age" in line:
            elements = line.split(',')
            result_dict['id'] = elements[1]
            result_dict['age'] = elements[10]
            continue
        elif "demographics_form,country" in line:
            elements = line.split(',')
            result_dict['country'] = elements[10]
            continue
        elif "selected_answer_practice,Selection" in line:
            elements = line.split(',')
            result_dict[elements[14]] = elements[10] == 'correct'
            continue
        elif "timeout_item,Start,Start" in line:
            start_time = 0
            continue
        elif "timeout_item,End,End" in line:
            if start_time:
                end_time = 100
                result_dict['time_taken'] = end_time - start_time
            else:
                result_dict['time_taken'] = 6000
            continue
        elif "selected_answer_item,Selection" in line:
            elements = line.split(',')
            result_dict['trial_correct'] = elements[10] == 'correct'
            result_dict['set_id'] = elements[indices['set_id'] - 1]
            result_dict['sentence'] = elements[indices['sentence'] - 1]
            result_dict['question'] = elements[indices['question'] - 1]
            result_dict['sent_type'] = elements[indices['sent_type'] - 1]
            result_dict['quest_type'] = elements[indices['question_type'] - 1]
            continue
    
    if 'id' not in result_dict or 'trial_correct' not in result_dict:
        result_dict = {}

    return result_dict


def find_missing_values(results_df: pd.DataFrame) -> dict:

    """
    Filters out experinents where participants did not answer correctly to the practice questions.
    """

    # filters out rows where either practice_1 or practice_2 is false
    filtered_df = results_df[
        (results_df['practice_1'] == True) & (results_df['practice_2'] == True)
    ]

    # filtered_df = filtered_df[filtered_df['order'] == "rev"]

    # for each set_id count the number of occurrences
    set_counts = filtered_df['group_id'].value_counts()
    # find set_ids that have less than 10 occurrences
    missing_sets = set_counts[set_counts < 10].index.tolist()
    missing_numbers = (10 - set_counts[set_counts < 10]).values.tolist()

    missing_dict = {int(i): j for i, j in zip(missing_sets, missing_numbers)}
    return filtered_df, missing_dict



def main(input_file: str, output_file: str, additional_values: str = None):

    values_to_extract = deepcopy(DEFAULT_VALUES)
    if additional_values:
        values_to_extract += additional_values.split(';')
    
    with open(input_file, 'r') as f:
        data = f.read()

    total_results = defaultdict(list)    
    single_results = data.split('# Results on')[1:]

    already_in = list()
    for i, single_result in enumerate(single_results):
        # if 'Tue, 01 Jul' not in single_result and 'Wed, 02 Jul' not in single_result:
        #     continue
        user_results = single_results_parse(single_result, values_to_extract)
        if user_results.get('id') in already_in:
            continue
        already_in.append(user_results.get('id'))
        for key, value in user_results.items():
            total_results[key].append(value)
        if 'practice_2,Dogs live under water and only come up to tan.' in single_result:
            total_results['order'].append('reg')
        elif 'practice_1,Dogs live under water and only come up to tan.' in single_result:
            total_results['order'].append('rev')
    
    df = pd.DataFrame.from_dict(total_results)

    filtered_df, missing_items = find_missing_values(df)
    filtered_df.to_csv(output_file, index=False)

    with open(output_file.replace('.csv', '_missing.json'), 'w') as f:
        json.dump(missing_items, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parse human experiment results")
    parser.add_argument('-i', '--input_file', type=str, help="Path to where the results are kept")
    parser.add_argument('-o', '--output_file', type=str, help="Path where we want to keep the results")
    parser.add_argument('-a', '--additional_values', type=str, default=None, help="Additional elements we want to parse out")
    args = parser.parse_args()
    main(**vars(args))