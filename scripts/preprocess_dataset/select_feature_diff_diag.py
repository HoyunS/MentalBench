# ----------------------------------
# Select features for different diagnosis
# ----------------------------------
import os, json
import copy
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from knowledge_graph import KnowledgeGraph
from question_setter import QuestionSetter


# call final features
def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: dict, file_path: str):
    if os.path.exists(file_path):
        print(f"Selected features for differential diagnosis already exists. Skipping.")
        return
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Selected features for differential diagnosis saved to {file_path}")
    return



def select_features(kg_obj: KnowledgeGraph, qs_obj: QuestionSetter, diff_diag_code, difficulty: str='high'):
    dir_path = './../resources/features/seed/final/{}/json/'
    
    dir_read = dir_path.format('medium')
    dir_save = dir_path.format('high')

    diff_diag_info = kg_obj.get_differential_diagnosis_info(diff_diag_code)
    
    codes = kg_obj.extract_disease_from_differential_diagnosis(diff_diag_code)

    main_disease_code = codes[0]
    sub_disease_code = codes[1]
    
    # call main disease final features (medium difficulty)
    main_disease_features = read_json(os.path.join(dir_read, f"{main_disease_code}.json"))

    diff_diag_features = dict()
    for i, (idx, item) in enumerate(main_disease_features.items()):
        if i >= 5:
            break    # only select first 5 features for diff diag
        # remove diagnostic criteria from main disease features
        diagnostic_criteria = item['question']['sampled_features']['sampled_diagnostic_criteria']
        # additional_requirements = diagnostic_criteria['additional_requirements'] if 'additional_requirements' in diagnostic_criteria else None
        item['question']['sampled_features'].pop('sampled_diagnostic_criteria', None)

        # call additional_requirements for main and sub diseases
        additional_requirements_main = kg_obj.get_disease_features_lookup()[main_disease_code]['diagnostic_criteria'].get('additional_requirements', None)
        additional_requirements_sub = kg_obj.get_disease_features_lookup()[sub_disease_code]['diagnostic_criteria'].get('additional_requirements', None)

        # prepare options for diff diag question
        option_candidates = qs_obj.list_options_for_question(item['question']['sampled_features'])
        must_contain_options = [
            {"disease_code": sub_disease_code,
             "disease_name": kg_obj.convert_code_to_name(sub_disease_code)}
             ]
        mcq_option_dict = qs_obj.sample_options_for_mcq(option_candidates,
                                                      must_contain_option_list=must_contain_options,
                                                      must_remove_option_list=[],
                                                      n_options=4,
                                                      n_highly_likely = 0, n_moderately_likely=1, n_less_likely=1)
        
        index_types_both = ['answer' if t == 'must_contain_option' else t for t in mcq_option_dict['index_types']]  
        index_types_option_b = ['must_contain_option' if t == 'answer' 
                                else 'answer' if t == 'must_contain_option' 
                                else t for t in mcq_option_dict['index_types']]

        # add diff diag
        questions_for_diff_diag = {
            "question_both": {
                'main_disease': main_disease_code,
                'sub_disease': sub_disease_code,
                'sampled_features': item['question']['sampled_features'],
                'additional_condition': diff_diag_info['additional_condition'] if diff_diag_info['additional_condition'] is not None and diff_diag_info['additional_condition'] != "None" else None,
                'key_difference': diff_diag_info['key_difference']
            },
            "question_a": {
                "rule": diff_diag_info['rules'][main_disease_code],
                "additional_requirements": additional_requirements_main
            },
            "question_b": {
                "rule": diff_diag_info['rules'][sub_disease_code],
                "additional_requirements": additional_requirements_sub
            },
            "option_both": {
                'options': mcq_option_dict['options'],
                'answer_index': mcq_option_dict['answer_index'] + mcq_option_dict['must_contain_option_index'],
                'index_types': index_types_both,
                'options_info': mcq_option_dict['options_info']
            },
            "option_a": {
                'options': mcq_option_dict['options'],
                'answer_index': mcq_option_dict['answer_index'],
                'index_types': mcq_option_dict['index_types'],
                'options_info': mcq_option_dict['options_info']
            },
            "option_b": {
                'options': mcq_option_dict['options'],
                'answer_index': mcq_option_dict['must_contain_option_index'],
                'index_types': index_types_option_b,
                'options_info': mcq_option_dict['options_info']
            },
            # save diagnostic criteria separately
            "_removed_diagnostic_criteria": diagnostic_criteria
        }

        diff_diag_features[idx] = questions_for_diff_diag

    # save selected features for diff diag
    save_path = os.path.join(dir_save, f"{diff_diag_code}.json")
    save_json(diff_diag_features, save_path)

if __name__ == "__main__":
    kg_obj = KnowledgeGraph(args=None)
    qs_obj = QuestionSetter(kg_obj)

    diff_diags = kg_obj.get_differential_diagnosis_nodes().keys()
    for diff_diag_code in diff_diags:
        select_features(kg_obj, qs_obj, diff_diag_code, difficulty='high')
