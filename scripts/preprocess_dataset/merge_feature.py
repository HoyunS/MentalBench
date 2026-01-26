import ast
import os, json
import random
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))

from knowledge_graph import KnowledgeGraph
from question_setter import QuestionSetter


def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_xlsx(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # sort by the DataFrame's index (or by column named 'index' if present)
    if 'index' in df.columns:
        df = df.sort_values(by='index').reset_index(drop=True)
    
    return df

def save_json(data: dict, difficulty: str, disease_code: str):
    file_path = f"./../resources/features/seed/final/{difficulty}/json/{disease_code}.json"
    if os.path.exists(file_path):
        print(f"Final features for disease {disease_code} at difficulty {difficulty} already exists. Skipping.")
        return
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Merged validated features saved to {file_path}")
    return

def merge(kg_obj: KnowledgeGraph, qs_obj: QuestionSetter, disease_code, difficulty):
    # read files
    read_path = f"./../resources/features/seed/validated/{difficulty}/"
    save_path = f"./../resources/features/seed/final/{difficulty}/json/"
    csv_path = f"csv/{disease_code}_done.xlsx"
    json_path = f"json/{disease_code}.json"
    json_origin = read_json(os.path.join(read_path, json_path))
    df_validated = read_xlsx(os.path.join(read_path, csv_path))

    json_validated = dict()
    for (json_index, json_item), (_, df_item) in zip(json_origin.items(), df_validated.iterrows()):
        assert str(json_index) == str(df_item['index']), f"Index mismatch: {json_index} vs {df_item['index']}"
        
        if df_item['validated'] != 1:
            continue    # skip if not validated
        
        # conpare [question][sampled_features][sampled_diagnostic_criteria][duration]
        origin_duration = json_item['question']['sampled_features']['sampled_diagnostic_criteria'].get('duration', None)
        df_duration = df_item['question-sampled_features-sampled_diagnostic_criteria-duration'] if 'question-sampled_features-sampled_diagnostic_criteria-duration' in df_item else None
        ## update duration if different
        if df_duration is not None:
            if df_duration != origin_duration:
                json_item['question']['sampled_features']['sampled_diagnostic_criteria']['duration'] = df_duration

        # add additional_requirements for easy and medium difficulty
        if difficulty in ['easy', 'medium']:
            additional_requirements = kg_obj.get_disease_features_lookup()[disease_code]['diagnostic_criteria'].get('additional_requirements', None)
            if additional_requirements is not None and additional_requirements != []:
                json_item['question']['sampled_features']['sampled_diagnostic_criteria']['additional_requirements'] = additional_requirements
                

        # compare [question][sampled_features][sampled_symptom_groups]
        origin_symptom_groups = json_item['question']['sampled_features'].get('sampled_symptom_groups', None)
        df_symptom_groups_str = df_item['question-sampled_features-sampled_symptom_groups']
        ## ensure proper JSON format
        # df_symptom_groups_str = ast.literal_eval(df_symptom_groups_str) if pd.notna(df_symptom_groups_str) else None
        # df_symptom_groups_str = json.dumps(df_symptom_groups_str)
        df_symptom_groups_str = df_symptom_groups_str.replace("'", '"')
        df_symptom_groups_str = df_symptom_groups_str.replace('True', 'true').replace('False', 'false')

        try:
            df_symptom_groups_json = json.loads(df_symptom_groups_str) if pd.notna(df_symptom_groups_str) else None
        except Exception as e:
            print(f"Error reading symptom groups for disease {disease_code}, index {json_index}")
            print(df_symptom_groups_str)
            raise Exception(f"JSON decode error: {e}")
        
        ## update symptom groups if different
        full_symptoms_pool = set()
        for sg_key, sg_content in df_symptom_groups_json.items():
            # if there is no 'name' field in the validated json
            if 'name' not in sg_content:
                sg_content['name'] = sg_key
            
            # symptom names to codes
            symptom_codes = []
            for symptom_name in sg_content['symptoms']:
                symptom_code = kg_obj.convert_name_to_code(symptom_name)
                symptom_codes.append(symptom_code)
            df_symptom_groups_json[sg_key]['symptoms'] = symptom_codes
            full_symptoms_pool.update(symptom_codes)

            # duration
            # use validated duration if exists

        json_item['question']['sampled_features']['sampled_symptom_groups'] = df_symptom_groups_json


        # update sampled_symptoms to include all symptoms from symptom groups
        json_item['question']['sampled_features']['sampled_symptoms'] = sorted(list(full_symptoms_pool))

        # update sampled_symptoms_descriptions accordingly
        symptom_descriptions = dict()
        for symptom_code in json_item['question']['sampled_features']['sampled_symptoms']:
            description = kg_obj.get_symptom_info(symptom_code)
            symptom_descriptions[symptom_code] = description
        json_item['question']['sampled_features']['sampled_symptoms_descriptions'] = symptom_descriptions
        

        
        # update options
        if disease_code in ["D007", "D009", "D011", "D012", "D018", "D019"]:
            option_candidates = qs_obj.list_options_for_question(json_item['question']['sampled_features'])
            must_contain_options, must_remove_options = qs_obj.get_contain_and_remove_options_for_special_cases(disease_code)
            
            mcq_option_dict = qs_obj.sample_options_for_mcq(option_candidates,
                                                      must_contain_option_list=must_contain_options,
                                                      must_remove_option_list=must_remove_options,
                                                      n_options=4,
                                                      n_highly_likely = 0, n_moderately_likely=0, n_less_likely=0)
                    
            json_item['option'] = {
                'options': mcq_option_dict['options'],
                'answer_index': mcq_option_dict['answer_index'],
                'index_types': mcq_option_dict['index_types'],
                'options_info': mcq_option_dict['options_info']
            }

        # add to validated json
        json_validated[json_index] = json_item

    # save validated json
    save_json(json_validated,
              difficulty, disease_code)
                

if __name__ == "__main__":
    kg_obj = KnowledgeGraph(args=None)
    qs_obj = QuestionSetter(kg_obj)

    disease_nodes = kg_obj.get_disease_nodes()

    difficulty = 'low'
    for disease_code in disease_nodes:
        try:
            merge(kg_obj, qs_obj, disease_code, difficulty)
        except Exception as e:
            print(f"Error merging features for disease {disease_code} at difficulty {difficulty}: {e}")
        

    difficulty = 'medium'
    for disease_code in disease_nodes:
        try:
            merge(kg_obj, qs_obj, disease_code, difficulty)
        except Exception as e:
            print(f"Error merging features for disease {disease_code} at difficulty {difficulty}: {e}")