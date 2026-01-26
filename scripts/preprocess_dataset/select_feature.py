import os, json
import copy
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))

from knowledge_graph import KnowledgeGraph
from question_setter import QuestionSetter


def save_selected_features_json(data: dict, disease_code: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{disease_code}.json")
    
    if os.path.exists(output_path):
        print(f"Selected features for disease {disease_code} already exists. Skipping.")
        return
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Selected features for disease {disease_code} saved to {output_path}")
    return


def save_selected_features_csv(df: pd.DataFrame, disease_code: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{disease_code}.csv")
    
    if os.path.exists(output_path):
        print(f"Selected features for disease {disease_code} already exists. Skipping.")
        return
    
    df.to_csv(output_path, index=False)
    print(f"Selected features for disease {disease_code} saved to {output_path}")
    return

def select_features(kg_obj: KnowledgeGraph, qs_obj: QuestionSetter,
                     disease_code: str, difficulty: str, iteration: int):
    disease_features = kg_obj.get_disease_features_lookup().get(disease_code, [])
    if not disease_features:
        print(f"No features found for disease {disease_code}. Skipping.")
        return
    
    df_selected_features = pd.DataFrame()
    json_selected_features = dict()
    i = 0
    while i < iteration:
        sampled_features_and_options = qs_obj.create_questions_from_disease(disease_code, difficulty_level=difficulty, n_options=4)

        # check duplicates
        if sampled_features_and_options in json_selected_features.values():
            print(f"Duplicate sampled features found for disease {disease_code} at iteration {i}. Skipping.")
            continue

        # save as json
        json_selected_features[i] = sampled_features_and_options

        temp = dict()

        def _traverse(obj, parent_key=''):
            if parent_key.split('-')[-1] == "sampled_symptom_groups":
                obj_copy = copy.deepcopy(obj)
                
                for sg, sg_content in obj.items():
                    symptom_list_txt = []
                    for symp_code in sg_content['symptoms']:
                        symptom_name = kg_obj.convert_code_to_name(symp_code)
                        symptom_list_txt.append(symptom_name)

                    # replace symptom codes with names
                    obj_copy[sg]['symptoms'] = symptom_list_txt
                    obj_copy[sg].pop('name', None)
                temp[parent_key] = str(obj_copy)
                return

            if parent_key.split('-')[-1] == "sampled_symptoms":
                return
            if parent_key.split('-')[-1] == "sampled_symptoms_descriptions":
                return

            if parent_key.split('-')[-1] == "answer_index":
                return
            if parent_key.split('-')[-1] == "options_info":
                return
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    
                    
                    key = f"{parent_key}-{k}" if parent_key else k
                    _traverse(v, key)
            else:
                temp[parent_key] = str(obj)

        _traverse(sampled_features_and_options)
        

        df_selected_features = pd.concat([df_selected_features, pd.DataFrame([temp])], ignore_index=True)
        i += 1

    # save
    save_selected_features_json(json_selected_features, disease_code, output_dir=f'./../resources/features/seed/sampled/{difficulty}/json')
    save_selected_features_csv(df_selected_features, disease_code, output_dir=f'./../resources/features/seed/sampled/{difficulty}/csv')

    return

if __name__ == "__main__":
    kg_obj = KnowledgeGraph(args=None)
    qs_obj = QuestionSetter(kg_obj)

    disease_nodes = kg_obj.get_disease_nodes()

    difficulty = 'low'
    n = 10  # number of features to select
    for disease_code in disease_nodes:
        select_features(kg_obj, qs_obj, disease_code, difficulty, n)


    difficulty = 'medium'
    n = 20  # number of features to select
    for disease_code in disease_nodes:
        select_features(kg_obj, qs_obj, disease_code, difficulty, n)

   