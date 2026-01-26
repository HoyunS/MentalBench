import os, json
import copy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../')))
from knowledge_graph import KnowledgeGraph
from question_setter import QuestionSetter


def read_json(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data: dict, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Variated features saved to {file_path}")
    return

def variate(qs_obj, disease_code, difficulty, n_exp):
    # don't do if already done
    read_path = f"./../resources/features/seed/final/{difficulty}/json/{disease_code}.json"
    save_path = f"./../resources/features/variation/{difficulty}/json/{disease_code}.json"
    if os.path.exists(save_path):
        print(f"Variated features for disease {disease_code} at difficulty {difficulty} already exists. Skipping.")
        return
    
    os.makedirs(f"./../resources/features/variation/{difficulty}/json/", exist_ok=True)
    
    data = read_json(read_path)
    
    variated_data = dict()

    for key, item in data.items():
        for i_exp in range(n_exp):
            variated_key = f"f_{key}-exp_{i_exp}"

            # expression variation
            item_copy = copy.deepcopy(item)

            if difficulty in ['low', 'medium']:
                ## functional impairment
                #### it's already true/false, so no variation needed

                ## traumatic_stressor_required
                if 'traumatic_stressor_required' in item_copy['question']['sampled_features']['sampled_diagnostic_criteria']:
                    item_copy['question']['sampled_features']['sampled_diagnostic_criteria']['traumatic_stressor_required'] = {
                        'required': True
                    }
                
                ## psychosocial_stressor_required
                if 'psychosocial_stressor_required' in item_copy['question']['sampled_features']['sampled_diagnostic_criteria']:
                    item_copy['question']['sampled_features']['sampled_diagnostic_criteria']['psychosocial_stressor_required'] = {
                        'required': True
                    }
                
                ## subtype
                symptom_descriptions = item_copy['question']['sampled_features']['sampled_symptoms_descriptions']
                new_symptom_descriptions = {}
                for symp_key, symp_desc in symptom_descriptions.items():
                    new_symptom_descriptions[symp_key] = qs_obj.sample_symptom_expression_type(symp_desc)
                item_copy['question']['sampled_features']['sampled_symptoms_descriptions'] = new_symptom_descriptions
            elif difficulty == 'high':
                ## functional impairment
                #### it's already true/false, so no variation needed

                ## traumatic_stressor_required
                if 'traumatic_stressor_required' in item_copy['_removed_diagnostic_criteria']:
                    item_copy['_removed_diagnostic_criteria']['traumatic_stressor_required'] = {
                        'required': True
                    }
                
                ## psychosocial_stressor_required
                if 'psychosocial_stressor_required' in item_copy['_removed_diagnostic_criteria']:
                    item_copy['_removed_diagnostic_criteria']['psychosocial_stressor_required'] = {
                        'required': True
                    }
                
                ## subtype
                symptom_descriptions = item_copy['question_both']['sampled_features']['sampled_symptoms_descriptions']
                new_symptom_descriptions = {}
                for symp_key, symp_desc in symptom_descriptions.items():
                    new_symptom_descriptions[symp_key] = qs_obj.sample_symptom_expression_type(symp_desc)
                item_copy['question_both']['sampled_features']['sampled_symptoms_descriptions'] = new_symptom_descriptions
            else:
                raise ValueError(f"Invalid difficulty level: {difficulty}")
            
            # add to variated data
            variated_data[variated_key] = item_copy
        
    save_json(variated_data, save_path)
    print(f"Variated features for disease {disease_code} at difficulty {difficulty} completed: {len(variated_data)}.")

    return

if __name__ == "__main__":
    kg_obj = KnowledgeGraph(args=None)
    qs_obj = QuestionSetter(kg_obj)

    disease_nodes = kg_obj.get_disease_nodes()

    n_exp = 5  # number of variations per feature set

    difficulty = 'low'
    for disease_code in disease_nodes:
        try:
            variate(qs_obj, disease_code, difficulty, n_exp)
        except Exception as e:
            print(f"Error variating features for disease {disease_code} at difficulty {difficulty}: {e}")
            continue
        

    difficulty = 'medium'
    for disease_code in disease_nodes:
        try:
            variate(qs_obj, disease_code, difficulty, n_exp)
        except Exception as e:
            print(f"Error variating features for disease {disease_code} at difficulty {difficulty}: {e}")
            continue

    difficulty = 'high'
    diff_diags = kg_obj.get_differential_diagnosis_nodes().keys()
    for diff_diag_code in diff_diags:
        variate(qs_obj, diff_diag_code, difficulty, n_exp)