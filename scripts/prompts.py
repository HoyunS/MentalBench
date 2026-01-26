def parse_kg(args, kg):
    #['disease_name', 'sampled_diagnostic_criteria', 'sampled_symptom_groups', 'sampled_symptoms', 'sampled_symptoms_descriptions']
    
    texts = ["**Clinical Presentation (Symptoms to Integrate):**"]
    for idx, symptoms in enumerate(kg['sampled_symptom_groups']):
        text = []
        symptom_group_name = kg['sampled_symptom_groups'][symptoms]['name']
        text.append("**{}. {} Requirements:**".format(idx+1, symptom_group_name))

        sym_descriptions = []
        for syms in kg['sampled_symptom_groups'][symptoms]['symptoms']:
            if 'subtypes' not in kg['sampled_symptoms_descriptions'][syms]:
                sym_descriptions.append("- {} ({})".format(kg['sampled_symptoms_descriptions'][syms]['name'], kg['sampled_symptoms_descriptions'][syms]['description']))
            else:
                for sym_name in kg['sampled_symptoms_descriptions'][syms]['subtypes']:
                    sym_descriptions.append("- {} ({})".format(sym_name, kg['sampled_symptoms_descriptions'][syms]['subtypes'][sym_name]))
        text.append('\n'.join(sym_descriptions))

        if 'duration' in kg['sampled_symptom_groups'][symptoms]:
            duration = kg['sampled_symptom_groups'][symptoms]['duration']
            text.append('Duration of {}: {}'.format(symptoms, duration))
        if 'functional_impairment_required' in kg['sampled_symptom_groups'][symptoms]:
            text.append('Functional Impairment Due to {}: {}'.format(symptoms, kg['sampled_symptom_groups'][symptoms]['functional_impairment_required']))
        texts.append('\n'.join(text))


    if 'sampled_diagnostic_criteria' in kg:
        criteria = kg['sampled_diagnostic_criteria']
        texts.append('**Critical Diagnostic Criteria:**')
        if 'duration' in criteria:
            texts.append('- Duration: {}'.format(criteria['duration']))
        if 'functional_impairment_required' in criteria:
            texts.append('- Functional Impairment: {}'.format(criteria['functional_impairment_required']))
        if 'psychosocial_stressor_required' in criteria:
            texts.append('- Psychosocial Stressor: {}'.format(criteria['psychosocial_stressor_required']))
        if 'traumatic_stressor_required' in criteria:
            texts.append('- Traumatic Stressor: {}'.format(criteria['traumatic_stressor_required']))
        if 'additional_requirements' in criteria:
            additionals = criteria['additional_requirements']
            texts.append('- Additional Requirements:')
            for idx, c in enumerate(additionals):
                texts.append('{}) {}'.format(idx+1, c))
    return '\n\n'.join(texts)


def parse_kg_clear(args, kg, additional_requirements=None):
    #['disease_name', 'sampled_diagnostic_criteria', 'sampled_symptom_groups', 'sampled_symptoms', 'sampled_symptoms_descriptions']
    texts = ["**Clinical Presentation (Symptoms to Integrate):**"]
    for idx, symptoms in enumerate(kg['sampled_symptom_groups']):
        text = []
        symptom_group_name = kg['sampled_symptom_groups'][symptoms]['name']
        text.append("**{}. {} Requirements:**".format(idx+1, symptom_group_name))

        sym_descriptions = []
        for syms in kg['sampled_symptom_groups'][symptoms]['symptoms']:
            if 'subtypes' not in kg['sampled_symptoms_descriptions'][syms]:
                sym_descriptions.append("- {} ({})".format(kg['sampled_symptoms_descriptions'][syms]['name'], kg['sampled_symptoms_descriptions'][syms]['description']))
            else:
                for sym_name in kg['sampled_symptoms_descriptions'][syms]['subtypes']:
                    sym_descriptions.append("- {} ({})".format(sym_name, kg['sampled_symptoms_descriptions'][syms]['subtypes'][sym_name]))
        text.append('\n'.join(sym_descriptions))

        if 'duration' in kg['sampled_symptom_groups'][symptoms]:
            duration = kg['sampled_symptom_groups'][symptoms]['duration']
            text.append('Duration of {}: {}'.format(symptoms, duration))
        if 'functional_impairment_required' in kg['sampled_symptom_groups'][symptoms]:
            text.append('Functional Impairment Due to {}: {}'.format(symptoms, kg['sampled_symptom_groups'][symptoms]['functional_impairment_required']))
        texts.append('\n'.join(text))


    if additional_requirements is not None:
        texts.append('- Additional Requirements:')
        for idx, c in enumerate(additional_requirements):
            texts.append('{}) {}'.format(idx+1, c))
            
    return '\n\n'.join(texts)


def get_demographic_prompt(demographics):
    demographic = """
    **Patient Demographics:**
- **Age:** {}
- **Gender:** {}
- **Occupation:** {}
- **Marital Status:** {}
""".format(demographics['Age'], demographics['Gender'], demographics['Occupation'], demographics['Marital Status'])
    return demographic

def get_option_prompt(options):
    option = """
    **Multiple Choice Options:**
Please use the following options exactly as listed in your JSON output. Do not change the order.
A) {}
B) {}
C) {}
D) {}

**Task:**
Generate the clinical case summary and the JSON output based on the provided options.
    """.format(options[0], options[1], options[2], options[3])
    return option






def get_system_prompt_low(args):

    message = """
    You are a psychiatrist. 
    After interviewing a patient, document the case presentation in medical chart format.

    # Goal
    Create a professional **Clinical Case Summary** (Third-person perspective) based on the target symptoms and demographics.

    # Guidelines

    **Perspective:**
    - Write in third-person
    - Refer to the patient as "The patient" or "A [Age]-year-old [Gender]"

    **Patient Demographics:**
    - Include age, gender, marital status, occupation in the opening
    - Avoid temporal modifiers (e.g., "recently retired", "newly divorced")
    - IMPORTANT: Provided demographics are NOT stressors or causes of symptoms
    - For potentially stressor-like demographics (e.g., retirement, divorce, bereavement):
        a. Add a statement showing clear temporal separation between the demographic event and symptom onset
        b. The demographic event should have occurred significantly earlier than symptom onset

    **Terminology & Tone:**
    - Use standard medical terminology (avoid patient slang)
    - Maintain an objective, clinical tone

    **Constraints:**
    - Do not reveal the diagnosis name
    - Do not directly quote or copy the provided symptom descriptions

    **Requirements:**
    - Include the duration in clinical format
    - Incorporate all provided symptom categories as clinical observations
    - Keep the summary concise (approximately 150-200 words)

    """
    # Output
    #"(The clinical case summary)"
    return message


def get_system_prompt_medium(args):
#    message = """You are a psychiatric case writer. With the given symptoms, you must express the provided symptoms using varied wording. 
#Please create a description that describes a patient's health condition in 100 words."""


    message="""
    You are a patient experiencing mental health issues. 
    Describe your symptoms during a clinical interview with a psychiatrist.

    # Goal
    Create a realistic **Patient's Subjective Description** (Vignette) based on the target symptoms and demographics.

    # Guidelines

    **Perspective:**
    - Write in first-person ("I")

    **Demographic & Persona Integration (CRITICAL):**
    - **Adopt the Persona:** Use vocabulary, sentence structure, and concerns appropriate for the provided **Age, Gender, and Occupation**.
    - **Contextualize Symptoms:** Weave the symptoms into the specific lifestyle context.

    **Constraints:**
    - DO NOT explicitly state the diagnosis name
    - DO NOT list demographics robotically; show them through context.
	- DO NOT directly quote or copy the provided symptom descriptions

    **Requirements:**
    - Include the duration naturally in your description
    - Incorporate all provided symptom categories
    - Keep your description concise (approximately 150-200 words)

    
    """
    # Output
    #"(The patient vignette)"
    return message


def get_system_prompt_high_clear(args):
    message="""
    You are a patient experiencing mental health issues. 
    Describe your symptoms during a clinical interview with a psychiatrist.

    # Goal
    Create a realistic **Patient's Subjective Description** (Vignette) based on the target symptoms, demographics, additional condition (if provided), and provided rule.

    # Guidelines

    **Perspective:**
    - Write in first-person ("I")

    **Demographic & Persona Integration (CRITICAL):**
    - **Adopt the Persona:** Use vocabulary, sentence structure, and concerns appropriate for the provided **Age, Gender, and Occupation**.
    - **Contextualize Symptoms:** Weave the symptoms into the specific lifestyle context.

    **Symptom Integration (CRITICAL):**
    - **Core Symptoms:** You MUST include ALL provided symptoms comprehensively in your description.
    - **Additional Condition (If Provided):** 
      - If an additional condition is specified, you MUST also include manifestations of this condition naturally within your description.
      - The additional symptoms should feel concurrent and interwoven with core symptoms, not listed separately.
      - The additional symptoms should NOT dominate; they are co-occurring, not primary.
      - If no additional condition is provided, generate the vignette using ONLY the core symptoms.
    - **Rule Application (MOST CRITICAL):**
      - You will be provided with a specific rule for the target diagnosis.
      - Your description MUST clearly and unambiguously satisfy this rule.
      - Refer to the provided additional condition and key difference to naturally and explicitly incorporate the rule into your description.

    **Constraints:**
    - DO NOT explicitly state the diagnosis name
    - DO NOT list demographics robotically; show them through context
    - DO NOT directly quote or copy the provided symptom descriptions

    **Requirements:**
    - Include all provided symptoms, ensuring each is represented according to its description
    - Include the duration naturally in your description
    - Explicitly incorporate the provided rule into your description so that it clearly points to the target diagnosis 
    - Keep your description concise (approximately 150-200 words)

    """
    return message


def get_system_prompt_high_ambiguous(args):
    message="""
    You are a patient experiencing mental health issues. 
    Describe your symptoms during a clinical interview with a psychiatrist.

    # Goal
    Create a realistic **Patient's Subjective Description** (Vignette) based on the target symptoms, demographics, and additional condition (if provided).

    # Guidelines

    **Perspective:**
    - Write in first-person ("I")

    **Demographic & Persona Integration (CRITICAL):**
    - **Adopt the Persona:** Use vocabulary, sentence structure, and concerns appropriate for the provided **Age, Gender, and Occupation**.
    - **Contextualize Symptoms:** Weave the symptoms into the specific lifestyle context.

    **Symptom Integration (CRITICAL):**
    - **Core Symptoms:** You MUST include ALL provided symptoms comprehensively in your description.
    - **Additional Condition (If Provided):** 
      - If an additional condition is specified, you MUST also include manifestations of this condition naturally within your description.
      - The additional symptoms should feel concurrent and interwoven with core symptoms, not listed separately.
      - The additional symptoms should NOT dominate; they are co-occurring, not primary.
      - If no additional condition is provided, generate the vignette using ONLY the core symptoms.

    **Constraints:**
    - DO NOT explicitly state the diagnosis name
    - DO NOT list demographics robotically; show them through context.
    - DO NOT directly quote or copy the provided symptom descriptions

    **Requirements:**
    - Include all provided symptoms, ensuring each is represented according to its meaning
    - Include the duration naturally in your description
    - The generated description should be ambiguous enough that both disease could be considered as potential diagnoses
    - Keep your description concise (approximately 150-200 words)

    """
    return message




def get_eval_prompt_clear(args, question, option, answer):
    # "Select all applicable answers from the options (A, B, C, D)."
    query = """{}
Select **one or more** applicable answers from the options (A, B, C, D) based on the patient's presentation.
You must determine whether a **single** diagnosis or **multiple** diagnoses are correct.

**Output Rules:**
1. If only **one** option is correct, output just the letter (e.g., A).
2. If **multiple** options are correct, separate them with " & " (e.g., A & B).
3. Do NOT provide any explanation, reasoning, or introduction. Just the letters.

**Output Examples:**
- Single Answer: A.
- Multiple Answers: A & B.

{}
Answer:
""".format(question, option)
    return query


def get_eval_prompt(args, question, option, answer):
    # "Select all applicable answers from the options (A, B, C, D)."
    query = """{}
Q: Which of the following diagnoses are consistent with the patient's presentation? (Select all that apply)
Do NOT provide any explanation, reasoning, or introduction. Just the letter.
**Output Examples:**
- Single Answer: A.
- Multiple Answers: A & B.

{}
Answer:
""".format(question, option)
    return query


def get_eval_prompt_single(args, question, option, answer):
    #"Select the single best answer from the options (A, B, C, D)."
    query = """{}
Q: What is the most likely diagnosis?
Select the single best answer from the options (A, B, C, D).
Do NOT provide any explanation, reasoning, or introduction. Just the letter.
**Output Examples:**
- Single Answer: A.

{}
Answer:
""".format(question, option)
    return query




def get_type_A_user_prompt(args, features, options, demographics):

    demographic = get_demographic_prompt(demographics)


    # ['disease_name', 'sampled_diagnostic_criteria', 'sampled_symptom_groups', 'sampled_symptoms', 'sampled_symptoms_descriptions']
    disease_name = features['disease_name']
    names = """**Diagnosis Context:**
- **Target Disease (Correct Answer):** {}
""".format(disease_name)

    message = parse_kg(args, features)
    
    # option = get_option_prompt(options)

    # message = demographic+names+message+option

    message = demographic+names+message
    return message




def get_type_B_ambiguous_user_prompt(args, features, options, demographics, main_disease, sub_disease, additional_condition, key_difference, rule_main, rule_sub):
    demographic = get_demographic_prompt(demographics)

    names = """
**Diagnosis Context (Ambiguous / Insufficient Evidence):**
- **Suspected Condition A:** {}
- **Suspected Condition B:** {}
- **Additional Condition:** {}
*Note: The patient presents with symptoms relevant to both but LACKS definitive criteria for either.*
    """.format(main_disease, sub_disease, additional_condition)

    symptoms = parse_kg(args, features['question_both']['sampled_features'])

    constraints = """
**Ambiguity Logic Constraints (CRITICAL - The "Void" Logic):**
- **Goal:** Create a scenario where a definitive diagnosis is IMPOSSIBLE based on the text provided.
- **Missing Evidence for {}:** {} -> **MUST be vague or explicitly shorter/absent.**
- **Missing Evidence for {}:** {} -> **MUST be vague or unconfirmed.**
- **Constraint:** DO NOT include any pathognomonic features (determining symptoms) that would confirm A or B. Keep the severity and duration strictly in the "gray zone".
""".format(main_disease, rule_main, sub_disease, rule_sub)

    #option = get_option_prompt(options)

    message = demographic+names+symptoms+constraints

    return message


def get_type_B_clear_user_prompt(args, features, options, demographics, main_disease, sub_disease, additional_condition, key_difference, rule_main, rule_sub, additional_requirements):
    demographic = get_demographic_prompt(demographics)

    names = """
**Diagnosis Context:**
- **Target Disease (Correct Answer):** {}
- **Differential Diagnosis (Main Distractor):** {}
    """.format(main_disease, sub_disease)

    symptoms = parse_kg_clear(args, features['question_both']['sampled_features'], additional_requirements)

    constraints = """
**Differential Logic Constraints (CRITICAL):**
- **Additional condition:** {}
- **Key Difference:** {}
- **Rule for Target ({}):** {}
- **Rule for Differential ({}):** {}
""".format(additional_condition, key_difference, main_disease, rule_main, sub_disease, rule_sub)

    #option = get_option_prompt(options)

    message = demographic+names+symptoms+constraints

    return message


if __name__ == "__main__":

    from knowledge_graph import KnowledgeGraph
    from question_setter import QuestionSetter
    from demographics import get_demographics

    from main import get_args
    args = get_args()


    knowledge_graph_obj = KnowledgeGraph(args)
    qs = QuestionSetter(knowledge_graph_obj)

    nodes = knowledge_graph_obj.get_differential_diagnosis_nodes()
    disease_code = "D009-D008"


    features = qs.create_questions_from_differential_diagnosis(disease_code, difficulty_level='high',
            n_options=4, n_highly_likely=1, n_moderately_likely=1, n_less_likely=0)


    main_disease = knowledge_graph_obj.convert_code_to_name(features['question_both']['main_disease'])
    sub_disease = knowledge_graph_obj.convert_code_to_name(features['question_both']['sub_diseases'])

    demographics = get_demographics(args)

    additional_condition = features['question_both']['additional_condition']
    key_difference = features['question_both']['key_difference']
    rule_main = features['question_a']['rule']
    rule_sub = features['question_b']['rule']
    message = get_type_B_clear_user_prompt(args, features['question_both']['sampled_features'], features['option_both']['options'], 
                    demographics, main_disease, sub_disease, additional_condition, key_difference, rule_main, rule_sub)

    



    import IPython; IPython.embed(); exit(1)

