
import time, argparse, json
import os
import multiprocessing
import math
from tqdm import tqdm
from prompts import get_eval_prompt, get_eval_prompt_single, get_eval_prompt_clear
from knowledge_graph import KnowledgeGraph # get_kg
from models import get_generative_model, request
from vllm import SamplingParams
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./../resources/dataset/{}')
    parser.add_argument('--output_dir', type=str, default='./../output_clear/{}')

    # Model settings
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-8B')

    # generation parameter
    parser.add_argument('--mode', type=str, default='main')
    parser.add_argument('--difficulty', type=str, default='high')
    parser.add_argument('--question_type', type=str, default='multiple')
    parser.add_argument('--prompt_style', type=str, default='normal', help='normal, single, clear')

    parser.add_argument('--api_key', type=str, default="sk")
    parser.add_argument('--tp', type=int, default=1)
    # Instruction settings
    parser.add_argument('--instruction_k', type=int, default=5)
    parser.add_argument('--temperature', type=int, default=0)

    return parser.parse_args()


import re

def extract_answer(text):
    if not isinstance(text, str): return set()
    text = text.upper()
    
    # 1. Remove common prefixes (e.g., "Answer:") to prevent misinterpretation
    text = re.sub(r'^(?:ANSWER|OUTPUT|SELECTION|THE ANSWER IS)[:\s-]*', '', text.strip())

    # 2. [Core Regex] Find the first valid answer sequence.
    #    It looks for an initial letter (A-D) followed optionally by separators (&, ,, and) and more letters.
    #    This captures patterns like "A", "A & B", "A, B", "A and C".
    #    re.search stops at the first match, effectively ignoring subsequent text like "But B is..."
    match = re.search(r'([A-D](?:\s*(?:,|&|and)\s*[A-D])*)', text)

    if match:
        # Extract only the alphabets from the matched chunk (e.g., "A & B" -> {'A', 'B'}) and return as a set
        return set(re.findall(r'[A-D]', match.group(1)))
    
    return set()


def generate_gt(text):
    """
    Extracts the option letters from the Ground Truth (GT) text.
    Supports both period (.) and closing parenthesis ()) as separators.
    
    Examples:
        "C. Bipolar II" -> {'C'}
        "A) Major Depressive" -> {'A'}
        "A & B) Both Conditions" -> {'A', 'B'}
    """
    if not isinstance(text, str):
        return set()
    
    text = text.upper().strip()
    
    # 1. Split the text at the first occurrence of either a period (.) or a closing parenthesis ())
    #    and take the first part (the prefix).
    #    This removes the disease name following the option (e.g., "Vitamin A" in the text).
    text = re.split(r'[.)]', text, 1)[0]
        
    # 2. Extract option letters (A, B, C, D) from the prefix.
    #    Using \b (word boundary) ensures we don't match parts of other words.
    return set(re.findall(r'\b[A-D]\b', text))


def save_files(args, file_name, samples):
    print('**saving files: {} samples at {}'.format(len(samples), file_name))
    full_path = args.output_dir.format(file_name)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as fp:
        json.dump(samples, fp, indent=4, sort_keys=False, ensure_ascii=False)


def print_eval_stats(eval_stats):
    """
    Prints the evaluation statistics in a formatted table.
    Calculates accuracy per disease code and the overall total.
    """
    # 1. Define header format with fixed widths for alignment
    # :<10 (Left align, 10 spaces), :>8 (Right align, 8 spaces)
    header = f"| {'Disease':<10} | {'Correct':>8} | {'Wrong':>8} | {'Total':>8} | {'Accuracy':>10} |"
    divider = "-" * len(header)
    
    print("=" * len(header))
    print(header)
    print(divider)

    # 2. Variables to aggregate total statistics dynamically
    # (In case eval_stats['all'] was not updated during the loop)
    agg_correct = 0
    agg_wrong = 0
    agg_total = 0

    # 3. Sort keys alphabetically, excluding 'all' for now
    disease_codes = sorted([k for k in eval_stats.keys() if k != "all"])

    for code in disease_codes:
        stat = eval_stats[code]
        
        # Safely get values, defaulting to 0 if missing
        corr = stat.get('correct', 0)
        wrong = stat.get('incorrect', 0)
        tot = stat.get('total', 0)

        # Accumulate for the final 'TOTAL' row
        agg_correct += corr
        agg_wrong += wrong
        agg_total += tot

        # Calculate accuracy (Prevent ZeroDivisionError)
        acc = (corr / tot * 100) if tot > 0 else 0.0

        # Print the row for the current disease code
        print(f"| {code:<10} | {corr:>8} | {wrong:>8} | {tot:>8} | {acc:>9.2f}% |")

    print(divider)

    # 4. Handle the 'TOTAL' row
    # Use 'all' key from input if it has data; otherwise, use aggregated values
    if 'all' in eval_stats and eval_stats['all']['total'] > 0:
        final_corr = eval_stats['all']['correct']
        final_wrong = eval_stats['all']['incorrect']
        final_total = eval_stats['all']['total']
    else:
        final_corr = agg_correct
        final_wrong = agg_wrong
        final_total = agg_total

    # Calculate overall accuracy
    final_acc = (final_corr / final_total * 100) if final_total > 0 else 0.0

    # Print the summary row with emphasis
    print(f"| {'TOTAL':<10} | {final_corr:>8} | {final_wrong:>8} | {final_total:>8} | {final_acc:>9.2f}% |")
    print("=" * len(header))




def worker_api_eval(subset_nodes, args, generative_models, chunk_id=0):
    # Initialize client per process
    client, tokenizer = get_generative_model(args)

    if args.prompt_style == 'normal':
        prompt_func = get_eval_prompt
    elif args.prompt_style == 'single':
        prompt_func = get_eval_prompt_single
    elif args.prompt_style == 'clear':
        prompt_func = get_eval_prompt_clear
    else:
        prompt_func = get_eval_prompt
    
    local_eval_stats = {
        "type3": {"all": {"correct": 0, "incorrect": 0, "total": 0}},
        "type4": {"all": {"correct": 0, "incorrect": 0, "total": 0}}
    }
    
    local_input_tokens = 0
    local_output_tokens = 0

    # We use position=chunk_id to try to stack bars
    for idx, disease_code in enumerate(tqdm(subset_nodes, desc=f'P{chunk_id}', position=chunk_id, leave=True)):
        local_eval_stats["type3"][disease_code] = {"correct": 0, "incorrect": 0, "total": 0}
        local_eval_stats["type4"][disease_code] = {"correct": 0, "incorrect": 0, "total": 0}

        for generative_model in generative_models:
            [main_disease_code, sub_disease_code] = disease_code.split('-')
            
            # --- Type 3 ---
            file_name_ambig = '{}/{}/{}/type3/{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_path_ambig = args.data_dir.format(file_name_ambig)
            
            if False and os.path.exists(full_path_ambig):
                with open(full_path_ambig, 'r', encoding='utf-8') as f:
                    datas_ambig = json.load(f)
            else:
                datas_ambig = {}

            output_file_name = '{}/{}/{}/{}/type3/{}_{}.json'.format(args.model, args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_output_path = args.output_dir.format(output_file_name)
            
            results = dict()
            if os.path.exists(full_output_path):
                with open(full_output_path, 'r', encoding='utf-8') as f:
                    try: results = json.load(f)
                    except: results = dict()
            
            # Process Loop
            if datas_ambig:
                for data in datas_ambig:
                    if data not in results:
                        question, option, answer = datas_ambig[data]['question'], datas_ambig[data]['options'], datas_ambig[data]['answer']
                        message = prompt_func(args, question, option, answer)
                        system_message = ""
                        res, input_tokens, output_tokens = request(args, client, message, system_message, tokenizer, return_usage=True)
                        res = res.strip() if res else ""
                        local_input_tokens += input_tokens
                        local_output_tokens += output_tokens
                        results[data] = {'question':question, 'option':option, 'answer':answer, 'response':res}
                
                # Save results
                save_files(args, output_file_name, results) 

                # Stats calculation
                for data in datas_ambig: 
                    if data in results:
                        res = results[data].get('response', '')
                        answer = datas_ambig[data]['answer']
                        
                        res_answer_set = extract_answer(res)
                        gt_set = generate_gt(answer)

                        local_eval_stats['type3']['all']['total']+=1
                        local_eval_stats['type3'][disease_code]['total']+=1
                        if gt_set == res_answer_set and len(gt_set) > 0:
                            local_eval_stats['type3']['all']['correct']+=1
                            local_eval_stats['type3'][disease_code]['correct']+=1
                        else:
                            local_eval_stats['type3']['all']['incorrect']+=1
                            local_eval_stats['type3'][disease_code]['incorrect']+=1

            # --- Type 4a ---
            file_name_cleara = '{}/{}/{}/type4/a_{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_path_cleara = args.data_dir.format(file_name_cleara)
            
            if os.path.exists(full_path_cleara):
                with open(full_path_cleara, 'r', encoding='utf-8') as f:
                    datas_a = json.load(f)
            else:
                datas_a = {}
            
            output_file_name = '{}/{}/{}/{}/type4/a_{}_{}.json'.format(args.model, args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_output_path = args.output_dir.format(output_file_name)
            
            results = dict()
            if os.path.exists(full_output_path):
                with open(full_output_path, 'r', encoding='utf-8') as f:
                    try: results = json.load(f)
                    except: results = dict()
            
            if datas_a:
                for data in datas_a:
                    if data not in results:
                        question, option, answer = datas_a[data]['question'], datas_a[data]['options'], datas_a[data]['answer']
                        message = prompt_func(args, question, option, answer)
                        system_message = ""
                        res, input_tokens, output_tokens = request(args, client, message, system_message, tokenizer, return_usage=True)
                        res = res.strip() if res else ""
                        local_input_tokens += input_tokens
                        local_output_tokens += output_tokens
                        results[data] = {'question':question, 'option':option, 'answer':answer, 'response':res}
                
                save_files(args, output_file_name, results)

                for data in datas_a:
                    if data in results:
                        res = results[data].get('response', '')
                        answer = datas_a[data]['answer']
                        res_answer_set = extract_answer(res)
                        gt_set = generate_gt(answer)

                        local_eval_stats['type4']['all']['total']+=1
                        local_eval_stats['type4'][disease_code]['total']+=1
                        if gt_set == res_answer_set and len(gt_set) > 0:
                            local_eval_stats['type4']['all']['correct']+=1
                            local_eval_stats['type4'][disease_code]['correct']+=1
                        else:
                            local_eval_stats['type4']['all']['incorrect']+=1
                            local_eval_stats['type4'][disease_code]['incorrect']+=1

            # --- Type 4b ---
            file_name_clearb = '{}/{}/{}/type4/b_{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_path_clearb = args.data_dir.format(file_name_clearb)
            
            if os.path.exists(full_path_clearb):
                with open(full_path_clearb, 'r', encoding='utf-8') as f:
                    datas_b = json.load(f)
            else:
                datas_b = {}

            output_file_name = '{}/{}/{}/{}/type4/b_{}_{}.json'.format(args.model, args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
            full_output_path = args.output_dir.format(output_file_name)

            results = dict()
            if os.path.exists(full_output_path):
                with open(full_output_path, 'r', encoding='utf-8') as f:
                    try: results = json.load(f)
                    except: results = dict()

            if datas_b:
                for data in datas_b:
                    if data not in results:
                        question, option, answer = datas_b[data]['question'], datas_b[data]['options'], datas_b[data]['answer']
                        message = prompt_func(args, question, option, answer)
                        system_message = ""
                        res, input_tokens, output_tokens = request(args, client, message, system_message, tokenizer, return_usage=True)
                        res = res.strip() if res else ""
                        local_input_tokens += input_tokens
                        local_output_tokens += output_tokens
                        results[data] = {'question':question, 'option':option, 'answer':answer, 'response':res}

                save_files(args, output_file_name, results)

                for data in datas_b:
                    if data in results:
                        res = results[data].get('response', '')
                        answer = datas_b[data]['answer']
                        res_answer_set = extract_answer(res)
                        gt_set = generate_gt(answer)

                        local_eval_stats['type4']['all']['total']+=1
                        local_eval_stats['type4'][disease_code]['total']+=1
                        if gt_set == res_answer_set and len(gt_set) > 0:
                            local_eval_stats['type4']['all']['correct']+=1
                            local_eval_stats['type4'][disease_code]['correct']+=1
                        else:
                            local_eval_stats['type4']['all']['incorrect']+=1
                            local_eval_stats['type4'][disease_code]['incorrect']+=1
    
    return local_eval_stats, local_input_tokens, local_output_tokens

if __name__ == "__main__":
    args = get_args()
    print(args)
    knowledge_graph_obj = KnowledgeGraph(args)


    nodes = sorted(list(knowledge_graph_obj.get_differential_diagnosis_nodes().keys()))

    generative_models = ['gemini', 'gpt5','anthropic']
    client, tokenizer=get_generative_model(args)


    eval_stats = {"type3": {
            "all": {
                "correct": 0,
                "incorrect": 0,
                "total": 0
            }
        },
        "type4": {
            "all": {
                "correct": 0,
                "incorrect": 0,
                "total": 0
            }
        }
    }

    if 'gpt' in args.model or 'gemini' in args.model or 'anthropic' in args.model:
        # ========================================
        # API-based models (GPT, Gemini, Qwen235)
        # Parallel inference (Multi-processing)
        # ========================================
        
        # Determine number of processes (max 5)
        num_processes = min(13, len(nodes))
        print(f"Starting {num_processes} processes for API evaluation...")

        # Chunk nodes
        chunk_size = math.ceil(len(nodes) / num_processes)
        chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        
        # Prepare arguments for each process
        process_args = []
        for i, chunk in enumerate(chunks):
            # (subset_nodes, args, generative_models, chunk_id)
            process_args.append((chunk, args, generative_models, i))

        # Run multiprocessing pool
        total_input_tokens = 0
        total_output_tokens = 0
        
        # Use simple starmap
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(worker_api_eval, process_args)

        # Aggregate results
        print("\nAggregating results...")
        for local_stats, loc_in, loc_out in results:
            total_input_tokens += loc_in
            total_output_tokens += loc_out
            
            for type_key in ['type3', 'type4']:
                # Aggregate 'all'
                eval_stats[type_key]['all']['correct'] += local_stats[type_key]['all']['correct']
                eval_stats[type_key]['all']['incorrect'] += local_stats[type_key]['all']['incorrect']
                eval_stats[type_key]['all']['total'] += local_stats[type_key]['all']['total']
                
                # Aggregate/Copy per-disease stats
                for disease, stat in local_stats[type_key].items():
                    if disease == 'all': continue
                    if disease not in eval_stats[type_key]:
                         eval_stats[type_key][disease] = stat
                    else:
                         eval_stats[type_key][disease]['correct'] += stat['correct']
                         eval_stats[type_key][disease]['incorrect'] += stat['incorrect']
                         eval_stats[type_key][disease]['total'] += stat['total']

        # Print total token usage for API models
        print('\n' + '='*50)
        print('API Token Usage Summary')
        print('='*50)
        print(f'Total Input Tokens: {total_input_tokens:,}')
        print(f'Total Output Tokens: {total_output_tokens:,}')
        print(f'Total Tokens: {total_input_tokens + total_output_tokens:,}')
        print('='*50 + '\n')
    
    
    else:
        # ========================================
        # vLLM-based models (Other models)
        # Batch inference
        # ========================================
        def build_text(system_message: str, user_message: str, tokenizer):
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )


        def generate_batch(client, texts, max_tokens=4096, top_p=0.95):
            sampling_params = SamplingParams(top_p=top_p, max_tokens=max_tokens)
            outputs = client.generate(texts, sampling_params)
            return [o.outputs[0].text for o in outputs]


        # =========================
        # 1) Collect all inputs into a single batch
        # =========================

        texts_type3 = []
        meta_type3 = []  

        texts_type4a = []
        meta_type4a = []

        texts_type4b = []
        meta_type4b = []

        if args.prompt_style == 'normal':
            prompt_func = get_eval_prompt
        elif args.prompt_style == 'single':
            prompt_func = get_eval_prompt_single
        elif args.prompt_style == 'clear':
            prompt_func = get_eval_prompt_clear
        else:
            prompt_func = get_eval_prompt

        system_message = ""

        for disease_code in tqdm(nodes, desc="prepare disease", leave=True):

            eval_stats["type3"][disease_code] = {"correct": 0, "incorrect": 0, "total": 0}
            eval_stats["type4"][disease_code] = {"correct": 0, "incorrect": 0, "total": 0}

            for generative_model in generative_models:
                [main_disease_code, sub_disease_code] = disease_code.split('-')
                
                file_name_ambig = '{}/{}/{}/type3/{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
                full_path_ambig = args.data_dir.format(file_name_ambig)
                
                file_name_cleara = '{}/{}/{}/type4/a_{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
                full_path_cleara = args.data_dir.format(file_name_cleara)
                
                file_name_clearb = '{}/{}/{}/type4/b_{}_{}.json'.format(args.difficulty, main_disease_code, sub_disease_code, args.mode, generative_model)
                full_path_clearb = args.data_dir.format(file_name_clearb)

                # Type 3 - Ambiguous
                if os.path.exists(full_path_ambig):
                    with open(full_path_ambig, "r", encoding="utf-8") as f:
                        datas_ambig = json.load(f)

                    for data_key in datas_ambig:
                        question = datas_ambig[data_key]["question"]
                        option = datas_ambig[data_key]["options"]
                        answer = datas_ambig[data_key]["answer"]

                        message = prompt_func(args, question, option, answer)
                        if 'mentallama' in args.model_name:
                            texts_type3.append('Question:'+message)
                        else:
                            text = build_text(system_message, message, tokenizer)
                            texts_type3.append(text)
                        
                        meta_type3.append(
                            (disease_code, generative_model, data_key, question, option, answer, main_disease_code, sub_disease_code)
                        )
                else:
                    print(f"No such file: {file_name_ambig}")

                # Type 4a - Clear A
                if os.path.exists(full_path_cleara):
                    with open(full_path_cleara, "r", encoding="utf-8") as f:
                        datas_a = json.load(f)

                    for data_key in datas_a:
                        question = datas_a[data_key]["question"]
                        option = datas_a[data_key]["options"]
                        answer = datas_a[data_key]["answer"]

                        message = prompt_func(args, question, option, answer)
                        if 'mentallama' in args.model_name:
                            texts_type4a.append('Question:'+message)
                        else
                            text = build_text(system_message, message, tokenizer)
                            texts_type4a.append(text)
                        
                        meta_type4a.append(
                            (disease_code, generative_model, data_key, question, option, answer, main_disease_code, sub_disease_code)
                        )
                else:
                    print(f"No such file: {file_name_cleara}")

                # Type 4b - Clear B
                if os.path.exists(full_path_clearb):
                    with open(full_path_clearb, "r", encoding="utf-8") as f:
                        datas_b = json.load(f)

                    for data_key in datas_b:
                        question = datas_b[data_key]["question"]
                        option = datas_b[data_key]["options"]
                        answer = datas_b[data_key]["answer"]

                        message = prompt_func(args, question, option, answer)
                        if 'mentallama' in args.model_name:
                            texts_type4b.append('Question:'+message)
                        else:
                            text = build_text(system_message, message, tokenizer)
                            texts_type4b.append(text)
                        
                        meta_type4b.append(
                            (disease_code, generative_model, data_key, question, option, answer, main_disease_code, sub_disease_code)
                        )
                else:
                    print(f"No such file: {file_name_clearb}")


        print(f"Total batch size - Type3: {len(texts_type3)}, Type4a: {len(texts_type4a)}, Type4b: {len(texts_type4b)}")


        # =========================
        # 2) Single batch inference
        # =========================

        all_texts = texts_type3 + texts_type4a + texts_type4b
        print(f"Total batch size: {len(all_texts)}")
        
        all_responses = generate_batch(client, all_texts, max_tokens=300, top_p=0.95)

        responses_type3 = all_responses[:len(texts_type3)]
        responses_type4a = all_responses[len(texts_type3):len(texts_type3)+len(texts_type4a)]
        responses_type4b = all_responses[len(texts_type3)+len(texts_type4a):]


        # =========================
        # 3) Restore results and evaluate
        # =========================

        results_type3 = defaultdict(dict)
        wrong_type3 = defaultdict(dict)

        results_type4a = defaultdict(dict)
        wrong_type4a = defaultdict(dict)

        results_type4b = defaultdict(dict)
        wrong_type4b = defaultdict(dict)

        # Type 3 evaluation
        for (
            disease_code,
            generative_model,
            data_key,
            question,
            option,
            answer,
            main_disease_code,
            sub_disease_code,
        ), res in tqdm(zip(meta_type3, responses_type3), total=len(responses_type3), desc="evaluation type3"):

            res = res.strip()

            results_type3[(disease_code, generative_model)][data_key] = {
                "question": question,
                "option": option,
                "answer": answer,
                "response": res,
            }

            res_answer_set = extract_answer(res)
            gt_set = generate_gt(answer)

            eval_stats["type3"]["all"]["total"] += 1
            eval_stats["type3"][disease_code]["total"] += 1

            if gt_set == res_answer_set and len(gt_set) > 0:
                eval_stats["type3"]["all"]["correct"] += 1
                eval_stats["type3"][disease_code]["correct"] += 1
            else:
                eval_stats["type3"]["all"]["incorrect"] += 1
                eval_stats["type3"][disease_code]["incorrect"] += 1
                wrong_type3[(disease_code, generative_model)][data_key] = {
                    "answer": answer,
                    "res": res,
                }

        # Type 4a evaluation
        for (
            disease_code,
            generative_model,
            data_key,
            question,
            option,
            answer,
            main_disease_code,
            sub_disease_code,
        ), res in tqdm(zip(meta_type4a, responses_type4a), total=len(responses_type4a), desc="evaluation type4a"):

            res = res.strip()

            results_type4a[(disease_code, generative_model)][data_key] = {
                "question": question,
                "option": option,
                "answer": answer,
                "response": res,
            }

            res_answer_set = extract_answer(res)
            gt_set = generate_gt(answer)

            eval_stats["type4"]["all"]["total"] += 1
            eval_stats["type4"][disease_code]["total"] += 1

            if gt_set == res_answer_set and len(gt_set) > 0:
                eval_stats["type4"]["all"]["correct"] += 1
                eval_stats["type4"][disease_code]["correct"] += 1
            else:
                eval_stats["type4"]["all"]["incorrect"] += 1
                eval_stats["type4"][disease_code]["incorrect"] += 1
                wrong_type4a[(disease_code, generative_model)][data_key] = {
                    "answer": answer,
                    "res": res,
                }

        # Type 4b evaluation
        for (
            disease_code,
            generative_model,
            data_key,
            question,
            option,
            answer,
            main_disease_code,
            sub_disease_code,
        ), res in tqdm(zip(meta_type4b, responses_type4b), total=len(responses_type4b), desc="evaluation type4b"):

            res = res.strip()

            results_type4b[(disease_code, generative_model)][data_key] = {
                "question": question,
                "option": option,
                "answer": answer,
                "response": res,
            }

            res_answer_set = extract_answer(res)
            gt_set = generate_gt(answer)

            eval_stats["type4"]["all"]["total"] += 1
            eval_stats["type4"][disease_code]["total"] += 1

            if gt_set == res_answer_set and len(gt_set) > 0:
                eval_stats["type4"]["all"]["correct"] += 1
                eval_stats["type4"][disease_code]["correct"] += 1
            else:
                eval_stats["type4"]["all"]["incorrect"] += 1
                eval_stats["type4"][disease_code]["incorrect"] += 1
                wrong_type4b[(disease_code, generative_model)][data_key] = {
                    "answer": answer,
                    "res": res,
                }


        # =========================
        # 4) Save results by disease_code / generative_model
        # =========================

        # Type 3 save
        for (disease_code, generative_model), res_dict in results_type3.items():
            [main_disease_code, sub_disease_code] = disease_code.split('-')
            output_file_name = (
                f"{args.model}/{args.difficulty}/{main_disease_code}/{sub_disease_code}/"
                f"type3/{args.mode}_{generative_model}.json"
            )
            save_files(args, output_file_name, res_dict)

        # Type 4a save
        for (disease_code, generative_model), res_dict in results_type4a.items():
            [main_disease_code, sub_disease_code] = disease_code.split('-')
            output_file_name = (
                f"{args.model}/{args.difficulty}/{main_disease_code}/{sub_disease_code}/"
                f"type4/a_{args.mode}_{generative_model}.json"
            )
            save_files(args, output_file_name, res_dict)

        # Type 4b save
        for (disease_code, generative_model), res_dict in results_type4b.items():
            [main_disease_code, sub_disease_code] = disease_code.split('-')
            output_file_name = (
                f"{args.model}/{args.difficulty}/{main_disease_code}/{sub_disease_code}/"
                f"type4/b_{args.mode}_{generative_model}.json"
            )
            save_files(args, output_file_name, res_dict)

        # Print per-disease stats
        for disease_code in nodes:
            total_type3 = eval_stats["type3"][disease_code]["total"]
            acc_type3 = eval_stats["type3"][disease_code]["correct"] / total_type3 if total_type3 > 0 else 0.0

            total_type4 = eval_stats["type4"][disease_code]["total"]
            acc_type4 = eval_stats["type4"][disease_code]["correct"] / total_type4 if total_type4 > 0 else 0.0

            print(f"[{disease_code}] Type4 accuracy: {acc_type4:.4f} ({total_type4})")

        wrong = {**wrong_type3, **wrong_type4a, **wrong_type4b}

    print('Tested on: {}'.format(args.model))
    print('Difficulty: {}'.format(args.difficulty))
    print_eval_stats(eval_stats['type3'])
    print_eval_stats(eval_stats['type4'])

    #import IPython; IPython.embed(); exit(1)
