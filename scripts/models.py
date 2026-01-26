
from openai import OpenAI
import time, argparse, json
from tqdm import tqdm
import os

def get_generative_model(args):
    '''
    :param      args: get args with openai/google API key
    :return:    openai client (obj)
    '''

    model = None
    tokenizer = None

    if 'gpt' in  args.model:
        api_key = args.api_key
        model = OpenAI(api_key=api_key)

    elif 'gemini' in args.model or 'anthropic' in args.model:
        api_key = args.api_key
        model = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    else:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = LLM(model=args.model,tensor_parallel_size=args.tp,
                    gpu_memory_utilization=0.9
                    )

    return model, tokenizer



def generate_openai_model(args, client, messages, return_usage=False):
    cnt=0
    response = None
    model_name = args.model


    while True:
        try:
            cnt+=1
            if cnt==5:
                break
            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            break
            
        except Exception as e:
            print("Exception: ", e)
            time.sleep(10)

    if response == None:
        if return_usage:
            return None, 0, 0
        return None
    res = response.choices[0].message.content
    
    if return_usage:
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return res, input_tokens, output_tokens
    return res



def generate_openrouter(args, model, system_prompt, user_prompt, return_usage=False):
    while True:
        try:
            response = model.chat.completions.create(
                model=args.model,
                reasoning_effort = 'none',
                messages= [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                }
                            ]
                        }
                    ],
                )

            break
        except Exception as e:
            print(f"Fail to generate response with error: {e}")
            time.sleep(10)
    
    text = response.choices[0].message.content

    if return_usage:
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        return text, input_tokens, output_tokens
    return text

def request(args, client, message, system_message, tokenizer, return_usage=False): 
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message},
        ]

    if 'gpt' in args.model:
        if return_usage:
            res, input_tokens, output_tokens = generate_openai_model(args, client, messages, return_usage=True)
            return res, input_tokens, output_tokens
        else:
            res = generate_openai_model(args, client, messages)
    elif 'gemini' in args.model:
        if return_usage:
            res, input_tokens, output_tokens = generate_openrouter(args, client, system_message, message, return_usage=True)
            return res, input_tokens, output_tokens
        else:
            res = generate_openrouter(args, client, system_message, message)
    elif 'anthropic' in args.model:
        if return_usage:
            res, input_tokens, output_tokens = generate_openrouter(args, client, system_message, message, return_usage=True)
            return res, input_tokens, output_tokens
        else:
            res = generate_openrouter(args, client, system_message, message)


    else:
        from vllm import SamplingParams
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sampling_params = SamplingParams(top_p=0.95, max_tokens=300)
        output = client.generate([text], sampling_params)
        res = output[0].outputs[0].text

    if return_usage:
        return res, 0, 0
    return res