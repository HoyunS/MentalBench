model_list=(
    "Qwen/Qwen3-8B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-32B"
    # "Qwen/Qwen3-235B-A22B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "google/gemma-3-4b-it"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-it"
    # "klyang/MentaLLaMA-chat-13B"
    # "klyang/MentaLLaMA-chat-7B"
)

for model in "${model_list[@]}"; do
    echo "----------------------------------------"
    echo "Evaluating model: $model"

    filename="${model//\//_}" 
    python eval_type12.py --model "$model" --tp 8 --difficulty low   --output_dir './../output/{}' >> logs/"${filename}"_low.txt 2>&1
    python eval_type12.py --model "$model" --tp 8 --difficulty medium   --output_dir './../output/{}' >> logs/"${filename}"_medium.txt 2>&1
    
    echo "Done: $model (Log saved as ${filename}_*.txt)"
done


