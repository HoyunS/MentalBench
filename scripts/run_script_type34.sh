model_list=(
    "google/gemini-2.5-flash"
    "google/gemini-2.5-pro"
    "anthropic/claude-haiku-4.5"
    "anthropic/claude-sonnet-4.5"
)

for model in "${model_list[@]}"; do
    echo "----------------------------------------"
    echo "Starting evaluation for model: $model" 
    filename="${model//\//_}" 

    python eval_type34.py --model "$model" --prompt_style "clear" --output_dir './../output_clear/{}' --api_key "openrouter-key" >> logs_clear/"${filename}"_high.txt 2>&1 &
    
    echo "Launched jobs for: $model (Logs will be saved as ${filename}_*.txt)"
done

model_list=(
    "gpt-4o"
    "gpt-5-mini"
    "gpt-5.1"
)
for model in "${model_list[@]}"; do
    echo "----------------------------------------"
    echo "Starting evaluation for model: $model" 
    filename="${model//\//_}" 

    python eval_type34.py --model "$model" --prompt_style "clear" --output_dir './../output_clear/{}' --api_key "gpt-key" >> logs_clear/"${filename}"_high.txt 2>&1 &
    
    echo "Launched jobs for: $model (Logs will be saved as ${filename}_*.txt)"
done


echo "Waiting for all background jobs to finish..."
wait
echo "All evaluations completed."
