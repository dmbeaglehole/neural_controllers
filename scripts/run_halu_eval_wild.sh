methods=('rfm' 'rfm_linear' 'linear_rfm' 'rfm_logistic' 'logistic' 'linear')
# methods=('logistic')
models=('llama_3_8b_it' 'gemma_2_9b_it')  
for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        echo $method $model
        sbatch  --job-name="$method-$model" delta_setup "python -u run_multiclass_halu_eval_wild.py --control_method $method --model_name $model"
    done
done

# models=('llama' 'gemma') # 'openai')  
# models=('openai')
# for model in ${models[@]};
# do
#     echo $model
#     sbatch  --job-name="he-wild-$model" delta_setup "python -u run_multiclass_halu_eval_wild_judge.py --judge_type $model"
# done



# python -u run_halu_eval_wild.py --control_method rfm --hal_type out-of-scope --model gemma_2_9b_it --sample_weight inverse --seed 19
