methods=('rfm' 'linear' 'logistic' 'linear_rfm' 'rfm_linear')
# methods=('pca')
# models=('llama_3_8b_it' 'gemma_2_9b_it')
# models=('llama_3.3_70b_4bit_it')
models=('llama_3_8b_it')
for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        echo $method $model
        sbatch  --job-name="halu-eval-$method" delta_setup "python -u run_halu_eval.py --control_method $method --model $model"
    done
done

for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        echo $method $model
        sbatch  --job-name="halu-eval-gen-$method" delta_setup "python -u run_halu_eval.py --control_method $method --hal_type general --model $model"
    done
done



# methods=('llama' 'gemma') # 'openai'
# methods=('openai')

# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="halu-eval-$method" delta_setup "python -u run_halu_eval_judge.py --judge_type $method"
# done

# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="halu-eval-gen-$method" delta_setup "python -u run_halu_eval_judge.py --judge_type $method --hal_type general"
# done