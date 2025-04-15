methods=('linear' 'logistic' 'rfm_linear' 'linear_rfm')
# methods=('pca' 'rfm')
# models=('llama_3_8b_it' 'gemma_2_9b_it')
models=('llama_3.3_70b_4bit_it')

for method in ${methods[@]};
do
    for model in ${models[@]};
    do
        echo $method $model
        sbatch  --job-name="agentharm-$method" delta_setup "python -u run_agentharm.py --control_method $method --model $model"
    done
done

# methods=('llama' 'gemma')
# # methods=('llama')
# # methods=('openai')
# # methods=('gemma')
# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="agentharm-$method" delta_setup "python -u run_agentharm_judge.py --judge_type $method"
# done