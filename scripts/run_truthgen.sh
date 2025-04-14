# methods=('rfm' 'linear' 'logistic' 'rfm_linear' 'linear_rfm' 'pca')
methods=('logistic')
models=('llama_3_8b_it' 'gemma_2_9b_it')
for method in ${methods[@]};
do
    for model in ${models[@]};
    do
        echo $method $model
        sbatch  --job-name="truthgen-$method" delta_setup "python -u run_truthgen.py --control_method $method --model $model"
    done
done

# methods=('llama' 'gemma')
# # methods=('llama')
# # methods=('openai')
# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="truthgen-$method" delta_setup "python -u run_truthgen_judge.py --judge_type $method"
# done