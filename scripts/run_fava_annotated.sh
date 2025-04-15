methods=('rfm' 'linear' 'logistic' 'linear_rfm' 'rfm_linear')
# methods=('pca')
# models=('llama_3_8b_it' 'gemma_2_9b_it')
# models=('llama_3.3_70b_4bit_it')
models=('llama_3_8b_it')
# start_seed=0
for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        echo $method $model
        sbatch  --job-name="fava-$method" delta_setup "python -u run_fava_annotated.py --control_method $method --model $model"
    done
done


# methods=('llama' 'gemma') # 'openai'
# methods=('llama')
# methods=('openai')
# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="fava-$method" delta_setup "python -u run_fava_annotated_judge.py --judge_type $method"
# done