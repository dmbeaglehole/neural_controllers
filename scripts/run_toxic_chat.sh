methods=('rfm' 'linear' 'logistic' 'rfm_linear' 'linear_rfm' 'pca')
# methods=('logistic')
# models=('gemma_2_9b_it' 'llama_3_8b_it')
models=('llama_3_8b_it')
n_seeds=5
seeds=(0 1 2 3 4)
# seeds=(0)

for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        for seed in ${seeds[@]};
        do
            echo $method $model $seed
            sbatch  --job-name="toxic-chat-$method" delta_setup "python -u run_toxic_chat.py --control_method $method --n_seeds $n_seeds --model $model --seed $seed"
        done
    done
done


# methods=('llama' 'gemma')
# # methods=('openai' 't5-large-ft')
# # methods=('gemma')
# for method in ${methods[@]};
# do
#     echo $method $coef
#     sbatch  --job-name="toxic-chat-$method" delta_setup "python -u run_toxic_chat_judge.py --judge_type $method"
# done