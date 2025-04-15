methods=('rfm' 'linear' 'logistic' 'rfm_linear' 'linear_rfm')
# methods=('pca')
# models=('llama_3_8b_it' 'gemma_2_9b_it')
# models=('llama_3.3_70b_4bit_it')
models=('llama_3_8b_it')

for model in ${models[@]};
do
    for method in ${methods[@]};
    do
        echo $method $model
        sbatch  --job-name="he-wild-$method" delta_setup "python -u run_multiclass_halu_eval_wild.py --control_method $method --model $model"
    done
done
