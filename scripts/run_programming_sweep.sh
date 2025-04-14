methods=('logistic')

langs=('javascript')
models=("llama_3_8b_it")
coefs=(0.4 0.5 0.6 0.7 0.8)

for lang in ${langs[@]};
do
    for model in ${models[@]};
    do
        for method in ${methods[@]};
        do
            for coef in ${coefs[@]};
            do
                echo $lang $model $method $coef
                sbatch  --job-name="$model-$lang-$method" delta_setup "python -u programming.py --dest $lang --model $model --control_method $method --coef $coef"
            done
        done
    done
done

langs=('javascript')
models=("gemma_2_9b_it")
coefs=(4.0 5.0 6.0 7.0 8.0 9.0)
for lang in ${langs[@]};
do
    for model in ${models[@]};
    do
        for method in ${methods[@]};
        do
            for coef in ${coefs[@]};
            do
                echo $lang $model $method $coef
                sbatch  --job-name="$model-$lang-$method" delta_setup "python -u programming.py --dest $lang --model $model --control_method $method --coef $coef"
            done
        done
    done
done
