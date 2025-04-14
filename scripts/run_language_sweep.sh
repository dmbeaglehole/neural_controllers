methods=('logistic')

langs=('spanish')
# langs=('chinese' 'german' 'spanish')
models=("gemma_2_9b_it")
coefs=(4.0 5.0 6.0 7.0 8.0 9.0)

for coef in ${coefs[@]};
do
    for lang in ${langs[@]};
    do
        for model in ${models[@]};
        do
            for method in ${methods[@]};
            do
                echo $lang $model $method
                sbatch  --job-name="$model-$lang-$method" delta_setup "python -u languages.py --dest $lang --model $model --control_method $method --coef $coef"
            done
        done
    done
done

models=("llama_3_8b_it")
coefs=(0.2 0.3 0.4 0.5 0.6 0.7)
for coef in ${coefs[@]};
do
    for lang in ${langs[@]};
    do
        for model in ${models[@]};
        do
            for method in ${methods[@]};
            do
                echo $lang $model $method
                sbatch  --job-name="$model-$lang-$method" delta_setup "python -u languages.py --dest $lang --model $model --control_method $method --coef $coef"
            done
        done
    done
done
