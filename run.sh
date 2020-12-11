for i in 'entropy' 'threshold' 'max_value'
do
    python -u train.py\
    --source D \
    --target A  \
    --weights_type $i
done