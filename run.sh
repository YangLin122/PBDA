for i in 0.0 0.1 0.2 0.5 1.0 2.0
do
    python -u train.py\
    --source D \
    --target A  \
    --lambda2 $i
done