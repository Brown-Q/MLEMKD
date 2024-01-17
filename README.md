## Quick Start

### Environment variables & dependencies
```
conda create -n MLEMKD python=3.7

conda activate MLEMKD

pip install -r requirement.txt
```

### Process data
First, unzip and unpack the data files 
```
tar -zxvf data-release.tar.gz
```
For the two ICEWS datasets `ICEWS14` and `ICEWS05-15`, go into the dataset folder in the `./data` directory and run the following command to construct the static graph.
```
cd ./data/<dataset>
python ent2word.py
```

### Train models
Then the following commands can be used to train the TKGR models. 

1. Make dictionary to save models
```
mkdir models
```

2. Pre-train models
```
cd src
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation pretrain --stage stage1 --role student

python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation pretrain --stage stage1 --role teacher
```

3. First-stage distillation
```
cd src
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation distill --stage stage1 --role student
```

4. Second-stage distillation
```
cd src
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation distill --stage stage2 --role teacher

python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation distill --stage stage2 --role student
```


### Evaluate models
To generate the evaluation results of models, simply add the `--test` flag in the commands above. 

```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n_layers_teacher 2 --n_layers_student 1 --evaluate-every 1 --gpu=0 --n_hidden_teacher 400 --n_hidden_student 100 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --situation pretrain --stage stage1 --role student --test
```
