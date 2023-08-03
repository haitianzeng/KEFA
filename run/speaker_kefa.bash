name="Speaker_kefa_1"
flag="--attn soft --angleFeatSize 128
      --train speaker
      --model_name "Speaker_kefa"
      --hparams "weight_drop=0.95,w_loss_att=0.125,warmup_iter=10000,fact_dropout=0.0,top_k_facts=3"
      --subout max --dropout 0.55 --optim adam --lr 0.0002 --iters 120000 --maxAction 35 --featdropout 0.3 --batchSize 64"
mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=./build python3 r2r_src/train.py $flag --name $name