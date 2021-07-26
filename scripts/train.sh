export DATAPATH=/face/irving/data/  # path to your Market1501


export MODEL=ckpts/log-arcface-half
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --root  ${DATAPATH} \
    --dataset market1501 \
    --output ckpts/log-arcface-half \
    --train_half 1 \
    --gpu 0,1  #

wait 

export MODEL=ckpts/log-arcface-full
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --root /face/irving/data/ \
    --dataset market1501 \
    --output ckpts/log-arcface-full \
    --train_half 0 \
    --gpu 0,1  #