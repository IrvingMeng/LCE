export DATAPATH=/face/irving/data/  # path to your Market1501
export OLD_MODEL=ckpts/log-arcface-half
export NEW_MODEL=ckpts/log-arcface-LCE

# param for LCE
export LA=100
export LB=1


# generate the class centers

echo 'generate information from old model'
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --output ${OLD_MODEL} \
    --root  ${DATAPATH} \
    --dataset market1501 \
    --eval 1 \
    --save_lcefeat 1 \
    --resume ${OLD_MODEL}/best_model.pth.tar 


# train the lce
echo 'train the lce'
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --root  ${DATAPATH} \
    --dataset market1501 \
    --output ${NEW_MODEL}_${LA}_${LB} \
    --train_half 0 \
    --use_lce 1 \
    --path_ccb $OLD_MODEL \
    --lambda_a ${LA} \
    --lambda_b ${LB} \
    --gpu 0,1  #