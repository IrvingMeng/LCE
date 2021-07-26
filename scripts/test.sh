export DATAPATH=/face/irving/data/  # path to your Market1501
export OLD_MODEL=ckpts/log-arcface-half
# export NEW_MODEL=ckpts/log-arcface-full
export NEW_MODEL=ckpts/log-arcface-LCE_100_1


echo 'generate features from old model'
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --output ${OLD_MODEL} \
    --root  ${DATAPATH} \
    --dataset market1501 \
    --eval 1 \
    --resume ${OLD_MODEL}/best_model.pth.tar 


echo 'generate features from new model'  
python3 main.py \
    --cfg configs/res50_arcface.yaml \
    --output ${NEW_MODEL} \
    --root  ${DATAPATH} \
    --dataset market1501 \
    --eval 1 \
    --resume ${NEW_MODEL}/best_model.pth.tar 


echo 'compatible performance'
python3 eval/eval.py \
    --gallery_path ${NEW_MODEL} \
    --query_path ${OLD_MODEL} 
    
