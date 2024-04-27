python train.py --dataset 'clintox' \
                  --runseed '0' \
                  --batch_size '64' \
                  --mweight_decay '0.0005'  \
                  --mdropout '0.2'  \
                  --mattdropout '0.1' \
                  --mlr '0.0005'  \
                  --jingtiaolr '0.0002' \
                  --result_file 'result.txt'  \
                  --pretrained_file './pretrained_model/pre-trained_M2UMol.pt'