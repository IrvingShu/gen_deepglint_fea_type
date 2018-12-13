nohup python -u ./src/main.py \
     --feature-dir=/workspace/data/deepgint-testdata-feature/model-r100-spa-m2.0-faces_emore-ep136-500  \
     --feature-dims=500 \
     --output=./model-r100-spa-m2.0-faces_emore-ep136-500d.bin > ./nohup.log 2>&1 &
