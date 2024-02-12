container_name=$1

cd ..
CUDA_VISIBLE_DEVICES='0'

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --name $container_name \
    --mount src=$(pwd),dst=/flybyml,type=bind \
    --mount src=/media/data2/,dst=/data,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -p 7777:8888 \
    -w /flybyml \
    litcoderr/flybyml:latest \
    bash -c "bash" \