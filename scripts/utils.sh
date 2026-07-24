_execute () {
    local device_id=${1}
    local record_root=${2}
    local episode_num=${3}

    CUDA_VISIBLE_DEVICES=${device_id} python src/pnp.py \
    --headless \
    --record \
    --record-root ${record_root} \
    --record-episodes ${episode_num} \
    --record-fps 60 \
    --/renderer/multiGpu/enabled=false \
    --/renderer/multiGpu/maxGpuCount=1 \
    --/renderer/activeGpu=${device_id} \
    --/physics/cudaDevice=0
}