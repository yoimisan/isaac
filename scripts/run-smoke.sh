#!/bin/bash
set -euo pipefail

DEVICE_ID=${1:-0}
MODE=${2:-1}
LOG_DIR="logs/data_collection"
RECORD_ROOT="data/tmp/smoke_clean"


PROJECT_DIR=$(cd -- $(dirname $(dirname -- "${BASH_SOURCE[0]}")) && pwd)
source ${PROJECT_DIR}/scripts/utils.sh

pre_clean () {
    mkdir -p ${LOG_DIR}
    if [[ -e ${RECORD_ROOT} ]]; then
        echo "${RECORD_ROOT} exists, delete."
        rm -rf ${RECORD_ROOT}
    fi
}

run_single () {
    echo "Run single smoke test."
    sleep 3

    _execute ${DEVICE_ID} ${RECORD_ROOT} 1
}

run_multi_thread () {
    echo "Run multi thread smoke test."
    sleep 3

    for WORKER_ID in 0 1; do
        RECORD_DIR="${RECORD_ROOT}/worker_${WORKER_ID}"

        _execute ${DEVICE_ID} ${RECORD_DIR} 3 \
            > "${LOG_DIR}/worker_${WORKER_ID}.log" 2>&1 &  
        echo "Started worker: ${WORKER_ID}"
        sleep 2
    done
}


pre_clean


if [[ ${MODE} -eq 1 ]]; then
    run_single
elif [[ ${MODE} -eq 2 ]]; then
    run_multi_thread 
else
    echo "Invalide mode: ${MODE}"
fi