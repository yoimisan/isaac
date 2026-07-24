#!/bin/bash
set -euo pipefail

PROJECT_DIR=$(cd -- $(dirname $(dirname -- "${BASH_SOURCE[0]}")) && pwd)

source ${PROJECT_DIR}/scripts/utils.sh

EPISODE_NUM=$1



for args in "${@:2}"; do
    DEVICE_ID=${args}
    RECORD_ROOT=${PROJECT_DIR}/data/${DEVICE_ID}
    LOG_FILE=${PROJECT_DIR}/logs/${DEVICE_ID}.log

    echo DEVICE: ${DEVICE_ID}, RECORD_ROOT: ${RECORD_ROOT}, LOGING: ${LOG_FILE}
    sleep 2

    if [[ -e ${RECORD_ROOT} ]]; then
        read -r -p "Found \"${RECORD_ROOT}\" exists, delete it?(y/n):" tmp
        if [[ "${tmp}" == "y" || "${tmp}" == "yes" ]]; then
            rm -rf ${RECORD_ROOT}
        else
            exit 0
        fi
    fi

    _execute ${DEVICE_ID} ${RECORD_ROOT} ${EPISODE_NUM} \
        > ${LOG_FILE} 2>&1 &
    
    sleep 2
done