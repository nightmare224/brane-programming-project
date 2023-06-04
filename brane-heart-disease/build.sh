#!/bin/bash

ABSPATH=`readlink -f $0`
DIRPATH=`dirname $ABSPATH`
cd ${DIRPATH}

## log levels: DEBUG,INFO,WARN,ERROR,FATAL ##
function log() {
  timestamp=`date "+%Y-%m-%d %H:%M:%S"`
  echo "[${USER}][${timestamp}][${1}]: ${2}"
}

function build_package() {
  package_name=${1}

  log "INFO" "Building the ${package_name} package"
  err_msg="$(brane build ./packages/${package_name}/container.yml 2>&1)"
  ret_val=$?
  if [[ ${ret_val} -eq 0 ]]; then
    log "INFO" "The ${package_name} package build successfully"
  else
    log "ERROR" "Fail to build the ${package_name} package"
    exit 1
  fi
}

function build_dataset() {
  dataset_name=${1}

  log "INFO" "Building the ${dataset_name} data"
  err_msg="$(brane data build ./data/data.yml 2>&1)"
  ret_val=$?
  if [[ ${ret_val} -eq 0 ]]; then
    log "INFO" "The ${dataset_name} dataset build successfully"
  else
    log "ERROR" "Fail to build the ${dataset_name} dataset. ${err_msg}."
    exit 1
  fi
}

main() {

  log "INFO" "### Build Packages ###"
  build_package "preprocessing"
  build_package "model"
  build_package "visualization"

  log "INFO" "### Build Dataset ###"
  build_dataset "heart disease"
  
}

main "$@"
