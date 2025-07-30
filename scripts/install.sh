#! /bin/bash

python scripts/install.py ${@:1}

if [ -z "${INFINI_ROOT}" ]; then
    sleep 1
    . ~/.bashrc
fi
