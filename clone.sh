#!/bin/sh
set -ex
mkdir -p "$1"
cd "$1"
git init
git lfs install
git remote add origin "$2"
git fetch origin "$3" --depth=1
git checkout FETCH_HEAD
git reset --hard FETCH_HEAD
git lfs pull
du -sh .
