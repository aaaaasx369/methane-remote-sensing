#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: ./PUSH_TO_GITHUB.sh git@github.com:USERNAME/REPO.git"
  exit 2
fi

git init
git add .
git commit -m "Initial curated methane research code snapshot"
git branch -M main
git remote add origin "$1"
git push -u origin main
