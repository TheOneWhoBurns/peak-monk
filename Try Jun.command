#!/bin/zsh
cd -- "${0:A:h}" || exit 1
.venv/bin/python scripts/try_jun.py
result=$?
if (( result != 0 )); then
  echo "Jun stopped. The message above explains why. Press Return to close."
  read -r
fi
exit "$result"
