#!/bin/bash

# Short bash script to extract all tags used in a training file
# The file must be provided through stdin, and the result is dumped to the
# stdout.
# You must specify the number of the column you want to work with. That
# col number will be extracted, splitted by the ' ' separator and computed
# its unique values, which will be written to stdout.

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 column_index"
  exit 1
fi

col_index=$1
cat /dev/stdin|while IFS=, read -ra ARR
do
  echo ${ARR[$col_index]}
done | sort | uniq | tr ' ' '\n' | sort | uniq > /dev/stdout
exit 0
