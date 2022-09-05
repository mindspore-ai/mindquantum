#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

# ==============================================================================

SRC_FILE="$BASEPATH/../../../.whitelizard.txt"
DST_FILE="$BASEPATH/whitelizard.txt"

# ------------------------------------------------------------------------------

while read -r line; do
    if [ -z "$line" ]; then
        echo ''
    elif [[ "$line" =~ ^#.* ]]; then
        echo "$line"
    elif [[ "$line" =~ ^(mindquantum|ccsrc|tests)/(.*) ]]; then
        printf "mindquantum/%s/%s\n" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    else
        echo "$line"
    fi
done < "$SRC_FILE" > "$DST_FILE"

# ==============================================================================
