#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

# ==============================================================================

SRC_FILE="$BASEPATH/../../../.cppcheck.suppressions"
DST_FILE="$BASEPATH/filter_cppcheck.txt"

# ------------------------------------------------------------------------------

while read -r line; do
    if [ -z "$line" ]; then
        echo ''
    elif [[ "$line" =~ .*\*.* ]]; then
        :
    elif [[ "$line" =~ ([a-zA-Z_]+):(.*) ]]; then
        warn_id="${BASH_REMATCH[1]}"
        file_location="${BASH_REMATCH[2]}"

        file_location_print=$(printf '"%s"' "mindquantum/$file_location")
        printf '%-90s  "%s"\n' "$file_location_print" "$warn_id"
    fi
done < "$SRC_FILE" > "$DST_FILE"

# ==============================================================================
