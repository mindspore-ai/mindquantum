#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

# ==============================================================================

SRC_FILE="$BASEPATH/../../../.cppcheck.suppressions"
DST_FILE="$BASEPATH/filter_cppcheck.txt"

# ------------------------------------------------------------------------------

while read -r line; do
    if [ -z "$line" ]; then
        echo ''
    elif [[ "$line" =~ ([a-zA-Z_]+):([^*]+)/\* ]]; then
        # Match lines like this: 'XXXX:path/to/folder/*'
        #
        # The way the filtering works in ms-pipeline is essentially:
        # > cat ${WORKSPACE}/cppcheck_error/${filename}|grep "${keyXXX}" > /dev/null
        # where `keyXXX` is one of the fields between `"` in filter_cppcheck.txt
        #
        # -> if we simply add a line like `"path/to/folder/" "XXXX"` it should get filtered out alright

        warn_id="${BASH_REMATCH[1]}"
        file_location="${BASH_REMATCH[2]}/"  # NB: BASH_REMATCH[2] is the folder without the trailing '/*'
        file_location_print=$(printf '"%s"' "mindquantum/$file_location")
        printf '%-90s  "%s"\n' "$file_location_print" "$warn_id"
    elif [[ "$line" =~ ([a-zA-Z_]+):(.*) ]]; then
        warn_id="${BASH_REMATCH[1]}"
        file_location="${BASH_REMATCH[2]}"
        file_location_print=$(printf '"%s"' "mindquantum/$file_location")
        printf '%-90s  "%s"\n' "$file_location_print" "$warn_id"
    fi
done < "$SRC_FILE" > "$DST_FILE"

# ==============================================================================
