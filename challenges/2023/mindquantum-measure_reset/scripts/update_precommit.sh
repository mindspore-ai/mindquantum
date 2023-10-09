#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )
ROOTPATH="$BASEPATH/.."

# ==============================================================================

function get_max() {
    m=-1
    while [ -n "$1" ]; do
        if [ ${#1} -gt $m ]; then
            m=${#1}
        fi
        shift
    done
    echo "$m"
}


# ==============================================================================

fname=(config
       config-format
       config-gitee
       config-format-gitee)

exts=(.yaml
      .yml
      .yaml
      .yml)

# ==============================================================================

pargs=(-j4 --link --line-buffer)

nmax=$(get_max "${fname[@]}")
tagstrings=()
for f in "${fname[@]}"; do
    formatted=$(printf '[%s]' "$f")
    tagstrings+=("$(printf '%-*s' "$nmax" "$formatted")")
done

# ------------------------------------------------------------------------------

parallel "${pargs[@]}" --tagstring '{3}' pre-commit autoupdate -c "$ROOTPATH/.pre-commit-{1}{2}" ::: "${fname[@]}" ::: "${exts[@]}" ::: "${tagstrings[@]}"

# ==============================================================================
