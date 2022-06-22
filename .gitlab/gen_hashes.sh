#! /bin/bash

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Load common bash helper functions
# shellcheck source=SCRIPTDIR/../scripts/bash_compat.sh
. "$BASEPATH/../scripts/bash_compat.sh"

root_dir=$(realpath "$BASEPATH/..")

# ==============================================================================

build_script_hash=$(md5sum "$root_dir/build_locally.sh" | cut -f1 -d' ')
third_party_hash=$(find "$root_dir/third_party" -type f -exec md5sum {} \; | sort -k 2 | md5sum | cut -f1 -d' ')

overall_hash=$(echo "$build_script_hash $third_party_hash" | md5sum | cut -f1 -d' ')

# ==============================================================================

cat << EOF > "$BASEPATH/hashes.yml"
---

variables:
  BUILD_SCRIPT_HASH: ${build_script_hash}
  THIRD_PARTY_HASH: ${third_party_hash}
  OVERALL_HASH: ${overall_hash}
EOF
