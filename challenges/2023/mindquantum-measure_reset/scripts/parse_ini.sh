#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shellcheck disable=SC2154

BASEPATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )

[ "${_sourced_parse_ini}" != "" ] && return || _sourced_parse_ini=.

# ==============================================================================

# shellcheck source=SCRIPTDIR/bash_compat.sh
. "$BASEPATH/bash_compat.sh"

# ==============================================================================

if [ -z "$AWK" ]; then
    if command -v gawk >/dev/null 2>&1; then
        AWK='gawk'
    elif command -v awk >/dev/null 2>&1; then
        AWK='awk'
    else
        echo 'ERROR: Unable to locate gawk or awk!' 1>&2
    fi
fi

# ==============================================================================

function read_ini_sections() {
    if [ -z "$AWK" ]; then
        return
    fi
    local filename="$1"
    # shellcheck disable=SC2016
    $AWK -v i=0 '{
         if ($1 ~ /^\[/) {
           gsub(/\[/, "", $1);
           gsub(/\]/, "", $1);
           section=tolower($1);
           # section=tolower(gensub(/\[(.+)\]/,"\\1",1,$1));
           configuration[i++]=section;
         }
       }
       END {
         for (idx in configuration) {
           key=configuration[idx];
           gsub( "-", "_", key);
           print key;
         }
       }' "${filename}"
}

# ------------------------------------------------------------------------------

# parse_ini_file <filename>
function parse_ini_file() {
    if [ -z "$AWK" ]; then
        return
    fi

    local filename sections awk_args eval_str

    filename="$1"
    sections="$(read_ini_sections "$filename")"

    # NB: cleanup any variables that may come from another run of this function
    unset "${!configuration_@}"

    declare_A configuration_sections
    configuration_sections+=( main )
    # shellcheck disable=SC2034
    declare_AA configuration_main
    for section in $sections; do
        array_name="configuration_${section//./_}"
        declare_AA "${array_name}"
        configuration_sections+=( "${section}" )
    done

    awk_args=(-F=)
    # shellcheck disable=SC2016

    if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        eval_str=$($AWK "${awk_args[@]}" '{
              if (section == "") { section="main"; }
              if ($1 ~ /^\[/) {
                gsub(/\[/, "", $1);
                gsub(/\]/, "", $1);
                section=tolower($1);
                # section=tolower(gensub(/\[(.+)\]/,"\\1",1,$1));
              }
              else if ($1 !~ /^$/ && $1 !~ /^;|#/) {
                gsub(/^[ \t]+|[ \t]+$/, "", $1);
                if ($1 ~ /[.*\[\]]/) {
                    is_array_val=1;
                }
                else {
                    is_array_val=0;
                }
                gsub(/[\[\]]/, "", $1);
                gsub(/^[ \t]+|[ \t]+$/, "", $2);
                is_array[section,$1]=is_array_val;

                if (tolower($2) ~ /^yes|true|no|false$/) {
                    value=tolower($2);
                }
                else if ($2 ~ /^[0-9]+$/) {
                    value=$2;
                }
                else if (is_array_val) {
                    value=$2;
                }
                else {
                    value="'\''"$2"'\''";
                }
                if (configuration[section,$1] == "") {
                    configuration[section,$1]=value;
                }
                else {
                  if (is_array_val) {
                      configuration[section,$1]=configuration[section,$1]"|"value;
                  }
                  else {
                      configuration[section,$1]=configuration[section,$1]" "value;
                  }
                }
              }
            }
            END {
              for (combined in configuration) {
                  split(combined, separate, SUBSEP);
                  section=separate[1];
                  key=separate[2];
                  section_comment="  # ["section"]"
                  if (force_set) {
                      prefix="";
                  }
                  else {
                      prefix="[[ ${_"key"_was_set:-0} -eq 0 ]] && ";
                  }
                  section_name = section;
                  gsub( "-", "_", section_name);
                  gsub( "\\.", "_", section_name);
                  if (is_array[section,key]) {
                    print "set_AA configuration_" section_name " " key " \""configuration[section,key]"\";"section_comment;
                  }
                  else {
                    print "set_AA configuration_" section_name " " key " " configuration[section,key]";"section_comment;
                  }
            }
         }' "${filename}")
    else
        eval_str=$($AWK "${awk_args[@]}" '{
              if (section == "") { section="main"; }
              if ($1 ~ /^\[/) {
                gsub(/\[/, "", $1);
                gsub(/\]/, "", $1);
                section=tolower($1);
                # section=tolower(gensub(/\[(.+)\]/,"\\1",1,$1));
              }
              else if ($1 !~ /^$/ && $1 !~ /^;|#/) {
                gsub(/^[ \t]+|[ \t]+$/, "", $1);
                if ($1 ~ /[.*\[\]]/) {
                    is_array_val=1;
                }
                else {
                    is_array_val=0;
                }
                gsub(/[\[\]]/, "", $1);
                gsub(/^[ \t]+|[ \t]+$/, "", $2);
                is_array[section,$1]=is_array_val;

                if (tolower($2) ~ /^yes|true|no|false$/) {
                    value=tolower($2);
                }
                else if ($2 ~ /^[0-9]+$/) {
                    value=$2;
                }
                else if (is_array_val) {
                    value=$2;
                }
                else {
                    value="'\''"$2"'\''";
                }
                if (configuration[section,$1] == "") {
                    configuration[section,$1]=value;
                }
                else {
                  if (is_array_val) {
                      configuration[section,$1]=configuration[section,$1]"|"value;
                  }
                  else {
                      configuration[section,$1]=configuration[section,$1]" "value;
                  }
                }
              }
            }
            END {
              for (combined in configuration) {
                  split(combined, separate, SUBSEP);
                  section=separate[1];
                  key=separate[2];
                  section_comment="  # ["section"]"
                  if (force_set) {
                      prefix="";
                  }
                  else {
                      prefix="[[ ${_"key"_was_set:-0} -eq 0 ]] && ";
                  }
                  section_name = section;
                  gsub( "-", "_", section_name);
                  gsub( "\\.", "_", section_name);
                  if (is_array[section,key]) {
                    print "configuration_" section_name "[\""key"\"]=\""configuration[section,key]"\";"section_comment;
                  }
                  else {
                    print "configuration_" section_name "[\""key"\"]="configuration[section,key]";"section_comment;
                  }
            }
         }' "${filename}")
    fi
    eval "$eval_str"
}
