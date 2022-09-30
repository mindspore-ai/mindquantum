# ==============================================================================
#
# Copyright 2022 <Huawei Technologies Co., Ltd>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

# lint_cmake: -whitespace/indent

find_package(Threads REQUIRED)

# Set names of each component of abseil-cpp
foreach(
  comp
  algorithm
  algorithm_container
  any
  any_invocable
  atomic_hook
  bad_any_cast
  bad_any_cast_impl
  bad_optional_access
  bad_variant_access
  base
  base_internal
  bind_front
  bits
  btree
  city
  civil_time
  cleanup
  cleanup_internal
  compare
  compressed_tuple
  config
  container_common
  container_memory
  cord
  cord_internal
  cordz_functions
  cordz_handle
  cordz_info
  cordz_sample_token
  cordz_statistics
  cordz_update_scope
  cordz_update_tracker
  core_headers
  counting_allocator
  debugging
  debugging_internal
  demangle_internal
  dynamic_annotations
  endian
  errno_saver
  examine_stack
  exponential_biased
  failure_signal_handler
  fast_type_id
  fixed_array
  flags
  flags_commandlineflag
  flags_commandlineflag_internal
  flags_config
  flags_internal
  flags_marshalling
  flags_parse
  flags_path_util
  flags_private_handle_accessor
  flags_program_name
  flags_reflection
  flags_usage
  flags_usage_internal
  flat_hash_map
  flat_hash_set
  function_ref
  graphcycles_internal
  hash
  hash_function_defaults
  hash_policy_traits
  hashtable_debug
  hashtable_debug_hooks
  hashtablez_sampler
  inlined_vector
  inlined_vector_internal
  int128
  kernel_timeout_internal
  layout
  leak_check
  log_severity
  low_level_hash
  malloc_internal
  memory
  meta
  node_hash_map
  node_hash_set
  node_slot_policy
  numeric
  numeric_representation
  optional
  periodic_sampler
  prefetch
  pretty_function
  random_bit_gen_ref
  random_distributions
  random_internal_distribution_caller
  random_internal_distribution_test_util
  random_internal_fast_uniform_bits
  random_internal_fastmath
  random_internal_generate_real
  random_internal_iostream_state_saver
  random_internal_mock_helpers
  random_internal_nonsecure_base
  random_internal_pcg_engine
  random_internal_platform
  random_internal_pool_urbg
  random_internal_randen
  random_internal_randen_engine
  random_internal_randen_hwaes
  random_internal_randen_hwaes_impl
  random_internal_randen_slow
  random_internal_salted_seed_seq
  random_internal_seed_material
  random_internal_traits
  random_internal_uniform_helper
  random_internal_wide_multiply
  random_random
  random_seed_gen_exception
  random_seed_sequences
  raw_hash_map
  raw_hash_set
  raw_logging_internal
  sample_recorder
  scoped_set_env
  span
  spinlock_wait
  stacktrace
  status
  statusor
  str_format
  str_format_internal
  strerror
  strings
  strings_internal
  symbolize
  synchronization
  throw_delegate
  time
  time_zone
  type_traits
  utility
  variant)
  set(${_pkg}_${comp}_NAMES absl_${comp})
endforeach()

set(${_pkg}_algorithm_DEPENDENCIES config)
set(${_pkg}_algorithm_container_DEPENDENCIES algorithm core_headers meta)
set(${_pkg}_any_DEPENDENCIES bad_any_cast config core_headers fast_type_id type_traits utility)
set(${_pkg}_any_invocable_DEPENDENCIES base_internal config core_headers type_traits utility)
set(${_pkg}_atomic_hook_DEPENDENCIES config core_headers)
set(${_pkg}_bad_any_cast_DEPENDENCIES bad_any_cast_impl config)
set(${_pkg}_bad_any_cast_impl_DEPENDENCIES config raw_logging_internal)
set(${_pkg}_bad_optional_access_DEPENDENCIES config raw_logging_internal)
set(${_pkg}_bad_variant_access_DEPENDENCIES config raw_logging_internal)
set(${_pkg}_base_DEPENDENCIES
    atomic_hook
    base_internal
    config
    core_headers
    dynamic_annotations
    log_severity
    raw_logging_internal
    spinlock_wait
    type_traits)
set(${_pkg}_base_EXTERNAL_DEPENDENCIES Threads::Threads)
set(${_pkg}_base_internal_DEPENDENCIES config type_traits)
set(${_pkg}_bind_front_DEPENDENCIES base_internal compressed_tuple)
set(${_pkg}_bits_DEPENDENCIES core_headers)
set(${_pkg}_btree_DEPENDENCIES
    container_common
    compare
    compressed_tuple
    container_memory
    cord
    core_headers
    layout
    memory
    raw_logging_internal
    strings
    throw_delegate
    type_traits
    utility)
set(${_pkg}_city_DEPENDENCIES config core_headers endian)
set(${_pkg}_cleanup_DEPENDENCIES cleanup_internal config core_headers)
set(${_pkg}_cleanup_internal_DEPENDENCIES base_internal core_headers utility)
set(${_pkg}_compare_DEPENDENCIES core_headers type_traits)
set(${_pkg}_compressed_tuple_DEPENDENCIES utility)
set(${_pkg}_container_common_DEPENDENCIES type_traits)
set(${_pkg}_container_memory_DEPENDENCIES config memory type_traits utility)
set(${_pkg}_cord_DEPENDENCIES
    base
    config
    cord_internal
    cordz_functions
    cordz_info
    cordz_update_scope
    cordz_update_tracker
    core_headers
    endian
    fixed_array
    function_ref
    inlined_vector
    optional
    raw_logging_internal
    span
    strings
    type_traits)
set(${_pkg}_cord_internal_DEPENDENCIES
    base_internal
    compressed_tuple
    config
    core_headers
    endian
    inlined_vector
    layout
    raw_logging_internal
    strings
    throw_delegate
    type_traits)
set(${_pkg}_cordz_functions_DEPENDENCIES config core_headers exponential_biased raw_logging_internal)
set(${_pkg}_cordz_handle_DEPENDENCIES base config raw_logging_internal synchronization)
set(${_pkg}_cordz_info_DEPENDENCIES
    base
    config
    cord_internal
    cordz_functions
    cordz_handle
    cordz_statistics
    cordz_update_tracker
    core_headers
    inlined_vector
    span
    raw_logging_internal
    stacktrace
    synchronization)
set(${_pkg}_cordz_sample_token_DEPENDENCIES config cordz_handle cordz_info)
set(${_pkg}_cordz_statistics_DEPENDENCIES config core_headers cordz_update_tracker synchronization)
set(${_pkg}_cordz_update_scope_DEPENDENCIES config cord_internal cordz_info cordz_update_tracker core_headers)
set(${_pkg}_cordz_update_tracker_DEPENDENCIES config)
set(${_pkg}_core_headers_DEPENDENCIES config)
set(${_pkg}_counting_allocator_DEPENDENCIES config)
set(${_pkg}_debugging_DEPENDENCIES stacktrace leak_check)
set(${_pkg}_debugging_internal_DEPENDENCIES core_headers config dynamic_annotations errno_saver raw_logging_internal)
set(${_pkg}_demangle_internal_DEPENDENCIES base core_headers)
set(${_pkg}_dynamic_annotations_DEPENDENCIES config)
set(${_pkg}_endian_DEPENDENCIES base config core_headers)
set(${_pkg}_errno_saver_DEPENDENCIES config)
set(${_pkg}_examine_stack_DEPENDENCIES stacktrace symbolize config core_headers raw_logging_internal)
set(${_pkg}_exponential_biased_DEPENDENCIES config core_headers)
set(${_pkg}_failure_signal_handler_DEPENDENCIES examine_stack stacktrace base config core_headers raw_logging_internal)
set(${_pkg}_fast_type_id_DEPENDENCIES config)
set(${_pkg}_fixed_array_DEPENDENCIES
    compressed_tuple
    algorithm
    config
    core_headers
    dynamic_annotations
    throw_delegate
    memory)
set(${_pkg}_flags_DEPENDENCIES
    config
    flags_commandlineflag
    flags_config
    flags_internal
    flags_reflection
    base
    core_headers
    strings)
set(${_pkg}_flags_commandlineflag_DEPENDENCIES config fast_type_id flags_commandlineflag_internal optional strings)
set(${_pkg}_flags_commandlineflag_internal_DEPENDENCIES config dynamic_annotations fast_type_id)
set(${_pkg}_flags_config_DEPENDENCIES config flags_path_util flags_program_name core_headers strings synchronization)
set(${_pkg}_flags_internal_DEPENDENCIES
    base
    config
    flags_commandlineflag
    flags_commandlineflag_internal
    flags_config
    flags_marshalling
    synchronization
    meta
    utility)
set(${_pkg}_flags_marshalling_DEPENDENCIES config core_headers log_severity optional strings str_format)
set(${_pkg}_flags_parse_DEPENDENCIES
    config
    core_headers
    flags_config
    flags
    flags_commandlineflag
    flags_commandlineflag_internal
    flags_internal
    flags_private_handle_accessor
    flags_program_name
    flags_reflection
    flags_usage
    strings
    synchronization)
set(${_pkg}_flags_path_util_DEPENDENCIES config strings)
set(${_pkg}_flags_private_handle_accessor_DEPENDENCIES config flags_commandlineflag flags_commandlineflag_internal
                                                       strings)
set(${_pkg}_flags_program_name_DEPENDENCIES config core_headers flags_path_util strings synchronization)
set(${_pkg}_flags_reflection_DEPENDENCIES
    config
    flags_commandlineflag
    flags_private_handle_accessor
    flags_config
    strings
    synchronization
    flat_hash_map)
set(${_pkg}_flags_usage_DEPENDENCIES config core_headers flags_usage_internal strings synchronization)
set(${_pkg}_flags_usage_internal_DEPENDENCIES
    config
    flags_config
    flags
    flags_commandlineflag
    flags_internal
    flags_path_util
    flags_private_handle_accessor
    flags_program_name
    flags_reflection
    flat_hash_map
    strings
    synchronization)
set(${_pkg}_flat_hash_map_DEPENDENCIES container_memory core_headers hash_function_defaults raw_hash_map
                                       algorithm_container memory)
set(${_pkg}_flat_hash_set_DEPENDENCIES container_memory hash_function_defaults raw_hash_set algorithm_container
                                       core_headers memory)
set(${_pkg}_function_ref_DEPENDENCIES base_internal core_headers meta)
set(${_pkg}_graphcycles_internal_DEPENDENCIES base base_internal config core_headers malloc_internal
                                              raw_logging_internal)
set(${_pkg}_hash_DEPENDENCIES
    city
    config
    core_headers
    endian
    fixed_array
    function_ref
    meta
    int128
    strings
    optional
    variant
    utility
    low_level_hash)
set(${_pkg}_hash_function_defaults_DEPENDENCIES config cord hash strings)
set(${_pkg}_hash_policy_traits_DEPENDENCIES meta)
set(${_pkg}_hashtable_debug_DEPENDENCIES hashtable_debug_hooks)
set(${_pkg}_hashtable_debug_hooks_DEPENDENCIES config)
set(${_pkg}_hashtablez_sampler_DEPENDENCIES base config exponential_biased sample_recorder synchronization)
set(${_pkg}_inlined_vector_DEPENDENCIES algorithm core_headers inlined_vector_internal throw_delegate memory)
set(${_pkg}_inlined_vector_internal_DEPENDENCIES compressed_tuple core_headers memory span type_traits)
set(${_pkg}_int128_DEPENDENCIES config core_headers bits)
set(${_pkg}_kernel_timeout_internal_DEPENDENCIES core_headers raw_logging_internal time)
set(${_pkg}_layout_DEPENDENCIES config core_headers meta strings span utility)
set(${_pkg}_leak_check_DEPENDENCIES config core_headers)
set(${_pkg}_log_severity_DEPENDENCIES core_headers)
set(${_pkg}_low_level_hash_DEPENDENCIES bits config endian int128)
set(${_pkg}_malloc_internal_DEPENDENCIES base base_internal config core_headers dynamic_annotations
                                         raw_logging_internal)
set(${_pkg}_malloc_internal_EXTERNAL_DEPENDENCIES Threads::Threads)
set(${_pkg}_memory_DEPENDENCIES core_headers meta)
set(${_pkg}_meta_DEPENDENCIES type_traits)
set(${_pkg}_node_hash_map_DEPENDENCIES
    container_memory
    core_headers
    hash_function_defaults
    node_slot_policy
    raw_hash_map
    algorithm_container
    memory)
set(${_pkg}_node_hash_set_DEPENDENCIES core_headers hash_function_defaults node_slot_policy raw_hash_set
                                       algorithm_container memory)
set(${_pkg}_node_slot_policy_DEPENDENCIES config)
set(${_pkg}_numeric_DEPENDENCIES int128)
set(${_pkg}_numeric_representation_DEPENDENCIES config)
set(${_pkg}_optional_DEPENDENCIES
    bad_optional_access
    base_internal
    config
    core_headers
    memory
    type_traits
    utility)
set(${_pkg}_periodic_sampler_DEPENDENCIES core_headers exponential_biased)
set(${_pkg}_prefetch_DEPENDENCIES config)
set(${_pkg}_random_bit_gen_ref_DEPENDENCIES core_headers random_internal_distribution_caller
                                            random_internal_fast_uniform_bits type_traits)
set(${_pkg}_random_distributions_DEPENDENCIES
    base_internal
    config
    core_headers
    random_internal_generate_real
    random_internal_distribution_caller
    random_internal_fast_uniform_bits
    random_internal_fastmath
    random_internal_iostream_state_saver
    random_internal_traits
    random_internal_uniform_helper
    random_internal_wide_multiply
    strings
    type_traits)
set(${_pkg}_random_internal_distribution_caller_DEPENDENCIES config utility fast_type_id)
set(${_pkg}_random_internal_distribution_test_util_DEPENDENCIES config core_headers raw_logging_internal strings
                                                                str_format span)
set(${_pkg}_random_internal_fast_uniform_bits_DEPENDENCIES config)
set(${_pkg}_random_internal_fastmath_DEPENDENCIES bits)
set(${_pkg}_random_internal_generate_real_DEPENDENCIES bits random_internal_fastmath random_internal_traits type_traits)
set(${_pkg}_random_internal_iostream_state_saver_DEPENDENCIES int128 type_traits)
set(${_pkg}_random_internal_mock_helpers_DEPENDENCIES fast_type_id optional)
set(${_pkg}_random_internal_nonsecure_base_DEPENDENCIES
    core_headers
    inlined_vector
    random_internal_pool_urbg
    random_internal_salted_seed_seq
    random_internal_seed_material
    span
    type_traits)
set(${_pkg}_random_internal_pcg_engine_DEPENDENCIES config int128 random_internal_fastmath
                                                    random_internal_iostream_state_saver type_traits)
set(${_pkg}_random_internal_platform_DEPENDENCIES config)
set(${_pkg}_random_internal_pool_urbg_DEPENDENCIES
    base
    config
    core_headers
    endian
    random_internal_randen
    random_internal_seed_material
    random_internal_traits
    random_seed_gen_exception
    raw_logging_internal
    span)
set(${_pkg}_random_internal_randen_DEPENDENCIES random_internal_platform random_internal_randen_hwaes
                                                random_internal_randen_slow)
set(${_pkg}_random_internal_randen_engine_DEPENDENCIES endian random_internal_iostream_state_saver
                                                       random_internal_randen raw_logging_internal type_traits)
set(${_pkg}_random_internal_randen_hwaes_DEPENDENCIES random_internal_platform random_internal_randen_hwaes_impl config)
set(${_pkg}_random_internal_randen_hwaes_impl_DEPENDENCIES random_internal_platform config)
set(${_pkg}_random_internal_randen_slow_DEPENDENCIES random_internal_platform config)
set(${_pkg}_random_internal_salted_seed_seq_DEPENDENCIES inlined_vector optional span random_internal_seed_material
                                                         type_traits)
set(${_pkg}_random_internal_seed_material_DEPENDENCIES core_headers optional random_internal_fast_uniform_bits
                                                       raw_logging_internal span strings)
set(${_pkg}_random_internal_traits_DEPENDENCIES config)
set(${_pkg}_random_internal_uniform_helper_DEPENDENCIES config random_internal_traits type_traits)
set(${_pkg}_random_internal_wide_multiply_DEPENDENCIES bits config int128)
set(${_pkg}_random_random_DEPENDENCIES random_distributions random_internal_nonsecure_base random_internal_pcg_engine
                                       random_internal_pool_urbg random_internal_randen_engine random_seed_sequences)
set(${_pkg}_random_seed_gen_exception_DEPENDENCIES config)
set(${_pkg}_random_seed_sequences_DEPENDENCIES
    config
    inlined_vector
    random_internal_pool_urbg
    random_internal_salted_seed_seq
    random_internal_seed_material
    random_seed_gen_exception
    span)
set(${_pkg}_raw_hash_map_DEPENDENCIES container_memory raw_hash_set throw_delegate)
set(${_pkg}_raw_hash_set_DEPENDENCIES
    bits
    compressed_tuple
    config
    container_common
    container_memory
    core_headers
    endian
    hash_policy_traits
    hashtable_debug_hooks
    memory
    meta
    optional
    prefetch
    utility
    hashtablez_sampler)
set(${_pkg}_raw_logging_internal_DEPENDENCIES atomic_hook config core_headers errno_saver log_severity)
set(${_pkg}_sample_recorder_DEPENDENCIES base synchronization)
set(${_pkg}_scoped_set_env_DEPENDENCIES config raw_logging_internal)
set(${_pkg}_span_DEPENDENCIES algorithm core_headers throw_delegate type_traits)
set(${_pkg}_spinlock_wait_DEPENDENCIES base_internal core_headers errno_saver)
set(${_pkg}_stacktrace_DEPENDENCIES debugging_internal config core_headers)
set(${_pkg}_status_DEPENDENCIES
    atomic_hook
    config
    cord
    core_headers
    function_ref
    inlined_vector
    optional
    raw_logging_internal
    stacktrace
    str_format
    strerror
    strings
    symbolize)
set(${_pkg}_statusor_DEPENDENCIES
    base
    status
    core_headers
    raw_logging_internal
    type_traits
    strings
    utility
    variant)
set(${_pkg}_str_format_DEPENDENCIES str_format_internal)
set(${_pkg}_str_format_internal_DEPENDENCIES
    bits
    strings
    config
    core_headers
    numeric_representation
    type_traits
    utility
    int128
    span)
set(${_pkg}_strerror_DEPENDENCIES config core_headers errno_saver)
set(${_pkg}_strings_DEPENDENCIES
    strings_internal
    base
    bits
    config
    core_headers
    endian
    int128
    memory
    raw_logging_internal
    throw_delegate
    type_traits)
set(${_pkg}_strings_internal_DEPENDENCIES config core_headers endian raw_logging_internal type_traits)
set(${_pkg}_symbolize_DEPENDENCIES
    debugging_internal
    demangle_internal
    base
    config
    core_headers
    dynamic_annotations
    malloc_internal
    raw_logging_internal
    strings)
set(${_pkg}_synchronization_DEPENDENCIES
    graphcycles_internal
    kernel_timeout_internal
    atomic_hook
    base
    base_internal
    config
    core_headers
    dynamic_annotations
    malloc_internal
    raw_logging_internal
    stacktrace
    symbolize
    time)
set(${_pkg}_synchronization_EXTERNAL_DEPENDENCIES Threads::Threads)
set(${_pkg}_throw_delegate_DEPENDENCIES config raw_logging_internal)
set(${_pkg}_time_DEPENDENCIES
    base
    civil_time
    core_headers
    int128
    raw_logging_internal
    strings
    time_zone)
set(${_pkg}_type_traits_DEPENDENCIES config)
set(${_pkg}_utility_DEPENDENCIES base_internal config type_traits)
set(${_pkg}_variant_DEPENDENCIES bad_variant_access base_internal config core_headers type_traits utility)
