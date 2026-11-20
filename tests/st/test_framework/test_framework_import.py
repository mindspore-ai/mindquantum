# Copyright 2026 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Lazy framework exports with and without MindSpore."""

import importlib.util

import pytest

from mindquantum import framework

pytestmark = pytest.mark.level0

_HAS_MINDSPORE = importlib.util.find_spec('mindspore') is not None


def test_unknown_attribute_raises_attribute_error():
    """
    Feature: Attribute error semantics of ``mindquantum.framework``.

    Description: Access an attribute that does not belong to the framework module.

    Expectation: The standard ``AttributeError`` is raised instead of the historical ``ImportError``.
    """
    missing_name = 'not_a_framework_symbol'
    with pytest.raises(AttributeError, match="has no attribute 'not_a_framework_symbol'"):
        getattr(framework, missing_name)

    assert not hasattr(framework, missing_name)


@pytest.mark.skipif(not _HAS_MINDSPORE, reason='MindSpore is not installed')
def test_known_symbol_is_exported_when_mindspore_is_available():
    """
    Feature: Framework symbol export.

    Description: Access a documented framework symbol while MindSpore is installed.

    Expectation: The symbol exists and is exported through ``__all__``.
    """
    assert set(framework.__all__) == set(framework.framework_modules)
    for name in framework.framework_modules:
        assert getattr(framework, name) is not None


@pytest.mark.skipif(_HAS_MINDSPORE, reason='MindSpore is installed, the disabled-module hint is not used')
def test_known_symbol_explains_missing_mindspore():
    """
    Feature: Diagnostic message of the disabled framework module.

    Description: Access a documented framework symbol while MindSpore is missing.

    Expectation: An ``ImportError`` explains that MindSpore is required, and ``__all__`` stays empty.
    """
    assert framework.__all__ == []
    for name in ('MQOps', 'MQN2Ops', 'QRamVecOps', 'MQLayer'):
        with pytest.raises(ImportError, match='MindSpore not installed'):
            getattr(framework, name)
