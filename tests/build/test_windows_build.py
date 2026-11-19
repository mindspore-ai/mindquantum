# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Exercise Windows release entry points without installing dependencies or compiling MindQuantum."""

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(name='windows_release', params=['bat', 'ps1'])
def windows_release_fixture(tmp_path, request):
    """Keep the real entry point and argument/command helpers; replace only environment bootstrap and the builder."""
    extension = request.param
    if extension == 'bat':
        if sys.platform != 'win32':
            pytest.skip('Batch entry point requires Windows cmd.exe')
        command = [os.environ['COMSPEC'], '/d', '/c', 'call', 'build.bat', '/NoDelocate']
    else:
        powershell = shutil.which('pwsh') or shutil.which('powershell')
        if not powershell:
            pytest.skip('PowerShell is not installed')
        command = [powershell, '-NoLogo', '-NoProfile', '-File', './build.ps1', '-NoConfig', '-NoDelocate']

    shutil.copyfile(ROOT / f'build.{extension}', tmp_path / f'build.{extension}')
    shutil.copytree(ROOT / 'scripts', tmp_path / 'scripts')
    (tmp_path / 'third_party/fmt').mkdir(parents=True)

    # The CI library-copy step needs an import library, but these CLI tests never link against it.
    python_home = tmp_path / 'python'
    venv.EnvBuilder().create(python_home)
    python = python_home / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    (python_home / 'libs').mkdir()
    (python_home / f'libs/python{sys.version_info.major}{sys.version_info.minor}.lib').touch()
    bootstrap = tmp_path / 'scripts/build'
    for name in ('python_virtualenv_activate', 'python_virtualenv_update', 'locate_cmake', 'locate_python3'):
        script = 'exit /b 0\n' if extension == 'bat' else '$global:LASTEXITCODE = 0\n'
        if name == 'locate_python3':
            assignment = f'set PYTHON={python.name}\n' if extension == 'bat' else f"$PYTHON = '{python.name}'\n"
            script = assignment + script
        (bootstrap / f'{name}.{extension}').write_text(script, encoding='utf-8')

    (tmp_path / 'build.py').write_text(
        'import json, os, sys\n'
        'from pathlib import Path\n'
        'Path("arguments.json").write_text(json.dumps(sys.argv[1:]))\n'
        'status = int(os.environ["TEST_BUILD_STATUS"])\n'
        'if status:\n'
        '    sys.exit(status)\n'
        'if os.environ["TEST_EMIT_WHEEL"] == "1":\n'
        '    Path("dist").mkdir(exist_ok=True)\n'
        '    Path("dist/fixture.whl").write_bytes(b"built wheel")\n',
        encoding='utf-8',
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in ('PATH', 'SYSTEMROOT', 'COMSPEC', 'TEMP', 'TMP', 'HOME', 'USERPROFILE')
    }
    environment.update(
        PATH=str(python.parent) + os.pathsep + os.environ['PATH'],
        DEVCLOUD_CI='1',
        n_jobs_default='2',
        TEST_BUILD_STATUS='0',
        TEST_EMIT_WHEEL='1',
    )
    return command, environment, extension


@pytest.mark.parametrize('gpu', [False, True])
def test_windows_ci_preserves_backend_selection(tmp_path, windows_release, gpu):
    """
    Feature: CI does not implicitly request a GPU build.
    Description: Execute the release entry point with and without explicit GPU selection in a CI environment.
    Expectation: The builder receives the selected backend, and a successful build publishes its wheel.
    """
    command, environment, extension = windows_release
    if gpu:
        command += ['/Gpu' if extension == 'bat' else '-Gpu']
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = json.loads((tmp_path / 'arguments.json').read_text(encoding='utf-8'))
    index = arguments.index('-C--global-option=ENABLE_CUDA')
    assert arguments[index - 1] == f'-C--global-option=--{"set" if gpu else "unset"}'
    assert '-C--global-option=MINDSPORE_CI' in arguments
    assert (tmp_path / 'output/fixture.whl').read_bytes() == b'built wheel'
    assert not (tmp_path / 'dist/fixture.whl').exists()
    assert 'Successfully created mindquantum package' in ' '.join(result.stdout.split())


@pytest.mark.parametrize('failure', ['build', 'publish'])
def test_windows_release_propagates_failure(tmp_path, windows_release, failure):
    """
    Feature: A failed release must not report success or publish stale artifacts.
    Description: Fail the wheel builder with a distinctive exit code, or produce no wheel to publish.
    Expectation: The entry point fails, preserves the builder's exit code, and leaves stale wheels untouched.
    """
    command, environment, _ = windows_release
    if failure == 'build':
        environment['TEST_BUILD_STATUS'] = '37'
        (tmp_path / 'dist').mkdir()
        (tmp_path / 'dist/stale.whl').write_bytes(b'stale wheel')
    else:
        environment['TEST_EMIT_WHEEL'] = '0'
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)
    assert (tmp_path / 'arguments.json').exists(), result.stdout + result.stderr
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'Successfully created mindquantum package' not in ' '.join(result.stdout.split())
    assert not list((tmp_path / 'output').glob('*.whl'))
    if failure == 'build':
        assert result.returncode == 37, result.stdout + result.stderr
        assert (tmp_path / 'dist/stale.whl').read_bytes() == b'stale wheel'
