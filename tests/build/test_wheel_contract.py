# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Check wheel contents and failure semantics with real CMake and ELF builds."""

import ctypes
import hashlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from zipfile import ZipFile

import pytest
from setuptools import Distribution

ROOT = Path(__file__).resolve().parents[2]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(sys.platform == 'win32', reason='Bash release entry points support Linux and macOS')
@pytest.mark.parametrize('script', ['build.sh', 'build_locally.sh'])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize(
    'selector,target,variable',
    [('arch', 'dav-3510', 'CMAKE_ASC_ARCHITECTURES'), ('soc', 'Ascend910B2', 'MQ_ASCEND_SOC')],
)
def test_ascend_target_argument(tmp_path, script, enabled, selector, target, variable):
    """
    Feature: Explicit Ascend targets require NPU support and pass through without a product-model allowlist.
    Description: Dry-run both public build entry points with an architecture or SoC, with and without NPU support.
    Expectation: An enabled target reaches CMake; a target for a disabled backend fails immediately.
    """
    arguments = []
    if enabled:
        arguments.append('--npu')
    result = subprocess.run(
        [
            'bash',
            str(ROOT / script),
            '--no-config',
            '-n',
            f'--npu-{selector}={target}',
            f'--venv={sys.prefix}',
            '-B',
            str(tmp_path / 'build'),
            *arguments,
        ],
        cwd=ROOT,
        env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'LANG': 'C.UTF-8', 'VIRTUAL_ENV': sys.prefix},
        text=True,
        capture_output=True,
        check=False,
    )
    if not enabled:
        assert result.returncode != 0, result.stdout + result.stderr
        assert '--npu-arch and --npu-soc require --npu' in result.stdout + result.stderr
        return
    assert result.returncode == 0, result.stdout + result.stderr
    expected = (
        f'-C--global-option={variable} -C--global-option={target}'
        if script == 'build.sh'
        else f'-D{variable}:STRING={target}'
    )
    assert expected in result.stdout
    other_variable = 'MQ_ASCEND_SOC' if selector == 'arch' else 'CMAKE_ASC_ARCHITECTURES'
    assert other_variable not in result.stdout


@pytest.mark.skipif(sys.platform == 'win32', reason='Bash release entry points support Linux and macOS')
@pytest.mark.parametrize('script', ['build.sh', 'build_locally.sh'])
@pytest.mark.parametrize('selection', ['cpu', 'gpu', 'npu', 'no-npu'])
def test_ascend_config_is_backend_default(tmp_path, script, selection):
    """
    Feature: An Ascend target in build.conf is a default, not a request to enable Ascend.
    Description: Dry-run CPU, GPU, NPU and explicitly disabled NPU builds with the documented target configuration.
    Expectation: Only the enabled NPU build forwards its target to CMake.
    """
    config = tmp_path / 'build.conf'
    config.write_text(
        '[common]\nnpu_arch = dav-3510\n' + f'enable_npu = {"true" if selection == "no-npu" else "false"}\n',
        encoding='utf-8',
    )
    arguments = []
    if selection != 'cpu':
        arguments.append(f'--{selection}')
    result = subprocess.run(
        [
            'bash',
            str(ROOT / script),
            f'--config={config}',
            '-n',
            f'--venv={sys.prefix}',
            '-B',
            str(tmp_path / 'build'),
            *arguments,
        ],
        cwd=ROOT,
        env={'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'LANG': 'C.UTF-8', 'VIRTUAL_ENV': sys.prefix},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if selection == 'npu':
        expected = (
            '-C--global-option=CMAKE_ASC_ARCHITECTURES -C--global-option=dav-3510'
            if script == 'build.sh'
            else '-DCMAKE_ASC_ARCHITECTURES:STRING=dav-3510'
        )
        assert expected in result.stdout
    else:
        assert 'CMAKE_ASC_ARCHITECTURES' not in result.stdout
    assert 'MQ_ASCEND_SOC' not in result.stdout


@pytest.mark.parametrize('selector', ['CMAKE_ASC_ARCHITECTURES', 'MQ_ASCEND_SOC'])
@pytest.mark.parametrize('target', ['first;second', 'two targets', 'a"quote', 'a\\backslash'])
def test_ascend_target_is_one_identifier(tmp_path, selector, target):
    """
    Feature: Each Ascend build has one scalar compiler target, not a CMake list or quoted fragment.
    Description: Configure invalid target values without a CANN installation.
    Expectation: Target validation fails before SDK discovery, compiler definitions or JSON generation.
    """
    source = tmp_path / 'target.cmake'
    source.write_text(f'include("{(ROOT / "cmake/ascend.cmake").as_posix()}")\n', encoding='utf-8')
    result = subprocess.run(
        ['cmake', '-DENABLE_ASCEND=ON', f'-D{selector}={target}', '-P', str(source)],
        env={key: value for key, value in os.environ.items() if key != 'ASCEND_HOME_PATH'},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert f'{selector} must be a single compiler target identifier' in result.stderr


@pytest.mark.parametrize('enabled', [False, True])
def test_selected_extension_is_required(tmp_path, monkeypatch, enabled):
    """
    Feature: Selected CMake extension enforcement and wheel staging contract.
    Description: Build nested CMake targets into initially absent directories, with an optional failing extension.
    Expectation: The wheel contains the CPU module and installed data; a selected failing extension aborts the build.
    """
    setup = load_module(ROOT / 'setup.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(setup, 'cur_dir', tmp_path)
    monkeypatch.setattr(setup, 'cmake_extra_options', [f'-DENABLE_EXTRA={"ON" if enabled else "OFF"}'])
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    native = tmp_path / 'native'
    native.mkdir()
    (native / 'cpu.c').write_text('int answer(void) { return 17; }\n', encoding='utf-8')
    (native / 'CMakeLists.txt').write_text(
        'add_library(cpu MODULE cpu.c)\n'
        'set_target_properties(cpu PROPERTIES PREFIX "" '
        f'SUFFIX "{suffix}" LIBRARY_OUTPUT_DIRECTORY "${{CPU_OUTPUT_DIR}}")\n',
        encoding='utf-8',
    )
    (tmp_path / 'extra.c').write_text('#error selected extension must not disappear\n', encoding='utf-8')
    (tmp_path / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(WheelContract C)\n'
        'add_subdirectory(native)\n'
        'set_property(GLOBAL PROPERTY _python_targets cpu)\n'
        'if(ENABLE_EXTRA)\n'
        '  add_library(extra MODULE extra.c)\n'
        '  set_property(GLOBAL APPEND PROPERTY _python_targets extra)\n'
        'endif()\n'
        'install(FILES native/cpu.c DESTINATION share)\n'
        f'include("{(ROOT / "cmake/python_build_info.cmake").as_posix()}")\n',
        encoding='utf-8',
    )
    distribution = Distribution(
        {
            'name': 'mindquantum',
            'version': '1.0',
            'ext_modules': [
                setup.CMakeExtension('mindquantum.cpu'),
                setup.CMakeExtension('mindquantum.extra', optional=True),
            ],
            'cmdclass': {'build_ext': setup.CMakeBuildExt, 'bdist_wheel': setup.BdistWheel},
        }
    )
    distribution.script_name = 'setup.py'
    if enabled:
        distribution.get_command_obj('build_ext').install_light = True
        with pytest.raises(setup.BuildFailedError):
            distribution.run_command('bdist_wheel')
        assert not list((tmp_path / 'dist').glob('*.whl'))
    else:
        distribution.run_command('bdist_wheel')
        with ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as wheel:
            binary = tmp_path / f'cpu{suffix}'
            binary.write_bytes(wheel.read(f'mindquantum/{binary.name}'))
            assert not any('/extra.' in name for name in wheel.namelist())
            assert 'mindquantum/share/cpu.c' in wheel.namelist()
        assert ctypes.CDLL(str(binary)).answer() == 17


@pytest.mark.skipif(sys.platform != 'linux', reason='auditwheel repairs ELF wheels')
@pytest.mark.parametrize(
    'cann_environment,platform_override',
    [('active', None), ('active', 'manylinux_2_34'), ('unset', None), ('empty', None)],
)
def test_wheel_repair_keeps_vendor_runtime_external(tmp_path, monkeypatch, capfd, cann_environment, platform_override):
    """
    Feature: Automatic or explicit Linux wheel repair preserves external vendor runtimes and enforces ABI policy.
    Description: Repair CPU and NPU modules, then replace a CPU dependency with one requiring an unsupported glibc ABI.
    Expectation: CPU runs without the SDK; repair rejects missing SDKs and ABI violations instead of falling back.
    """
    monkeypatch.syspath_prepend(str(ROOT / '_build'))
    backend = load_module(ROOT / '_build/backend.py')
    sdk = tmp_path / 'sdk'
    sdk_lib = sdk / 'lib64'
    nested = sdk_lib / 'runtime'
    nested.mkdir(parents=True)
    sdk_alias = tmp_path / 'active-sdk'
    sdk_alias.symlink_to(sdk, target_is_directory=True)
    # The driver is outside the SDK. Excluding the SDK must also stop traversal into the driver.
    for name, directory, source, dependencies in (
        ('driver', tmp_path, 'int driver(void) { return 7; }', []),
        ('vendor', nested, 'extern int driver(void); int vendor(void) { return driver(); }', ['-ldriver']),
        ('helper', sdk_lib, 'int helper(void) { return 43; }', []),
        ('helper', tmp_path, 'int helper(void) { return 11; }', []),
        ('cpu', tmp_path, 'extern int helper(void); int answer(void) { return helper(); }', ['-lhelper']),
        ('npu', tmp_path, 'extern int vendor(void); int answer(void) { return vendor(); }', ['-lvendor']),
    ):
        path = directory / f'{name}.c'
        path.write_text(source, encoding='utf-8')
        subprocess.run(
            [
                'cc',
                '-shared',
                '-fPIC',
                # These fixtures use no libc. Do not inherit the host's CRT/glibc ABI floor.
                '-nostdlib',
                f'-Wl,-soname,lib{name}.so',
                str(path),
                f'-L{tmp_path}',
                f'-L{nested}',
                *dependencies,
                '-o',
                str(directory / f'lib{name}.so'),
            ],
            check=True,
        )
    metadata_script = tmp_path / 'metadata.cmake'
    metadata_script.write_text(
        f'set(PROJECT_BINARY_DIR "{tmp_path}")\n'
        'set_property(GLOBAL PROPERTY _python_targets cpu _mq_vector_npu)\n'
        'set(ENABLE_ASCEND ON)\nset(MQ_ASCEND_SOC Ascend910B2)\n'
        f'set(ASCEND_CANN_PACKAGE_PATH "{sdk_alias}")\n'
        f'include("{ROOT / "cmake/python_build_info.cmake"}")\n',
        encoding='utf-8',
    )
    subprocess.run(['cmake', '-P', str(metadata_script)], check=True)
    arch = platform.machine()
    python_tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    original = f'mindquantum-1.0-{python_tag}-{python_tag}-linux_{arch}.whl'
    fixture = tmp_path / original
    dist_info = 'mindquantum-1.0.dist-info'
    with ZipFile(fixture, 'w') as wheel:
        wheel.write(tmp_path / 'libcpu.so', 'mindquantum/cpu.so')
        wheel.write(tmp_path / 'libnpu.so', 'mindquantum/_mq_vector_npu.so')
        wheel.write(tmp_path / 'python-build-info.json', 'mindquantum/_build_info.json')
        wheel.writestr(f'{dist_info}/METADATA', 'Metadata-Version: 2.1\nName: mindquantum\nVersion: 1.0\n')
        wheel.writestr(
            f'{dist_info}/WHEEL',
            f'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: {python_tag}-{python_tag}-linux_{arch}\n',
        )
        record = f'{dist_info}/RECORD'
        wheel.writestr(record, ''.join(f'{name},,\n' for name in [*wheel.namelist(), record]))

    def build_fixture(wheel_directory, **_kwargs):
        shutil.copyfile(fixture, Path(wheel_directory, original))
        return original

    monkeypatch.setattr(backend.setuptools.build_meta, 'build_wheel', build_fixture)
    monkeypatch.setenv('ASCEND_HOME_PATH', str(sdk_alias))
    monkeypatch.setenv('LD_LIBRARY_PATH', f'{tmp_path}:{nested}')
    monkeypatch.setenv('MQ_DELOCATE_WHEEL', '1')
    target_platform = f'{platform_override}_{arch}' if platform_override else None
    if target_platform:
        monkeypatch.setenv('MQ_DELOCATE_WHEEL_PLAT', target_platform)
    else:
        monkeypatch.delenv('MQ_DELOCATE_WHEEL_PLAT', raising=False)
    output = tmp_path / 'wheelhouse'
    if cann_environment == 'unset':
        monkeypatch.delenv('ASCEND_HOME_PATH')
    elif cann_environment == 'empty':
        monkeypatch.setenv('ASCEND_HOME_PATH', '')
    if cann_environment != 'active':
        with pytest.raises(ValueError, match=r'nonempty ASCEND_HOME_PATH.*set_env\.sh'):
            backend.build_wheel(output)
        assert not list(output.glob('*.whl'))
        return

    filename = backend.build_wheel(output)
    repaired = output / filename
    assert repaired.exists()
    with ZipFile(repaired) as wheel:
        names = wheel.namelist()
        assert any('libhelper' in name for name in names)
        assert not any('libvendor' in name or 'libdriver' in name for name in names)
        tags = [line for line in wheel.read(f'{dist_info}/WHEEL').decode().splitlines() if line.startswith('Tag:')]
        if target_platform:
            assert filename == original.replace(f'linux_{arch}', target_platform)
            assert tags == [f'Tag: {python_tag}-{python_tag}-{target_platform}']
        else:
            assert tags
            assert all(tag.startswith(f'Tag: {python_tag}-{python_tag}-manylinux') for tag in tags)
            assert all(tag.endswith(f'_{arch}') for tag in tags)
            assert set(filename.removesuffix('.whl').rsplit('-', 1)[1].split('.')) == {
                tag.rsplit('-', 1)[1] for tag in tags
            }
        wheel.extractall(tmp_path / 'installed')
    # A separate process cannot reuse loaded libraries or LD_LIBRARY_PATH from the build.
    subprocess.run(
        [
            sys.executable,
            '-c',
            'import ctypes, sys; assert ctypes.CDLL(sys.argv[1]).answer() == 11',
            str(tmp_path / 'installed/mindquantum/cpu.so'),
        ],
        env={key: value for key, value in os.environ.items() if key not in ('LD_LIBRARY_PATH', 'ASCEND_HOME_PATH')},
        check=True,
    )
    assert (output / f'{filename}.sha256').read_text(encoding='utf-8').split() == [
        hashlib.sha256(repaired.read_bytes()).hexdigest(),
        filename,
    ]
    # Give the bundled CPU dependency a glibc requirement beyond every supported policy,
    # without depending on the build host's glibc version. This wheel must never be loaded.
    future_libc = tmp_path / 'future-libc.so'
    version_script = tmp_path / 'glibc.map'
    version_script.write_text('GLIBC_999.0 { global: newer_glibc; };\n', encoding='utf-8')
    source = tmp_path / 'future-glibc.c'
    source.write_text('int newer_glibc(void) { return 11; }\n', encoding='utf-8')
    subprocess.run(
        [
            'cc',
            '-shared',
            '-fPIC',
            '-nostdlib',
            '-Wl,-soname,libc.so.6',
            f'-Wl,--version-script={version_script}',
            str(source),
            '-o',
            str(future_libc),
        ],
        check=True,
    )
    source.write_text('extern int newer_glibc(void); int helper(void) { return newer_glibc(); }\n', encoding='utf-8')
    subprocess.run(
        [
            'cc',
            '-shared',
            '-fPIC',
            '-nostdlib',
            '-Wl,-soname,libhelper.so',
            str(source),
            str(future_libc),
            '-o',
            str(tmp_path / 'libhelper.so'),
        ],
        check=True,
    )
    capfd.readouterr()
    incompatible_output = tmp_path / 'incompatible-wheelhouse'
    with pytest.raises(subprocess.CalledProcessError):
        backend.build_wheel(incompatible_output)
    assert 'too-recent versioned symbols' in capfd.readouterr().err
    assert not list(incompatible_output.glob('*.whl'))

    monkeypatch.setenv('MQ_DELOCATE_WHEEL_PLAT', 'not_a_platform')
    with pytest.raises(subprocess.CalledProcessError):
        backend.build_wheel(tmp_path / 'bad-wheelhouse')


def test_pep517_metadata_accepts_cmake_options(tmp_path):
    """
    Feature: PEP 517 metadata generation with the same CMake options used for wheel construction.
    Description: Run dependency and metadata hooks with global CMake options and environment options.
    Expectation: Metadata succeeds without configuring CMake or deleting existing release outputs.
    """
    for relative in ('setup.py', '_build/backend.py', '_build/utils.py', 'mindquantum/utils/fdopen.py'):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    (tmp_path / 'pyproject.toml').write_text(
        '[project]\nname = "mindquantum"\nversion = "1.0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["mindquantum*"]\n',
        encoding='utf-8',
    )
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'keep').write_text('existing release', encoding='utf-8')
    script = (
        'import sys; sys.path.insert(0, "_build"); import backend\n'
        'config = {"--global-option": ["--set", "ENABLE_ASCEND", "--var", "MQ_ASCEND_SOC", "Ascend910B2"]}\n'
        'backend.get_requires_for_build_wheel(config)\n'
        'backend.prepare_metadata_for_build_wheel("metadata", config)\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', script],
        cwd=tmp_path,
        env={**os.environ, 'MQ_CIBW_BUILD_ARGS': '--unset,USE_NATIVE_INTRINSICS'},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Name: mindquantum\n' in (metadata / 'mindquantum-1.0.dist-info/METADATA').read_text(encoding='utf-8')
    assert (output / 'keep').read_text(encoding='utf-8') == 'existing release'
    assert not (tmp_path / 'build').exists()


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_cuda_requires_compiler(tmp_path, enabled):
    """
    Feature: Explicit CUDA selection must not silently produce a CPU-only configuration.
    Description: Configure with no CUDA compiler, with CUDA disabled and explicitly enabled.
    Expectation: CPU configuration succeeds; the requested CUDA configuration fails immediately.
    """
    (tmp_path / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(CudaContract NONE)\n'
        f'list(APPEND CMAKE_MODULE_PATH "{(ROOT / "cmake").as_posix()}" "{(ROOT / "cmake/commands").as_posix()}")\n'
        'include(macros)\ninclude(options)\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [
            'cmake',
            '-S',
            str(tmp_path),
            '-B',
            str(tmp_path / 'build'),
            f'-DENABLE_CUDA={"ON" if enabled else "OFF"}',
            '-DCMAKE_CUDA_COMPILER=NOTFOUND',
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if enabled:
        assert result.returncode != 0, result.stdout + result.stderr
        assert 'ENABLE_CUDA requires a CUDA compiler' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('enabled', [False, True])
def test_explicit_cuda_requires_toolkit(tmp_path, enabled):
    """
    Feature: Explicit CUDA selection requires a discoverable toolkit as well as a compiler.
    Description: Configure the dependency stage with toolkit discovery disabled, with CUDA off and on.
    Expectation: CPU configuration succeeds; a requested CUDA configuration fails instead of disabling CUDA.
    """
    (tmp_path / 'CMakeLists.txt').write_text(
        'cmake_minimum_required(VERSION 3.20)\nproject(CudaToolkitContract CXX)\n'
        f'list(APPEND CMAKE_MODULE_PATH "{(ROOT / "cmake").as_posix()}" "{(ROOT / "cmake/commands").as_posix()}")\n'
        'include(packages)\n',
        encoding='utf-8',
    )
    result = subprocess.run(
        [
            'cmake',
            '-S',
            str(tmp_path),
            '-B',
            str(tmp_path / 'build'),
            f'-DENABLE_CUDA:BOOL={"ON" if enabled else "OFF"}',
            '-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=TRUE',
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if enabled:
        assert result.returncode != 0, result.stdout + result.stderr
        assert 'CUDAToolkit' in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
