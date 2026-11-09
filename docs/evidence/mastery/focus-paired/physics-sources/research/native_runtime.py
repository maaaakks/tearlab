"""Build and launch portable CPU/CUDA primitives without a PyTorch extension.

Imports and CPU builds never initialize CUDA. PTX compilation also needs no GPU.
Runtime GPU initialization occurs only for an explicitly requested CUDA env.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import weakref

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE / 'native'
if not SRC.exists():
    SRC = HERE  # standalone deliverable layout
POINTER_NAMES = ('pos vel targets target_vel speed bpos bvel life cooldown '
                 'hostile_pos hostile_vel hostile_life attack_speed native_velocity eye '
                 'hp steps attack_timer immunity last_action fire_clock attacks move '
                 '_roles _counts _shooters _chaser_hits _old_pos _old_targets _alive '
                 '_active _totals _flags _mid _random _obs _mask _rbase _rtimer '
                 '_rspeed _rpos _rangle _rradius _rch_speed').split()
_CPU_LIBRARIES = {}
_CUDA_MODULES = {}


def compiler_environment():
    environment = os.environ.copy()
    if os.name == 'nt':
        compiler = shutil.which('cl.exe')
        if compiler:
            msvc = Path(compiler).parents[3]
            sdk = Path(os.environ.get('WindowsSdkDir', r'C:\Program Files (x86)\Windows Kits\10'))
            versions = sorted((sdk / 'Include').glob('10.*'))
            if versions:
                version = versions[-1].name
                includes = [msvc/'include', *[sdk/'Include'/version/p for p in ('ucrt', 'shared', 'um', 'winrt')]]
                libs = [msvc/'lib/x64', sdk/'Lib'/version/'ucrt/x64', sdk/'Lib'/version/'um/x64']
                environment['INCLUDE'] = os.pathsep.join(map(str, includes))
                environment['LIB'] = os.pathsep.join(map(str, libs))
    return environment


def source_digest():
    return hashlib.sha256(b''.join((SRC / f).read_bytes() for f in
                         ('isaac_kernels.h', 'isaac_kernels.cu', 'isaac_kernels_cpu.cpp'))).hexdigest()


def build_native(kind='cpu', cache=None, arch='89'):
    """Explicit offline compiler invocation; no GPU discovery or CUDA imports."""
    if kind not in ('cpu', 'cuda'):
        raise ValueError('kind must be cpu or cuda')
    if not arch.isdigit():
        raise ValueError('CUDA architecture must be digits, e.g. 89')
    cache = Path(cache or os.environ.get('ISAAC_NATIVE_CACHE', HERE / 'native_build'))
    cache.mkdir(parents=True, exist_ok=True)
    suffix = '.ptx' if kind == 'cuda' else ('.dll' if os.name == 'nt' else '.so')
    output = (cache / f'isaac_{kind}_{arch}_{source_digest()[:16]}{suffix}').resolve()
    if output.exists():
        return output
    if kind == 'cuda':
        nvcc = shutil.which('nvcc') or shutil.which('nvcc.exe')
        if not nvcc:
            raise RuntimeError('CUDA Toolkit nvcc is required to build PTX')
        command = [nvcc, '-ptx', '-std=c++17', f'-arch=compute_{arch}',
                   '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
                   str(SRC/'isaac_kernels.cu'), '-o', str(output)]
    elif os.name == 'nt':
        compiler = shutil.which('cl.exe')
        if not compiler:
            raise RuntimeError('Run in an MSVC development shell (cl.exe required)')
        command = [compiler, '/nologo', '/LD', '/O2', '/fp:strict', '/std:c++17',
                   str(SRC/'isaac_kernels_cpu.cpp'), f'/Fo{cache.resolve()}\\',
                   '/link', f'/OUT:{output}', f'/IMPLIB:{output.with_suffix(".lib")}']
    else:
        command = [shutil.which('c++') or 'c++', '-shared', '-fPIC', '-O2',
                   '-ffp-contract=off', '-std=c++17', str(SRC/'isaac_kernels_cpu.cpp'), '-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, env=compiler_environment())
    log = dict(kind=kind, command=command, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr, source_sha256=source_digest(),
               gpu_executed=False)
    output.with_suffix('.build.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'Native build failed: {result.stdout}\n{result.stderr}')
    return output


def build_resources(cache=None, arch='89'):
    """Offline ptxas register/shared-memory/spill report, not measured occupancy."""
    ptx = build_native('cuda', cache, arch)
    output = ptx.with_suffix('.cubin')
    command = [shutil.which('nvcc') or 'nvcc', '-cubin', '-std=c++17', f'-arch=sm_{arch}',
               '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
               '--resource-usage', '-Xptxas=-v', str(SRC/'isaac_kernels.cu'), '-o', str(output)]
    result = subprocess.run(command, capture_output=True, text=True, env=compiler_environment())
    report = output.with_suffix('.resources.json')
    report.write_text(json.dumps(dict(command=command, returncode=result.returncode,
                                      stdout=result.stdout, stderr=result.stderr,
                                      gpu_executed=False, source_sha256=source_digest()), indent=2), encoding='utf8')
    if result.returncode:
        raise RuntimeError(f'Offline resource build failed: {result.stdout}\n{result.stderr}')
    return report


def config_values(e):
    c, p, v = e.native, e.p, e.config
    return [p.room_width, p.room_height, c.idle_retention**.5,
            c.parallel_retention**.5, c.reverse_retention**.5, c.transverse_retention**.5,
            c.acceleration_half, c.wall_margin, c.wall_rebound, c.wall_bias,
            c.wall_correction_max, c.tear_inheritance, c.eye_offset, c.fire_ticks,
            c.wall_update_order, p.player_velocity_scale, p.tear_lifetime, p.hit_radius,
            p.contact_radius, v.projectile_hit_radius, v.projectile_lifetime,
            v.diagnostic_immunity_steps, p.fire_interval, p.max_steps, p.target_hits,
            p.target_speed_min, p.target_speed_max, float(e._mixed), e.enemies,
            v.projectile_speed_min, v.projectile_speed_max, v.warning_steps, c.command_delay_ticks]


class NativeRuntime:
    def __init__(self, env, cache=None, arch='89'):
        self.env = weakref.proxy(env)
        integers = {'hp','steps','attack_timer','immunity','last_action','fire_clock','attacks',
                    '_counts','_shooters','_chaser_hits','_random','_rtimer'}
        booleans = {'_roles','_alive','_active','_flags','_mask'}
        for name in POINTER_NAMES:
            value = getattr(env, name)
            expected = torch.int64 if name in integers else (torch.bool if name in booleans else torch.float32)
            if value.dtype != expected or value.device != env.pos.device or not value.is_contiguous():
                raise TypeError(f'Native ABI requires contiguous {expected} {name} on {env.pos.device}')
        self.config = torch.tensor(config_values(env), dtype=torch.float32, device=env.device)
        pointers = [getattr(env, k).data_ptr() for k in POINTER_NAMES]
        self.cpu = env.device.type == 'cpu'
        self.functions = {}
        self.binary = build_native('cpu' if self.cpu else 'cuda', cache, arch)
        if self.cpu:
            self.table = (ct.c_void_p * len(pointers))(*pointers)
            if str(self.binary) not in _CPU_LIBRARIES:
                _CPU_LIBRARIES[str(self.binary)] = ct.CDLL(str(self.binary))
            self.library = _CPU_LIBRARIES[str(self.binary)]
            for name, nargs in [('motion', 2), ('attacks', 1), ('projectiles', 3),
                                ('observation', 0), ('reset', 0), ('tick', 3), ('decision', 2)]:
                fn = getattr(self.library, 'cpu_' + name)
                extra = ([ct.c_void_p] + [ct.c_int] * (nargs - 1)) if name not in ('attacks', 'observation', 'reset') else [ct.c_int] * nargs
                fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, *extra]
                fn.restype = None
                self.functions[name] = fn
        else:
            self.table = torch.tensor(pointers, dtype=torch.int64, device=env.device)
            self.driver = ct.WinDLL('nvcuda.dll') if os.name == 'nt' else ct.CDLL('libcuda.so.1')
            d = self.driver
            d.cuInit.argtypes = [ct.c_uint]; d.cuInit.restype = ct.c_int
            d.cuModuleLoad.argtypes = [ct.POINTER(ct.c_void_p), ct.c_char_p];d.cuModuleLoad.restype = ct.c_int
            d.cuModuleGetFunction.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p];d.cuModuleGetFunction.restype = ct.c_int
            d.cuLaunchKernel.argtypes = [ct.c_void_p, *([ct.c_uint]*6), ct.c_uint,
                                        ct.c_void_p, ct.c_void_p, ct.c_void_p];d.cuLaunchKernel.restype = ct.c_int
            self._ok(d.cuInit(0))
            d.cuCtxGetCurrent.argtypes = [ct.POINTER(ct.c_void_p)];d.cuCtxGetCurrent.restype = ct.c_int
            context = ct.c_void_p()
            self._ok(d.cuCtxGetCurrent(ct.byref(context)))
            key = (str(self.binary), context.value)
            if key not in _CUDA_MODULES:
                module = ct.c_void_p()
                self._ok(d.cuModuleLoad(ct.byref(module), os.fsencode(self.binary)))
                _CUDA_MODULES[key] = module
            self.module = _CUDA_MODULES[key]
            for name in ('motion', 'attacks', 'projectiles', 'projectiles_warp', 'observation', 'reset', 'tick', 'decision', 'decision_warp'):
                fn = ct.c_void_p()
                self._ok(d.cuModuleGetFunction(ct.byref(fn), self.module, ('isaac_' + name).encode()))
                self.functions[name] = fn
            # Keep module loaded for captured graphs' lifetime. No implicit
            # destructor or CUDA context teardown while another job is active.

    @staticmethod
    def _ok(error):
        if error:
            raise RuntimeError(f'CUDA Driver API error {error}')

    def launch(self, name, action=None, masked=None, damage_ends=None):
        e = self.env
        extra = []
        if action is not None:
            extra.append(ct.c_void_p(action.data_ptr()))
        if masked is not None:
            extra.append(ct.c_int(int(masked)))
        if damage_ends is not None:
            extra.append(ct.c_int(int(damage_ends)))
        if self.cpu:
            name = name.replace('_warp', '')  # scalar CPU oracle for either mapping
            self.functions[name](self.table, ct.c_void_p(self.config.data_ptr()), e.n, *extra)
        else:
            args = [ct.c_void_p(self.table.data_ptr()), ct.c_void_p(self.config.data_ptr()), ct.c_int(e.n), *extra]
            ptrs = (ct.c_void_p * len(args))(*(ct.cast(ct.byref(a), ct.c_void_p) for a in args))
            threads = 128
            work = e.n * (236 if name == 'observation' else (32 if name.endswith('_warp') else 1))
            self._ok(self.driver.cuLaunchKernel(self.functions[name], (work+threads-1)//threads, 1, 1,
                                                threads, 1, 1, 0,
                                                ct.c_void_p(torch.cuda.current_stream(e.device).cuda_stream), ptrs, None))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['cpu', 'cuda', 'both'], default='cpu')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--arch', default='89')
    parser.add_argument('--resources', action='store_true')
    args = parser.parse_args()
    for kind in (['cpu', 'cuda'] if args.kind == 'both' else [args.kind]):
        print(build_native(kind, args.cache, args.arch), flush=True)
    if args.resources:
        print(build_resources(args.cache, args.arch), flush=True)
