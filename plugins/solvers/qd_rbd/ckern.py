"""C fallback kernel + ctypes binding.

`csrc/rbd.c` is a transliteration of the RNEA pass in `algorithms.py`.  It is
compiled on first use into ``$XDG_CACHE_HOME/qd_rbd`` (keyed by a hash of the
source and the flags), so nothing is built at import time and an edited source
never leaves a stale .so behind.

Set ``QD_RBD_C=0`` to force the Quadrants path, ``QD_RBD_CC`` to pick a
compiler, ``QD_RBD_CFLAGS`` to override the flags (e.g. drop ``-march=native``
for a portable build).

Why a second implementation exists at all is in REPORT.md §11/§12: the
Quadrants backend emits no vector code and no FP contraction, so the only way
to learn what the hardware can do with this kernel is to compile it elsewhere
and measure both.
"""

import ctypes
import hashlib
import os
import shlex
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "csrc", "rbd.c")

# On Windows there is usually no OpenMP-capable gcc driver: -fopenmp would fail
# the whole build, so the default flags drop it (the kernel then compiles
# single-threaded; the batch loop is still shared-nothing).  Override both the
# compiler and the flags with QD_RBD_CC / QD_RBD_CFLAGS, e.g. a portable zig cc:
#   QD_RBD_CC="python -m ziglang cc" (flags default, no OpenMP)
# or for a reproducible non-native build:
#   QD_RBD_CFLAGS="-O3 -shared -std=c11 -ffp-contract=fast"
_CFLAGS = [
    "-O3",
    "-march=native",
    "-fopenmp",
    "-fPIC",
    "-shared",
    "-std=c11",
    "-ffp-contract=fast",
]
if sys.platform == "win32":
    _CFLAGS = [
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        "-std=c11",
        "-ffp-contract=fast",
    ]

# Shared-library suffix for the cached artifact.  Windows loads whatever the
# file contains, but the extension keeps tooling (and humans) honest.
_SUF = ".dll" if sys.platform == "win32" else ".so"

# Below this batch size a single call is dominated by fixed overhead on both
# sides and the C path's advantage is just "no argument marshalling", which is
# not what we are measuring.  Measured crossover is << 48; see REPORT.md §12.
MIN_BATCH = 1

# How many instances the C kernel runs through one joint sweep together.  One
# instance is a 7-deep dependency chain, so a single stream leaves most of the
# core idle; K independent streams give the out-of-order engine something to
# interleave.  Measured optimum in REPORT.md §12.4.
LANES = int(os.environ.get("QD_RBD_LANES", "4"))

_struct = None
_lib = None
_err = None
_tried = False


class _Model(ctypes.Structure):
    _fields_ = [
        ("nj", ctypes.c_int),
        ("nv", ctypes.c_int),
        ("par", ctypes.POINTER(ctypes.c_int)),
        ("kind", ctypes.POINTER(ctypes.c_int)),
        ("iv", ctypes.POINTER(ctypes.c_int)),
        ("axis", ctypes.POINTER(ctypes.c_double)),
        ("Rpl", ctypes.POINTER(ctypes.c_double)),
        ("ppl", ctypes.POINTER(ctypes.c_double)),
        ("A0", ctypes.POINTER(ctypes.c_double)),
        ("B0", ctypes.POINTER(ctypes.c_double)),
        ("mass", ctypes.POINTER(ctypes.c_double)),
        ("grav", ctypes.c_double * 3),
    ]


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _build():
    global _lib, _err, _tried, _struct
    if _tried:
        return _lib
    _tried = True
    if os.environ.get("QD_RBD_C", "1") == "0":
        _err = "disabled by QD_RBD_C=0"
        return None
    if not os.path.exists(_SRC):
        _err = f"missing {_SRC}"
        return None
    # QD_RBD_CC may carry arguments, e.g. "python -m ziglang cc".
    cc = shlex.split(os.environ.get("QD_RBD_CC", "gcc"))
    flags = os.environ.get("QD_RBD_CFLAGS", "").split() or _CFLAGS
    src = open(_SRC).read()
    tag = _hash(src + " ".join(cc) + " ".join(flags) + sys.platform)
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    outdir = os.path.join(cache, "qd_rbd")
    os.makedirs(outdir, exist_ok=True)
    so = os.path.join(outdir, f"rbd_{tag}{_SUF}")
    if not os.path.exists(so):
        fd, tmp = tempfile.mkstemp(suffix=_SUF, dir=outdir)
        os.close(fd)
        cmd = [*cc, *flags, "-o", tmp, _SRC, "-lm"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            # compiler executable missing entirely (e.g. no gcc on Windows):
            # degrade to the Quadrants backend instead of crashing the caller
            _err = f"cannot run compiler {cmd[0]!r}: {e}"
            os.unlink(tmp)
            return None
        if p.returncode != 0:
            _err = (p.stderr or p.stdout).strip().splitlines()[-3:]
            os.unlink(tmp)
            return None
        os.replace(tmp, so)
    try:
        lib = ctypes.CDLL(so)
    except OSError as e:  # pragma: no cover
        _err = str(e)
        return None
    # Plain ``c_void_p``, not ``POINTER(c_double)``: ctypes charges ~1.6 us to
    # build a typed pointer from an ndarray (``arr.ctypes.data_as``) against
    # 0.8 us for the bare address, and at B=1 the whole call is 4 us of kernel
    # plus everything else is that conversion.  See REPORT.md §12.6.
    _V = ctypes.c_void_p
    lib.rbd_rnea_k.restype = None
    lib.rbd_rnea_k.argtypes = [_V, _V, _V, _V, _V, ctypes.c_int, ctypes.c_int]
    lib.rbd_rnea_bias_k.restype = None
    lib.rbd_rnea_bias_k.argtypes = [_V, _V, _V, _V, ctypes.c_int, ctypes.c_int]
    _struct = _Model
    _lib = lib
    # One thread knob for the whole library: if the caller pinned Quadrants
    # with QD_THREADS, the C side should not silently use a different pool.
    qdt = os.environ.get("QD_THREADS", "")
    if qdt.isdigit() and int(qdt) > 0:
        set_threads(int(qdt))
    return lib


def available():
    """True when the C kernel built (or was cached) successfully."""
    return _build() is not None


def error():
    """Why the C kernel is unavailable, once ``available()`` is False."""
    _build()
    return _err


# OpenMP runtime names per toolchain/platform (gcc, MinGW-w64, LLVM/libomp).
_OMP_LIBS = ("libgomp.so.1", "libgomp-1.dll", "libomp.so", "libomp.dll")


def _omp(sym):
    """Resolve ``sym`` from whichever OpenMP runtime this build links against."""
    for name in _OMP_LIBS:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        fn = getattr(lib, sym, None)
        if fn is not None:
            return fn
    return None


def set_threads(n):
    """Set the OpenMP pool size (0 = library default)."""
    fn = _omp("omp_set_num_threads")
    if fn is None:
        return False
    fn(ctypes.c_int(int(n)))
    return True


def get_threads():
    """Current OpenMP pool size (0 when no OpenMP runtime is loaded)."""
    _build()  # QD_THREADS is applied there, so report what is in force
    fn = _omp("omp_get_max_threads")
    if fn is None:
        return 0
    fn.restype = ctypes.c_int
    return fn()


def _ptr(a):
    """Raw address of a numpy buffer.  Cheapest form ctypes will take."""
    return a.__array_interface__["data"][0]


class CModel:
    """C-side view of a ``Model``: same numbers, C-friendly layout.

    Everything is copied once at construction.  The arrays are kept on the
    object so the pointers the struct holds stay valid.
    """

    def __init__(self, model):
        lib = _build()
        if lib is None:
            raise RuntimeError(f"qd_rbd C kernel unavailable: {error()}")
        nj = model.nj
        self.par = np.ascontiguousarray(model.parents, dtype=np.int32)
        self.kind = np.ascontiguousarray(model.jtype, dtype=np.int32)
        self.iv = np.ascontiguousarray(model.iv, dtype=np.int32)
        self.axis = np.ascontiguousarray(model.axis, dtype=np.float64)
        self.Rpl = np.ascontiguousarray(model.Rpl, dtype=np.float64)
        self.ppl = np.ascontiguousarray(model.ppl, dtype=np.float64)
        self.A0 = np.ascontiguousarray(model.I_A0, dtype=np.float64)
        self.B0 = np.ascontiguousarray(model.I_B0, dtype=np.float64)
        self.mass = np.ascontiguousarray(model.mass, dtype=np.float64)
        self.grav = np.ascontiguousarray(model.gravity, dtype=np.float64)
        self.nj, self.nv = nj, model.nv

        def i(p):
            return p.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        def d(p):
            return p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        self.st = _Model(
            nj,
            model.nv,
            i(self.par),
            i(self.kind),
            i(self.iv),
            d(self.axis),
            d(self.Rpl),
            d(self.ppl),
            d(self.A0),
            d(self.B0),
            d(self.mass),
            (ctypes.c_double * 3)(*self.grav),
        )
        self._stp = ctypes.addressof(self.st)
        self._lib = lib

    def rnea(self, q, v, a, tau, bias=False, K=None):
        B = q.shape[0]
        if K is None:
            # Interleaving only pays when a block of K instances is still
            # enough blocks to fill the pool; below that it costs parallelism.
            K = LANES if B >= 4 * LANES else 1
        if bias:
            self._lib.rbd_rnea_bias_k(self._stp, _ptr(q), _ptr(v), _ptr(tau), B, K)
        else:
            self._lib.rbd_rnea_k(self._stp, _ptr(q), _ptr(v), _ptr(a), _ptr(tau), B, K)
        return tau

    def __repr__(self):
        """Short summary with joint/dof counts."""
        return f"<qd_rbd.CModel nj={self.nj} nv={self.nv}>"
