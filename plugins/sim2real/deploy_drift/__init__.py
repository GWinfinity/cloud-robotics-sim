"""Deploy drift harness: measure closed-loop trajectory cost of precision transforms.

Turns the torch module attached to a learned-dynamics hook into deployable
artifacts and applies the precision transforms an edge/NPU delegate will apply
later:

* :func:`export_onnx` — f32 ONNX with a dynamic batch dimension,
* :func:`quantize_dynamic_int8` — dynamic INT8 weight quantization (activations
  stay f32; the ORT-CPU stand-in for what a QNN HTP compile does with a QDQ
  graph),
* :class:`OrtOperator` — a torch-callable ONNX Runtime session that drops into
  the simulation loop unchanged (the board-side deployment performs the same
  swap, except with ``providers=["QNNExecutionProvider", ...]``).

The numbers those transforms introduce are measured by :func:`drift`: run each
artifact closed-loop from the same initial states and compare trajectories
against the f32 torch reference. A deployment profile must record and budget
against the INT8 numbers.

Optional dependencies (``cloud-robotics-sim[deploy]``): ``onnxruntime``,
``onnx``, ``onnxconverter-common``. Every entry point imports them lazily and
raises a descriptive error when missing.
"""

from .core.export import convert_fp16, export_onnx, quantize_dynamic_int8
from .core.harness import OrtOperator, drift

__all__ = [
    "OrtOperator",
    "convert_fp16",
    "drift",
    "export_onnx",
    "quantize_dynamic_int8",
]
