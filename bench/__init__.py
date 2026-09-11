"""STiTy benchmark framework.

One YAML config defines a run; the pipeline is assembled in-process from a
component registry. No WebSocket server, no ports.

The repo has no installable root package, so make the project root importable
here -- bench reads pure functions out of evaluation/ (harness.scoring,
ast.metrics_ast) and drives the handler defined under evaluation/ and Qwen3-ASR/.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

__all__ = ["PROJECT_ROOT"]
