from collections import deque

import numpy as np

from core.errors import ConfigError
from core.utils import logging
from core.utils.config import as_component

log = logging.getLogger(__name__)

SAMPLING_RATE = 16000
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_FILE = "smart-turn-v3.2-cpu.onnx"
SMART_TURN_SECONDS = 8


class Tuner:
    SETTINGS: dict = {}

    def __init__(self, settings: dict, *, threshold: float, min_silence_ms: int):
        self.settings = settings
        self.base_threshold = threshold
        self.base_min_silence_ms = min_silence_ms

    async def load(self) -> None:
        pass

    def start(self) -> None:
        pass

    def threshold(self) -> float:
        return self.base_threshold

    def observe(self, prob: float, *, triggered: bool) -> None:
        pass

    def on_pause(self, pause_ms: float) -> None:
        pass

    def on_speech_start(self) -> None:
        pass

    def min_silence_ms(self, segment_sec: float) -> float:
        return self.base_min_silence_ms

    def should_end(self, silence_ms: float, segment_sec: float, audio) -> float | None:
        needed = self.min_silence_ms(segment_sec)
        return needed if silence_ms >= needed else None


class Fixed(Tuner):

    def should_end(self, silence_ms: float, segment_sec: float, audio) -> float | None:
        needed_samples = SAMPLING_RATE * self.base_min_silence_ms / 1000
        if silence_ms * SAMPLING_RATE / 1000 < needed_samples:
            return None
        return self.base_min_silence_ms


class Ramp(Tuner):
    SETTINGS = {
        "start_ms": ("start_ms", float),
        "end_ms": ("end_ms", float),
        "ramp_sec": ("ramp_sec", float),
        "max_segment_sec": ("max_segment_sec", float),
        "hard_min_ms": ("hard_min_ms", float),
    }

    def min_silence_ms(self, segment_sec: float) -> float:
        start = self.settings.get("start_ms", self.base_min_silence_ms)
        end = self.settings.get("end_ms", 300.0)
        ramp = max(self.settings.get("ramp_sec", 10.0), 1e-3)
        needed = start - (start - end) * min(1.0, segment_sec / ramp)
        longest = self.settings.get("max_segment_sec")
        if longest is not None and segment_sec >= longest:
            needed = min(needed, self.settings.get("hard_min_ms", 100.0))
        return needed


class Adaptive(Tuner):
    SETTINGS = {
        "pauses": ("pauses", bool),
        "pause_quantile": ("pause_quantile", float),
        "pause_factor": ("pause_factor", float),
        "min_pauses": ("min_pauses", int),
        "pause_memory": ("pause_memory", int),
        "floor_ms": ("floor_ms", float),
        "ceiling_ms": ("ceiling_ms", float),
        "noise_floor": ("noise_floor", bool),
        "noise_margin": ("noise_margin", float),
        "noise_window_sec": ("noise_window_sec", float),
        "max_threshold": ("max_threshold", float),
    }
    WINDOW_SEC = 512 / SAMPLING_RATE

    def start(self) -> None:
        self.pauses: deque = deque(maxlen=self.settings.get("pause_memory", 40))
        windows = int(self.settings.get("noise_window_sec", 5.0) / self.WINDOW_SEC)
        self.quiet_probs: deque = deque(maxlen=max(8, windows))
        self.current_threshold = self.base_threshold

    def threshold(self) -> float:
        return self.current_threshold

    def observe(self, prob: float, *, triggered: bool) -> None:
        if not self.settings.get("noise_floor") or triggered:
            return
        self.quiet_probs.append(prob)
        if len(self.quiet_probs) < self.quiet_probs.maxlen // 2:
            return
        floor = float(np.percentile(self.quiet_probs, 90))
        margin = self.settings.get("noise_margin", 0.25)
        ceiling = self.settings.get("max_threshold", 0.8)
        self.current_threshold = min(ceiling, max(self.base_threshold, floor + margin))

    def on_pause(self, pause_ms: float) -> None:
        self.pauses.append(pause_ms)

    def min_silence_ms(self, segment_sec: float) -> float:
        if not self.settings.get("pauses", True):
            return self.base_min_silence_ms
        if len(self.pauses) < self.settings.get("min_pauses", 5):
            return self.base_min_silence_ms
        typical = float(np.quantile(self.pauses, self.settings.get("pause_quantile", 0.9)))
        needed = typical * self.settings.get("pause_factor", 1.2)
        return float(np.clip(needed, self.settings.get("floor_ms", 250.0),
                             self.settings.get("ceiling_ms", self.base_min_silence_ms)))


class SmartTurn(Tuner):
    SETTINGS = {
        "model": ("model", str),
        "probe_ms": ("probe_ms", float),
        "max_silence_ms": ("max_silence_ms", float),
        "end_probability": ("end_probability", float),
    }

    async def load(self) -> None:
        try:
            import onnxruntime
        except ImportError as e:
            raise ConfigError(
                "vad tuner 'smart-turn' needs onnxruntime (uv add --project bench onnxruntime)"
            ) from e
        from transformers import WhisperFeatureExtractor

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            self._model_path(), sess_options=options, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        log.info("[LOAD] smart-turn %s", self._model_path())

    def _model_path(self) -> str:
        path = self.settings.get("model")
        if path and not path.startswith("hf:"):
            return path
        from huggingface_hub import hf_hub_download

        repo, _, name = (path or f"hf:{SMART_TURN_REPO}/{SMART_TURN_FILE}")[3:].rpartition("/")
        return hf_hub_download(repo, name)

    def start(self) -> None:
        self.probed_this_pause = False

    def on_speech_start(self) -> None:
        self.probed_this_pause = False

    def on_pause(self, pause_ms: float) -> None:
        self.probed_this_pause = False

    def should_end(self, silence_ms: float, segment_sec: float, audio) -> float | None:
        longest = self.settings.get("max_silence_ms", 1500.0)
        if silence_ms >= longest:
            return silence_ms
        if self.probed_this_pause or silence_ms < self.settings.get("probe_ms", 250.0):
            return None
        self.probed_this_pause = True
        probability = self.end_probability(audio())
        log.debug("[TURN-PROBE] silence_ms=%.0f p_end=%.3f", silence_ms, probability)
        if probability >= self.settings.get("end_probability", 0.5):
            return silence_ms
        return None

    def end_probability(self, samples: np.ndarray) -> float:
        size = SMART_TURN_SECONDS * SAMPLING_RATE
        samples = samples[-size:].astype(np.float32)
        if samples.shape[0] < size:
            samples = np.concatenate([np.zeros(size - samples.shape[0], dtype=np.float32), samples])
        inputs = self.features(samples, sampling_rate=SAMPLING_RATE, return_tensors="np",
                               padding="max_length", max_length=size, truncation=True,
                               do_normalize=True)
        features = inputs.input_features.astype(np.float32)
        return float(self.session.run(None, {"input_features": features})[0].reshape(-1)[0])


TUNERS = {"fixed": Fixed, "ramp": Ramp, "adaptive": Adaptive, "smart-turn": SmartTurn}


def parse_tuner(raw) -> dict:
    try:
        name, options = as_component(raw)
    except ValueError as e:
        raise ValueError(f"tuner: {e}") from None
    if name not in TUNERS:
        raise ValueError(f"unknown tuner {name!r} (available: {sorted(TUNERS)})")
    accepted = TUNERS[name].SETTINGS
    unknown = sorted(set(options) - set(accepted))
    if unknown:
        raise ValueError(f"tuner {name!r} has no setting(s) {unknown} "
                         f"(it accepts: {sorted(accepted) or 'nothing'})")
    return {"name": name, **{stored: read(options[key]) for key, (stored, read)
                             in accepted.items() if options.get(key) is not None}}


def build_tuner(spec: dict | None, *, threshold: float, min_silence_ms: int) -> Tuner:
    spec = dict(spec or {"name": "fixed"})
    name = spec.pop("name")
    return TUNERS[name](spec, threshold=threshold, min_silence_ms=min_silence_ms)
