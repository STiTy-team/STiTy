import numpy as np

from core.components.transcription.zprobe_prefix import pad_silence


def test_pad_silence_appends_zeros_of_the_given_seconds():
    wav = np.ones(16000, dtype=np.float32)
    out = pad_silence(wav, 1.0, np)
    assert out.shape[0] == 32000 and out.dtype == np.float32 and float(out[16000:].sum()) == 0.0
    assert pad_silence(wav, 0.0, np) is wav
