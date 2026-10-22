"""Glue for ComfyUI-TangoFlux on runners without torchcodec/ffmpeg libs.

Recent torchaudio routes torchaudio.save through torchcodec, which is absent on
Graydient runners, so TangoFluxVAEDecodeAndPlay fails after a successful render.
Replace torchaudio.save with a stdlib-wave writer (16-bit PCM WAV) when torchcodec
cannot be imported. No nodes are added.
"""
import importlib.util

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if importlib.util.find_spec("torchcodec") is None:
    import wave

    import numpy as np
    import torchaudio

    def _wav_save(path, waveform, sample_rate, *args, **kwargs):
        x = waveform.detach().cpu().float().clamp(-1, 1).numpy()
        pcm = (x.T * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(x.shape[0])
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(pcm.tobytes())

    torchaudio.save = _wav_save
