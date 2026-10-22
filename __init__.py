"""Glue for ComfyUI-TangoFlux on runners without torchcodec/ffmpeg libs.

Recent torchaudio routes torchaudio.save through torchcodec, which is absent on
Graydient runners, so TangoFluxVAEDecodeAndPlay fails after a successful render.
Replace torchaudio.save with a stdlib-wave writer (16-bit PCM WAV) when torchcodec
cannot be imported.

Also trim the AUDIO output of TangoFluxVAEDecodeAndPlay to the requested duration (with a 0.5 s fade-out):
the node only trims the copy it saves itself and returns the full 30 s decode, so a
downstream SaveAudio would otherwise write the sound followed by silence.
No nodes are added.
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


FADE_SECONDS = 0.5


def _trim_decode_output():
    import nodes as comfy_nodes
    import torch

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("TangoFluxVAEDecodeAndPlay")
    if cls is None or getattr(cls, "_tango_trim_patched", False):
        return
    orig = cls.play

    def play(self, vae, latents, *args, **kwargs):
        out = orig(self, vae, latents, *args, **kwargs)
        audio = out["result"][0]
        end = int(latents["duration"] * audio["sample_rate"])
        wave_t = audio["waveform"][..., :end]
        # TangoFlux ends hard at the requested duration; fade the last 0.5 s so it
        # does not sound chopped.
        n = min(int(FADE_SECONDS * audio["sample_rate"]), wave_t.shape[-1])
        if n > 0:
            ramp = torch.linspace(1.0, 0.0, n, device=wave_t.device, dtype=wave_t.dtype)
            wave_t = wave_t.clone()
            wave_t[..., -n:] = wave_t[..., -n:] * ramp
        audio["waveform"] = wave_t
        return out

    cls.play = play
    cls._tango_trim_patched = True


try:
    _trim_decode_output()
except Exception as e:  # never break ComfyUI startup
    print(f"[TangoFluxGlue] trim patch not applied: {e}")
