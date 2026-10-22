"""Glue for ComfyUI-TangoFlux on runners without torchcodec/ffmpeg libs.

Recent torchaudio routes torchaudio.save through torchcodec, which is absent on
Graydient runners, so TangoFluxVAEDecodeAndPlay fails after a successful render.
Replace torchaudio.save with a stdlib-wave writer (16-bit PCM WAV) when torchcodec
cannot be imported.

Also trim the AUDIO output of TangoFluxVAEDecodeAndPlay to the requested duration:
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


def _trim_decode_output():
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("TangoFluxVAEDecodeAndPlay")
    if cls is None or getattr(cls, "_tango_trim_patched", False):
        return
    orig = cls.play

    def play(self, vae, latents, *args, **kwargs):
        out = orig(self, vae, latents, *args, **kwargs)
        audio = out["result"][0]
        end = int(latents["duration"] * audio["sample_rate"])
        audio["waveform"] = audio["waveform"][..., :end]
        return out

    cls.play = play
    cls._tango_trim_patched = True


try:
    _trim_decode_output()
except Exception as e:  # never break ComfyUI startup
    print(f"[TangoFluxGlue] trim patch not applied: {e}")
