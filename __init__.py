"""Glue for ComfyUI-TangoFlux on runners without torchcodec/ffmpeg libs.

Recent torchaudio routes torchaudio.save through torchcodec, which is absent on
Graydient runners, so TangoFluxVAEDecodeAndPlay fails after a successful render.
Replace torchaudio.save with a stdlib-wave writer (16-bit PCM WAV) when torchcodec
cannot be imported.

Adds one node, TangoFluxTrimFade: TangoFluxVAEDecodeAndPlay only trims the copy it saves
itself and returns the full 30 s decode, so a downstream SaveAudio would write the sound
followed by silence. TangoFlux also stops hard at the requested duration, so the node
trims to `duration` and fades out the last `fade_seconds`.
"""
import importlib.util


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


class TangoFluxTrimFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "duration": ("INT", {"default": 10, "min": 1, "max": 30, "step": 1}),
                "fade_seconds": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "TangoFlux"

    def run(self, audio, duration, fade_seconds):
        import torch

        sr = int(audio["sample_rate"])
        wave_t = audio["waveform"][..., : int(duration * sr)].clone()
        n = min(int(fade_seconds * sr), wave_t.shape[-1])
        if n > 0:
            ramp = torch.linspace(1.0, 0.0, n, device=wave_t.device, dtype=wave_t.dtype)
            wave_t[..., -n:] = wave_t[..., -n:] * ramp
        return ({"waveform": wave_t, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {"TangoFluxTrimFade": TangoFluxTrimFade}
NODE_DISPLAY_NAME_MAPPINGS = {"TangoFluxTrimFade": "TangoFlux Trim + Fade"}
