# ComfyUI-TangoFluxGlue
Makes `LucipherDev/ComfyUI-TangoFlux` save audio on machines without torchcodec (patches `torchaudio.save` to write 16-bit WAV). Adds one node (TangoFluxTrimFade: trim to duration + fade-out).
