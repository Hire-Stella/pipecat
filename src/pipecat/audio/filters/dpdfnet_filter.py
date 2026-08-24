#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""DPDFNet speech-enhancement audio filter for Pipecat.

Runs Ceva's DPDFNet streaming speech-enhancement models (Apache-2.0,
https://github.com/ceva-ip/DPDFNet) directly on ONNX Runtime. The models come
in native 8 kHz variants, so telephony audio is enhanced without the
resampling round-trip that 48 kHz-only denoisers require.

The upstream ``dpdfnet`` package offers the same models behind a
``StreamEnhancer`` class; this filter reimplements its causal STFT loop against
the ONNX session so the runtime needs only ``onnxruntime`` and ``numpy``, both
already required by Pipecat.

Enhancement is a per-stream stateful operation: the model threads an RNN state
vector through every frame, so one filter instance serves exactly one audio
stream.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.resamplers.base_audio_resampler import SoxrQuality
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

try:
    import onnxruntime as ort
except ModuleNotFoundError as e:
    ort = None
    logger.error(f"Exception: {e}")
    logger.error("In order to use the DPDFNet filter, you need to install onnxruntime.")


def _vorbis_window(window_len: int) -> np.ndarray:
    """Build the Vorbis (sine-power) analysis/synthesis window.

    The window satisfies the Princen-Bradley condition at 50% overlap
    (``w[n]^2 + w[n + hop]^2 == 1``), which is what lets each frame commit one
    hop of output through plain overlap-add.

    Args:
        window_len: Window length in samples.

    Returns:
        The window as a float32 array.
    """
    half = window_len / 2
    indices = np.arange(window_len)
    sine = np.sin(0.5 * np.pi * (indices + 0.5) / half)
    return np.sin(0.5 * np.pi * sine * sine).astype(np.float32)


class DPDFNetFilter(BaseAudioFilter):
    """Audio filter using DPDFNet for streaming speech enhancement.

    Suppresses the non-speech noise floor — traffic, fans, room tone, hum,
    handset clatter — before audio reaches VAD, turn detection, and STT, which
    reduces turns opened by noise rather than speech. Like every
    speaker-independent enhancement model it is trained to preserve speech, so
    background *talkers* largely survive; use it against the noise floor, not
    against a competing voice.

    Adds roughly 40 ms of latency to the input path (the model's window plus
    overlap-add buffering) and costs about 0.6 ms of CPU per 10 ms of audio per
    stream, so concurrency headroom is worth measuring before enabling it
    fleet-wide.

    The model file is not bundled. Download a checkpoint matching the
    transport's sample rate from https://huggingface.co/Ceva-IP/DPDFNet
    (``onnx/dpdfnet2_8khz.onnx`` for telephony, ``onnx/dpdfnet2.onnx`` for
    16 kHz) and point ``model_path`` or ``DPDFNET_MODEL_PATH`` at it. Audio is
    resampled to the model's native rate when the two differ.

    Example::

        transport = FastAPIWebsocketTransport(
            params=FastAPIWebsocketParams(
                audio_in_filter=DPDFNetFilter(model_path="/models/dpdfnet2_8khz.onnx"),
            ),
        )
    """

    def __init__(
        self,
        *,
        model_path: str | None = None,
        resampler_quality: SoxrQuality = "QQ",
    ) -> None:
        """Initialize the DPDFNet speech-enhancement filter.

        Args:
            model_path: Path to a DPDFNet streaming ONNX model. Defaults to the
                ``DPDFNET_MODEL_PATH`` environment variable.
            resampler_quality: Resampler quality used when the transport's
                sample rate differs from the model's native rate. Defaults to
                "QQ" (quick) for lowest latency.
        """
        self._model_path = model_path or os.getenv("DPDFNET_MODEL_PATH")
        self._resampler_quality: SoxrQuality = resampler_quality

        self._filtering = True
        self._ready = False
        self._sample_rate = 0

        self._session = None
        self._model_sample_rate = 0
        self._in_spec_name = ""
        self._in_state_name = ""
        self._out_spec_name = ""
        self._out_state_name = ""
        self._init_state: np.ndarray | None = None

        self._window: np.ndarray | None = None
        self._win_len = 0
        self._hop_size = 0

        self._state: np.ndarray | None = None
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._out_buffer: np.ndarray | None = None

        self._resampler_in = None
        self._resampler_out = None

    async def start(self, sample_rate: int):
        """Load the model and prepare streaming state.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate

        if ort is None:
            logger.error("DPDFNet filter disabled: onnxruntime is not installed")
            return
        if not self._model_path:
            logger.error(
                "DPDFNet filter disabled: no model path given "
                "(pass model_path or set DPDFNET_MODEL_PATH)"
            )
            return
        if not os.path.isfile(self._model_path):
            logger.error(f"DPDFNet filter disabled: model file not found: {self._model_path}")
            return

        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to initialize DPDFNet: {e}")
            return

        if self._sample_rate != self._model_sample_rate:
            logger.info(
                f"DPDFNet filter enabling resampling: "
                f"{self._sample_rate} <-> {self._model_sample_rate}"
            )
            try:
                from pipecat.audio.resamplers.soxr_stream_resampler import (
                    SOXRStreamAudioResampler,
                )

                self._resampler_in = SOXRStreamAudioResampler(quality=self._resampler_quality)
                self._resampler_out = SOXRStreamAudioResampler(quality=self._resampler_quality)
            except ImportError as e:
                logger.error(f"Could not import SOXRStreamAudioResampler for resampling: {e}")
                return

        self._reset_stream_state()
        self._ready = True
        logger.debug(
            f"DPDFNet filter ready (model_rate={self._model_sample_rate}, "
            f"window={self._win_len}, hop={self._hop_size})"
        )

    def _load_model(self):
        """Create the ONNX session and derive the streaming parameters from it."""
        options = ort.SessionOptions()
        # One thread per stream: many concurrent calls share the box, and the
        # per-frame tensors are far too small to gain from intra-op threading.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self._model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) < 2 or len(outputs) < 2:
            raise ValueError(
                "Expected a streaming DPDFNet model with two inputs (spec, state) "
                "and two outputs (spec, state)"
            )
        self._in_spec_name = inputs[0].name
        self._in_state_name = inputs[1].name
        self._out_spec_name = outputs[0].name
        self._out_state_name = outputs[1].name

        metadata = self._session.get_modelmeta().custom_metadata_map
        self._model_sample_rate = int(metadata["sample_rate"])
        self._init_state = self._build_initial_state(metadata)

        # The spectrum input carries one frame of ``freq_bins`` complex pairs,
        # from which the window length follows.
        freq_bins = inputs[0].shape[-2]
        self._win_len = int((int(freq_bins) - 1) * 2)
        self._hop_size = self._win_len // 2
        self._window = _vorbis_window(self._win_len)

    @staticmethod
    def _build_initial_state(metadata: dict) -> np.ndarray:
        """Build the model's initial RNN state from its embedded metadata.

        The normalization stages start from trained values rather than zeros;
        the exported model carries them as metadata.

        Args:
            metadata: The ONNX model's custom metadata map.

        Returns:
            The initial state vector as a float32 array.
        """
        state_size = int(metadata["state_size"])
        erb_size = int(metadata["erb_norm_state_size"])
        spec_size = int(metadata["spec_norm_state_size"])
        erb_init = np.array(metadata["erb_norm_init"].split(","), dtype=np.float32)
        spec_init = np.array(metadata["spec_norm_init"].split(","), dtype=np.float32)

        state = np.zeros(state_size, dtype=np.float32)
        state[:erb_size] = erb_init
        state[erb_size : erb_size + spec_size] = spec_init
        return np.ascontiguousarray(state)

    def _reset_stream_state(self):
        """Clear the RNN state and the analysis/synthesis buffers."""
        self._state = self._init_state.copy() if self._init_state is not None else None
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._out_buffer = np.zeros(self._win_len, dtype=np.float32)

    async def stop(self):
        """Release the model when the transport stops."""
        self._ready = False
        self._session = None
        self._state = None
        self._out_buffer = None
        self._in_buffer = np.zeros(0, dtype=np.float32)
        self._resampler_in = None
        self._resampler_out = None

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        """Enhance audio, suppressing the non-speech noise floor.

        Buffers incoming audio into the model's window length and returns the
        enhanced samples committed by overlap-add. Returns empty bytes while the
        first window is still filling.

        Args:
            audio: Raw 16-bit PCM audio as bytes.

        Returns:
            Enhanced 16-bit PCM audio as bytes, which may be empty.
        """
        if not self._ready or not self._filtering:
            return audio
        if len(audio) == 0:
            return b""

        in_audio = audio
        if self._resampler_in:
            in_audio = await self._resampler_in.resample(
                audio, self._sample_rate, self._model_sample_rate
            )
            if len(in_audio) == 0:
                return b""

        samples = np.frombuffer(in_audio, dtype=np.int16).astype(np.float32) / 32768.0
        enhanced = self._enhance(samples)
        if enhanced.size == 0:
            return b""

        out_audio = (np.clip(enhanced, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        if self._resampler_out:
            return await self._resampler_out.resample(
                out_audio, self._model_sample_rate, self._sample_rate
            )
        return out_audio

    def _enhance(self, samples: np.ndarray) -> np.ndarray:
        """Run the model over whole frames buffered from ``samples``.

        Args:
            samples: Float32 mono samples at the model's native sample rate.

        Returns:
            The enhanced samples committed by overlap-add, possibly empty.
        """
        self._in_buffer = np.concatenate([self._in_buffer, samples])
        committed: list[np.ndarray] = []

        while len(self._in_buffer) >= self._win_len:
            spectrum = np.fft.rfft(self._in_buffer[: self._win_len] * self._window, n=self._win_len)
            spec_in = np.stack(
                [spectrum.real.astype(np.float32), spectrum.imag.astype(np.float32)], axis=-1
            )[np.newaxis, np.newaxis]

            spec_out, self._state = self._session.run(
                [self._out_spec_name, self._out_state_name],
                {self._in_spec_name: spec_in, self._in_state_name: self._state},
            )

            real_imag = spec_out[0, 0]
            frame = np.fft.irfft(real_imag[:, 0] + 1j * real_imag[:, 1], n=self._win_len)
            self._out_buffer += (frame * self._window).astype(np.float32)

            # The window satisfies COLA at 50% overlap, so one hop is final.
            committed.append(self._out_buffer[: self._hop_size].copy())
            self._out_buffer[: self._win_len - self._hop_size] = self._out_buffer[self._hop_size :]
            self._out_buffer[self._win_len - self._hop_size :] = 0.0
            self._in_buffer = self._in_buffer[self._hop_size :]

        if not committed:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(committed)
