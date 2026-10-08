#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox speech-to-text service implementation."""

import json
import time
from collections import Counter
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import SONIOX_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.deprecation import deprecated
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

KEEPALIVE_MESSAGE = '{"type": "keepalive"}'

FINALIZE_MESSAGE = '{"type": "finalize"}'

END_TOKEN = "<end>"

FINALIZED_TOKEN = "<fin>"

# Single-speaker mode: speech from a non-locked label is kept when its level is
# at least this fraction of the caller's running level. Chosen on crowded-room
# recordings where 0.5 never dropped the caller's words while removing the
# clearly quieter bystanders.
SINGLE_SPEAKER_LEVEL_RATIO = 0.5
# How much outgoing audio to keep energy readings for; Soniox finalizes tokens
# within a few seconds of the speech they cover.
ENERGY_HISTORY_MS = 30_000
# Finalized speech each of the first two voices must have produced before the
# lock is decided between them, so a single loud word cannot outvote a caller.
MIN_LOCK_SPEECH_MS = 1_500


class SonioxContextGeneralItem(BaseModel):
    """Represents a key-value pair for structured general context information."""

    key: str
    value: str


class SonioxContextTranslationTerm(BaseModel):
    """Represents a custom translation mapping for ambiguous or domain-specific terms."""

    source: str
    target: str


class SonioxContextObject(BaseModel):
    """Structured context that steers transcription and translation.

    Every section is optional; supply only the ones that are relevant. Soniox
    caps the whole object at 8k tokens.

    Learn more about context in the documentation:
    https://soniox.com/docs/stt/concepts/context
    """

    general: list[SonioxContextGeneralItem] | None = None
    text: str | None = None
    terms: list[str] | None = None
    translation_terms: list[SonioxContextTranslationTerm] | None = None


@deprecated(
    "`SonioxInputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. Use "
    "`SonioxSTTService.Settings` instead."
)
class SonioxInputParams(BaseModel):
    """Real-time transcription settings.

    .. deprecated:: 0.0.105
        Use ``settings=SonioxSTTService.Settings(...)`` instead.
        Will be removed in 2.0.0.

    See Soniox WebSocket API documentation for more details:
    https://soniox.com/docs/speech-to-text/api-reference/websocket-api#configuration-parameters

    Parameters:
        model: Model to use for transcription.
        audio_format: Audio format to use for transcription.
        num_channels: Number of channels to use for transcription.
        language_hints: List of language hints to use for transcription.
        language_hints_strict: If true, strictly enforce language hints (only transcribe in provided languages).
        context: Customization for transcription. String for models with context_version 1 and ContextObject for models with context_version 2.
        enable_speaker_diarization: Whether to enable speaker diarization. Tokens are annotated with speaker IDs.
        enable_language_identification: Whether to enable language identification. Tokens are annotated with language IDs.
        client_reference_id: Client reference ID to use for transcription.
    """

    model: str = "stt-rt-v5"

    audio_format: str | None = "pcm_s16le"
    num_channels: int | None = 1

    language_hints: list[Language] | None = None
    language_hints_strict: bool | None = None
    context: SonioxContextObject | str | None = None

    enable_speaker_diarization: bool | None = False
    enable_language_identification: bool | None = False

    client_reference_id: str | None = None


def is_end_token(token: dict) -> bool:
    """Determine if a token is an end token."""
    return token["text"] == END_TOKEN or token["text"] == FINALIZED_TOKEN


def language_to_soniox_language(language: Language) -> str:
    """Convert a Pipecat Language to a Soniox language code.

    For a list of all supported languages, see:
    https://soniox.com/docs/speech-to-text/core-concepts/supported-languages
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KK: "kk",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def _tokens_to_text(tokens: list[dict[str, Any]], label_speakers: bool = True) -> str:
    """Join token texts, prefixing ``@<speaker>: `` wherever the speaker changes.

    Tokens carry a ``speaker`` label only when speaker diarization is enabled,
    so transcripts are unchanged otherwise. ``label_speakers=False`` joins the
    text without labels, for single-speaker mode where only one speaker remains.
    """
    parts = []
    speaker = None
    for token in tokens:
        text = token["text"]
        token_speaker = token.get("speaker") if label_speakers else None
        if token_speaker is not None and token_speaker != speaker:
            speaker = token_speaker
            parts.append(f"{' ' if parts else ''}@{speaker}: ")
            text = text.lstrip()
        parts.append(text)
    return "".join(parts)


def _prepare_language_hints(
    language_hints: list[Language] | None,
) -> list[str] | None:
    if language_hints is None:
        return None

    prepared_languages = [language_to_soniox_language(lang) for lang in language_hints]
    # Remove duplicates (in case of language_hints with multiple regions).
    return list(set(prepared_languages))


def _language_from_tokens(tokens: list[dict]) -> Language | None:
    language_counts: Counter[Language] = Counter()

    for token in tokens:
        language = token.get("language")
        if not language:
            continue
        try:
            language_counts[Language(language)] += 1
        except ValueError:
            pass

    if not language_counts:
        return None

    return language_counts.most_common(1)[0][0]


@dataclass
class SonioxSTTSettings(STTSettings):
    """Settings for SonioxSTTService.

    Parameters:
        language_hints: List of language hints to use for transcription.
        language_hints_strict: If true, strictly enforce language hints.
        context: Customization for transcription. Either a
            :class:`SonioxContextObject` or a plain string of background text.
        enable_speaker_diarization: Whether to enable speaker diarization.
        single_speaker: With diarization enabled, lock onto the louder of the first
            two voices heard and from then on drop speech from other labels that is
            clearly quieter than the caller, so distant background voices neither
            reach the transcript nor open a user turn. Speech as loud as the caller
            is kept whatever its label, since real-time labels flip; two people at
            the same distance from the microphone therefore cannot be separated.
            Loudness is the signal energy of the outgoing audio over each span.
            Client-side only; nothing is sent to Soniox.
        enable_language_identification: Whether to enable language identification.
        max_endpoint_delay_ms: Max ms before endpoint detection finalizes the turn (500-3000).
        endpoint_sensitivity: Endpoint detection sensitivity (-1.0 to 1.0); higher finalizes sooner.
        endpoint_latency_adjustment_level: Reduces endpoint latency vs. the default (0-3); higher
            finalizes sooner but may reduce accuracy.
        client_reference_id: Client reference ID to use for transcription.

    The ``max_endpoint_delay_ms``, ``endpoint_sensitivity`` and
    ``endpoint_latency_adjustment_level`` settings only take effect when
    ``vad_force_turn_endpoint=False``; otherwise Soniox endpoint detection is
    disabled and these settings are ignored.
    """

    language_hints: list[Language] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_hints_strict: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    context: SonioxContextObject | str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_speaker_diarization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    single_speaker: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_language_identification: bool | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    max_endpoint_delay_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpoint_sensitivity: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpoint_latency_adjustment_level: int | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    client_reference_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SonioxSTTService(WebsocketSTTService):
    """Speech-to-Text service using Soniox's WebSocket API.

    This service connects to Soniox's WebSocket API for real-time transcription
    with support for multiple languages, custom context, speaker diarization,
    and more.

    For complete API documentation, see: https://soniox.com/docs/speech-to-text/api-reference/websocket-api
    """

    Settings = SonioxSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://stt-rt.soniox.com/transcribe-websocket",
        sample_rate: int | None = None,
        model: str | None = None,
        audio_format: str = "pcm_s16le",
        num_channels: int = 1,
        params: SonioxInputParams | None = None,
        vad_force_turn_endpoint: bool = True,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = SONIOX_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Soniox STT service.

        Args:
            api_key: Soniox API key.
            url: Soniox WebSocket API URL.
            sample_rate: Audio sample rate.
            model: Soniox model to use for transcription.

                .. deprecated:: 0.0.105
                    Use ``settings=SonioxSTTService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            audio_format: Audio format for transcription. Defaults to ``"pcm_s16le"``.
            num_channels: Number of audio channels. Defaults to 1.
            params: Additional configuration parameters, such as language hints, context and
                speaker diarization.

                .. deprecated:: 0.0.105
                    Use ``settings=SonioxSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            vad_force_turn_endpoint: Controls turn detection mode.
                When True (Pipecat mode, default): Soniox endpoint detection is
                disabled and a `VADUserStoppedSpeakingFrame` sends a finalize
                message to Soniox. When False (Soniox turn detection mode): Soniox
                endpoint detection is enabled and controls turn endings. Proposes a
                turn start on the local VAD signal when a VAD analyzer is
                configured (most responsive) or on the first transcript token
                otherwise, and a turn stop when the endpoint is detected.
            should_interrupt: Whether to interrupt the bot when the user starts speaking
                in Soniox turn detection mode (vad_force_turn_endpoint=False). Only applies
                when using Soniox's built-in endpoint detection. Passed along to the
                user turn strategies this service recommends, which own the
                interruption; a user-supplied ``user_turn_strategies`` overrides the
                recommendation and this setting with it. Defaults to True.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the STTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model="stt-rt-v5",
            language=None,
            language_hints=None,
            language_hints_strict=None,
            context=None,
            enable_speaker_diarization=False,
            single_speaker=False,
            enable_language_identification=False,
            max_endpoint_delay_ms=None,
            endpoint_sensitivity=None,
            endpoint_latency_adjustment_level=None,
            client_reference_id=None,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # --- 3. Deprecated params overrides ---
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.model = params.model
                if params.audio_format is not None:
                    audio_format = params.audio_format
                if params.num_channels is not None:
                    num_channels = params.num_channels
                default_settings.language_hints = params.language_hints
                default_settings.language_hints_strict = params.language_hints_strict
                default_settings.context = params.context
                default_settings.enable_speaker_diarization = params.enable_speaker_diarization
                default_settings.enable_language_identification = (
                    params.enable_language_identification
                )
                default_settings.client_reference_id = params.client_reference_id

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=1,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )

        # Fired once per connection in single-speaker mode with the label kept.
        self._register_event_handler("on_speaker_locked")

        self._api_key = api_key
        self._url = url
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._should_interrupt = should_interrupt

        # Init-only audio config
        self._audio_format = audio_format
        self._num_channels = num_channels

        self._final_transcription_buffer = []
        self._last_tokens_received: float | None = None
        # Single-speaker mode: audio power per outgoing chunk (start_ms, end_ms,
        # mean square), the locked label and the caller's running level (see
        # ``_decide_segment``).
        self._locked_speaker: str | None = None
        self._caller_level: float | None = None
        # label -> [energy-weighted level sum, speech ms], until the lock is decided.
        self._speaker_levels: dict[str, list[float]] = {}
        self._energy_windows: list[tuple[float, float, float]] = []
        self._audio_ms_sent = 0.0

        # Turn tracking for Soniox turn-detection mode.
        self._user_turn_open = False

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Soniox STT supports metrics generation.
        """
        return True

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Request external turn strategies in Soniox's turn-detection mode.

        With ``vad_force_turn_endpoint=False`` Soniox's endpoint detection decides
        turn endings and this service proposes turn boundaries, so the user
        aggregator resolves those rather than running local VAD/smart-turn. In the
        default Pipecat mode (``vad_force_turn_endpoint=True``) the STT proposes
        no turns, so the defaults are left in place. Applied unless the user
        passed their own ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if not self._vad_force_turn_endpoint:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
        return frame

    async def _user_turn_started(self):
        """Propose a turn start — Soniox turn-detection mode only.

        Soniox has no speech-started event, so the turn opens on the local VAD
        signal when available (fast path) or on the first transcript token
        otherwise; whichever arrives first wins, the other is a no-op. The
        proposal goes out before any transcription frames for the turn are
        pushed.
        """
        if self._vad_force_turn_endpoint or self._user_turn_open:
            return
        self._user_turn_open = True
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _user_turn_stopped(self):
        """Propose a turn stop — Soniox turn-detection mode only.

        Broadcast right after the endpoint's finalized TranscriptionFrame (same
        downstream queue, so ordering is preserved).
        """
        if self._vad_force_turn_endpoint or not self._user_turn_open:
            return
        self._user_turn_open = False
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply settings delta and reconnect if anything changed.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._request_reconnect()

        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Soniox STT websocket connection.

        Args:
            frame: The end frame.
        """
        # The end-of-audio frame has to reach the socket before teardown closes
        # it. Trailing final tokens are not waited for: that would hold shutdown
        # open for a transcript that generates no further turn.
        await self._send_stop_recording()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Soniox STT websocket connection.

        Compared to stop, this closes the connection without sending the
        end-of-audio frame.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio data to Soniox STT Service.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if self._websocket and self._websocket.state is State.OPEN:
            self._record_energy(audio)
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")

        yield None

    def _record_energy(self, audio: bytes) -> None:
        """Track the stream clock and, in single-speaker mode, per-chunk loudness."""
        bytes_per_ms = self.sample_rate * 2 * self._num_channels / 1000
        if not bytes_per_ms:
            return
        start = self._audio_ms_sent
        self._audio_ms_sent += len(audio) / bytes_per_ms
        if self._settings.single_speaker:
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            power = float(np.mean(np.square(samples))) if samples.size else 0.0
            self._energy_windows.append((start, self._audio_ms_sent, power))
            # Soniox finalizes tokens within a few seconds; keep a generous tail.
            horizon = self._audio_ms_sent - ENERGY_HISTORY_MS
            while self._energy_windows and self._energy_windows[0][1] < horizon:
                self._energy_windows.pop(0)

    def _level_between(self, start_ms: float, end_ms: float) -> float | None:
        """RMS of the outgoing audio over a span, or None if that audio was not seen."""
        energy = duration = 0.0
        for window_start, window_end, power in self._energy_windows:
            overlap = min(end_ms, window_end) - max(start_ms, window_start)
            if overlap > 0:
                energy += power * overlap
                duration += overlap
        return float(np.sqrt(energy / duration)) if duration else None

    def _passes_level(self, speaker: str, level: float | None) -> bool:
        """Apply the loudness cross-check to a labelled span once locked.

        Speech carrying the caller's label always passes: callers speak softly
        too, and dropping their quiet words costs more than the odd mislabelled
        bystander it would catch. Labels flip on real-time audio, so speech from
        another label is kept when it is at least SINGLE_SPEAKER_LEVEL_RATIO of
        the caller's running level (most likely the caller mislabelled) and
        dropped when clearly quieter (a distant voice). Unmeasured spans pass.
        """
        if speaker == self._locked_speaker or level is None or self._caller_level is None:
            return True
        return level >= SINGLE_SPEAKER_LEVEL_RATIO * self._caller_level

    def _decide_segment(self) -> bool:
        """Lock or judge the finalized segment in the buffer. Returns whether to keep it.

        The lock is decided once two different labels have been heard in
        separate segments: the label with the louder average level is kept as
        the caller. A lone label never locks, so a bystander talking alone
        before the caller cannot become the lock. Segments before the lock pass.
        """
        tokens = [
            t
            for t in self._final_transcription_buffer
            if t.get("speaker") is not None and t.get("start_ms") is not None
        ]
        if not tokens:
            return True
        speaker = Counter(t["speaker"] for t in tokens).most_common(1)[0][0]
        start_ms, end_ms = tokens[0]["start_ms"], tokens[-1]["end_ms"]
        level = self._level_between(start_ms, end_ms)
        if level is None:
            return True
        if self._locked_speaker is None:
            totals = self._speaker_levels.setdefault(speaker, [0.0, 0.0])
            totals[0] += level * (end_ms - start_ms)
            totals[1] += end_ms - start_ms
            # Both voices must have said enough for their levels to mean anything;
            # one loud word must not outvote a caller's quiet sentence.
            heard = {
                s: e / d for s, (e, d) in self._speaker_levels.items() if d >= MIN_LOCK_SPEECH_MS
            }
            if len(heard) < 2:
                return True
            # ponytail: permanent lock; relock on prolonged silence if that bites.
            locked = max(heard, key=lambda s: heard[s])
            self._locked_speaker = locked
            self._caller_level = heard[locked]
            logger.info(f"{self}: single-speaker mode locked onto speaker {locked}")
        keep = self._passes_level(speaker, level)
        # Track the caller's level only from speech clearly at their level, so a
        # quiet mislabelled bystander that slipped through cannot drag it down.
        caller_level = self._caller_level
        if (
            keep
            and caller_level is not None
            and speaker == self._locked_speaker
            and level >= SINGLE_SPEAKER_LEVEL_RATIO * caller_level
        ):
            self._caller_level = 0.8 * caller_level + 0.2 * level
        return keep

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Processes a frame of audio data, either buffering or transcribing it.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # In Soniox turn-detection mode the local VAD signal is the fast
            # path for opening the turn — the first transcript token arrives a
            # full network round-trip plus model TTFB later and remains the
            # fallback when no VAD analyzer is configured. No-op in Pipecat
            # mode or mid-turn.
            if self._vad_passes_level(frame.start_secs):
                await self._user_turn_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame) and self._vad_force_turn_endpoint:
            # Send finalize message to Soniox so we get the final tokens asap.
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(FINALIZE_MESSAGE)
                logger.debug(f"Triggered finalize event on: {frame.name=}, {direction=}")

    async def _send_stop_recording(self):
        """Send the end-of-audio frame, an empty message that ends the session."""
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send("")
            except Exception as e:
                logger.warning(f"{self}: end-of-audio send failed: {e}")

    async def _connect(self):
        """Connect to the Soniox service.

        Establishes websocket connection and starts receive and keepalive tasks.
        """
        await self._connect_websocket()

        await super()._connect()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from the Soniox service.

        Cleans up tasks and closes websocket connection.
        """
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the websocket connection to Soniox."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Soniox STT")

            self._websocket = await self._websocket_connect(self._url)

            if not self._websocket:
                await self.push_error(error_msg=f"Unable to connect to Soniox API at {self._url}")
                raise Exception(f"Unable to connect to Soniox API at {self._url}")

            # If vad_force_turn_endpoint is not enabled, we need to enable endpoint detection.
            # Either one or the other is required.
            enable_endpoint_detection = not self._vad_force_turn_endpoint

            s = self._settings

            context = s.context
            if isinstance(context, SonioxContextObject):
                context = context.model_dump()

            # Send the initial configuration message.
            config = {
                "api_key": self._api_key,
                "model": s.model,
                "audio_format": self._audio_format,
                "num_channels": self._num_channels,
                "enable_endpoint_detection": enable_endpoint_detection,
                "max_endpoint_delay_ms": s.max_endpoint_delay_ms,
                "endpoint_sensitivity": s.endpoint_sensitivity,
                "endpoint_latency_adjustment_level": s.endpoint_latency_adjustment_level,
                "sample_rate": self.sample_rate,
                "language_hints": _prepare_language_hints(assert_given(s.language_hints)),
                "language_hints_strict": s.language_hints_strict,
                "context": context,
                "enable_speaker_diarization": s.enable_speaker_diarization,
                "enable_language_identification": s.enable_language_identification,
                "client_reference_id": s.client_reference_id,
            }

            # Send the configuration message.
            await self._websocket.send(json.dumps(config))

            await self._call_event_handler("on_connected")
            logger.debug("Connected to Soniox STT")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Soniox: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the websocket connection to Soniox."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from Soniox STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _vad_passes_level(self, start_secs: float | None) -> bool:
        """Whether speech the local VAD just detected may open a turn.

        Once single-speaker mode has locked, a VAD start alone would let any
        voice in the room interrupt the bot before Soniox has labelled a word.
        The audio behind the VAD decision is compared with the caller's running
        level; a quieter voice is left to the token path, which also knows its
        label. Before the lock, or without a level reading, the fast path stays.
        """
        if not self._settings.single_speaker or self._locked_speaker is None:
            return True
        window_ms = (start_secs or 0.35) * 1000
        level = self._level_between(self._audio_ms_sent - window_ms, self._audio_ms_sent)
        return self._passes_level("", level)

    def _accepts_token(self, token: dict[str, Any]) -> bool:
        """Decide whether an interim token may open a turn and show as interim text.

        Everything passes until single-speaker mode has locked onto a speaker
        (see ``_decide_segment``). After that a token from another label passes
        only when the audio heard since it started is about as loud as the
        caller. Finalized tokens are not judged here; the whole segment is
        judged at the endpoint by ``_decide_segment``. Unlabelled or untimed
        tokens always pass, so a stream without diarization behaves as before.
        """
        if not self._settings.single_speaker or self._locked_speaker is None:
            return True
        speaker, start = token.get("speaker"), token.get("start_ms")
        if speaker is None or start is None:
            return True
        return self._passes_level(speaker, self._level_between(start, self._audio_ms_sent))

    async def _receive_messages(self):
        """Receive and process websocket messages.

        Continuously processes messages from the websocket connection.
        """
        # Transcription frame will be only sent after we get the "endpoint" event.
        self._final_transcription_buffer = []
        self._locked_speaker = None
        self._caller_level = None
        self._speaker_levels = {}
        self._energy_windows = []
        self._audio_ms_sent = 0.0
        label_speakers = not self._settings.single_speaker

        async def send_endpoint_transcript():
            if self._settings.single_speaker and self._final_transcription_buffer:
                was_locked = self._locked_speaker is not None
                keep = self._decide_segment()
                if not was_locked and self._locked_speaker is not None:
                    await self._call_event_handler("on_speaker_locked", self._locked_speaker)
                if not keep:
                    self._final_transcription_buffer = []
                elif not self._user_turn_open:
                    # Kept on loudness after its interim tokens were refused: open
                    # the turn now so the transcript lands inside a user turn.
                    await self._user_turn_started()
            if self._final_transcription_buffer:
                text = _tokens_to_text(self._final_transcription_buffer, label_speakers)
                language = _language_from_tokens(self._final_transcription_buffer)
                # Report usage before the transcription frame so tracing can
                # attach it to the STT span the frame closes.
                await self.emit_stt_usage_metrics()
                # Soniox only pushes TranscriptionFrame when an end token is received,
                # so every TranscriptionFrame is inherently finalized
                await self.push_frame(
                    TranscriptionFrame(
                        text=text,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=language,
                        result=self._final_transcription_buffer,
                        finalized=True,
                    )
                )
                await self._handle_transcription(text, is_final=True, language=language)
                self._final_transcription_buffer = []

        async def finalize_turn():
            await send_endpoint_transcript()
            await self._user_turn_stopped()

        async for message in self._get_websocket():
            try:
                content = json.loads(message)

                tokens = content["tokens"]

                if tokens:
                    if len(tokens) == 1 and tokens[0]["text"] == FINALIZED_TOKEN:
                        # Ignore finalized token, prevent auto-finalize cycling.
                        pass
                    else:
                        # Got at least one token, so we can reset the auto finalize delay.
                        self._last_tokens_received = time.time()

                # We will only send the final tokens after we get the "endpoint" event.
                non_final_transcription = []

                for token in tokens:
                    accepted = self._accepts_token(token)
                    if not accepted and not token["is_final"]:
                        # A locked-out voice neither shows as interim text nor opens a turn.
                        continue
                    if accepted and not is_end_token(token):
                        # In Soniox turn-detection mode, the first token of a new
                        # turn opens it (no-op in Pipecat mode or when a VAD
                        # signal already opened the turn).
                        await self._user_turn_started()
                    # Finalized tokens always reach the buffer; the whole segment is
                    # judged at the endpoint (see _decide_segment).
                    if token["is_final"]:
                        if is_end_token(token):
                            # Found an endpoint, tokens until here will be sent as transcript,
                            # the rest will be sent as interim tokens (even final tokens).
                            await finalize_turn()
                        else:
                            self._final_transcription_buffer.append(token)
                    else:
                        non_final_transcription.append(token)

                if self._final_transcription_buffer or non_final_transcription:
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            # Even final tokens are sent as interim tokens as we want to send
                            # nicely formatted messages - therefore waiting for the endpoint.
                            text=_tokens_to_text(
                                self._final_transcription_buffer + non_final_transcription,
                                label_speakers,
                            ),
                            user_id=self._user_id,
                            timestamp=time_now_iso8601(),
                            result=self._final_transcription_buffer + non_final_transcription,
                        )
                    )

                error_code = content.get("error_code")
                error_message = content.get("error_message")
                if error_code or error_message:
                    # In case of error, still send the final transcript (if any remaining
                    # in the buffer) and close any open user turn.
                    await finalize_turn()
                    await self.push_error(
                        error_msg=f"Error: {error_code} (_receive_messages) - {error_message}"
                    )

                finished = content.get("finished")
                if finished:
                    # When finished, still send the final transcript (if any remaining
                    # in the buffer) and close any open user turn.
                    await finalize_turn()
                    logger.debug("Transcription finished.")
                    return

            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.warning(f"Error processing message: {e}")

    async def _send_keepalive(self, silence: bytes):
        """Send a Soniox protocol-level keepalive message.

        Args:
            silence: Silent PCM audio bytes (unused, Soniox uses a protocol message).
        """
        if self._websocket is None:
            logger.warning(f"{self}: websocket unavailable, skipping keepalive")
            return
        await self._websocket.send(KEEPALIVE_MESSAGE)
