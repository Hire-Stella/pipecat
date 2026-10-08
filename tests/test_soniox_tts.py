#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import unittest

from websockets.protocol import State

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.soniox.tts import SonioxTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.context.word_completion_tracker import WordCompletionTracker
from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text


def _service(language: str | None) -> SonioxTTSService:
    service = SonioxTTSService.__new__(SonioxTTSService)
    service._name = "SonioxTTSService#0"
    service._settings = TTSSettings(language=language)
    service._partials = {}
    return service


def _timestamps(text: str, start: float = 0.0) -> dict:
    chars = list(text)
    return {
        "characters": chars,
        "character_start_times_seconds": [round(start + i * 0.1, 1) for i in range(len(chars))],
    }


def test_soniox_english_characters_assemble_into_words():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hi world. ")) == [
        ("Hi", 0.0),
        ("world.", 0.3),
    ]


def test_soniox_english_partial_word_carries_across_messages():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hel")) == []
    assert service._to_word_times("s1", _timestamps("lo you ", start=0.3)) == [
        ("Hello", 0.0),
        ("you", 0.6),
    ]


def test_soniox_english_final_partial_word_is_buffered_for_terminated():
    service = _service("en")

    assert service._to_word_times("s1", _timestamps("Hi you")) == [("Hi", 0.0)]
    # The receive loop flushes this buffered word when Soniox sends `terminated`.
    assert service._partials["s1"] == ("you", 0.3)


def test_soniox_streams_buffer_partial_words_independently():
    service = _service("en")

    service._to_word_times("s1", _timestamps("Hel"))
    service._to_word_times("s2", _timestamps("wor", start=1.0))

    assert service._partials["s1"] == ("Hel", 0.0)
    assert service._partials["s2"] == ("wor", 1.0)

    assert service._to_word_times("s1", _timestamps("lo ", start=0.3)) == [("Hello", 0.0)]
    assert service._to_word_times("s2", _timestamps("ld ", start=1.3)) == [("world", 1.0)]


def test_soniox_timestamp_length_mismatch_returns_empty():
    service = _service("en")

    assert (
        service._to_word_times(
            "s1", {"characters": ["H", "i"], "character_start_times_seconds": [0.0]}
        )
        == []
    )
    assert "s1" not in service._partials


def test_soniox_japanese_timestamps_emit_per_character():
    service = _service("ja")

    assert service._to_word_times("s1", _timestamps("こんにちは、私")) == [
        ("こ", 0.0),
        ("ん", 0.1),
        ("に", 0.2),
        ("ち", 0.3),
        ("は", 0.4),
        ("私", 0.6),
    ]
    assert "s1" not in service._partials


def test_soniox_chinese_timestamps_emit_per_character():
    service = _service("zh")

    assert service._to_word_times("s1", _timestamps("你好，世界。")) == [
        ("你", 0.0),
        ("好", 0.1),
        ("世", 0.3),
        ("界", 0.4),
    ]


def test_soniox_japanese_tokens_concatenate_without_spaces():
    service = _service("ja")
    tokens = service._to_word_times("s1", _timestamps("こんにちは、私"))

    includes_inter_frame_spaces = service._is_chinese_or_japanese_language()
    assert (
        concatenate_aggregated_text(
            [
                TextPartForConcatenation(
                    word, includes_inter_part_spaces=includes_inter_frame_spaces
                )
                for word, _start in tokens
            ]
        )
        == "こんにちは私"
    )


def test_soniox_japanese_punctuation_recovered_by_word_tracker():
    # CJK tokens drop punctuation (isalnum filter), which is safe: the frame
    # sequencer's WordCompletionTracker matches tokens by alphanumeric content
    # only and commits spans of the original text, sweeping adjacent punctuation
    # into each consumed span.
    text = "こんにちは、私はAIです。"
    service = _service("ja")
    tokens = service._to_word_times("s1", _timestamps(text))

    tracker = WordCompletionTracker(text)
    complete = False
    for word, _start in tokens:
        complete = tracker.add_word_and_check_complete(word)

    assert complete
    assert tracker.get_accumulated_user_facing_text() == text


def test_soniox_separates_consecutive_sentences_with_a_space():
    """Every run_tts payload is appended to one Soniox stream, so the text the
    server sees is the concatenation of all sentences in a turn. Without a
    trailing space they arrive glued ("Sam.Who"), and Soniox's normalizer reads
    the period as the literal word "dot" instead of a sentence break.
    """
    import re

    service = SonioxTTSService(api_key="not-a-real-key")
    sentences = [
        "Hello, thanks for calling.",
        "My name is Sam.",
        "Who do I have the pleasure of speaking with today?",
    ]

    wire = "".join(service._prepare_text_for_tts(s) for s in sentences)

    assert "Sam. Who" in wire
    assert not re.search(r"\w\.\w", wire), f"sentences glued on the wire: {wire!r}"


class _FakeSonioxWebSocket:
    """Stands in for the websockets connection: records sends, never replies."""

    def __init__(self):
        self.sent: list[dict] = []
        self.state = State.OPEN
        self._closed = asyncio.Event()

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration

    async def close(self):
        self.state = State.CLOSED
        self._closed.set()


class TestSonioxStreamConfig(unittest.IsolatedAsyncioTestCase):
    async def test_run_tts_configures_a_stream_the_connection_has_not_seen(self):
        """The eager config send at turn start does not survive a reconnect:
        ``_disconnect_websocket`` clears ``_configured_contexts``. Text for the
        stream must then be preceded by a fresh config, or Soniox rejects it
        with ``invalid_stream_state`` and the turn produces no audio."""
        ws = _FakeSonioxWebSocket()
        service = SonioxTTSService(api_key="not-a-real-key")

        async def fake_connect(url):
            return ws

        service._websocket_connect = fake_connect
        cleared_at: list[int] = []

        async def drop_config_like_a_reconnect():
            while not any("model" in m for m in ws.sent):
                await asyncio.sleep(0.01)
            service._configured_contexts.clear()
            cleared_at.append(len(ws.sent))

        driver = asyncio.create_task(drop_config_like_a_reconnect())
        try:
            await run_test(
                service,
                frames_to_send=[
                    LLMFullResponseStartFrame(),
                    SleepFrame(sleep=0.2),
                    TextFrame("Hello there."),
                    LLMFullResponseEndFrame(),
                    SleepFrame(sleep=0.2),
                ],
            )
        finally:
            driver.cancel()

        self.assertEqual(len(cleared_at), 1, ws.sent)
        after_clear = ws.sent[cleared_at[0] :]
        first_text = next(i for i, m in enumerate(after_clear) if m.get("text"))
        stream_id = after_clear[first_text]["stream_id"]
        configs_before_text = [
            m for m in after_clear[:first_text] if "model" in m and m["stream_id"] == stream_id
        ]
        self.assertEqual(len(configs_before_text), 1, after_clear)
