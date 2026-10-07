"""Enumeration types for pipecat utilities."""

from enum import Enum


class EndTaskReason(Enum):
    """Reasons for ending a task."""

    CALL_DURATION_EXCEEDED = "call_duration_exceeded"
    CALL_TRANSFERRED = "call_transferred"
    END_CALL = "end_call"
    # HireStella: kept from pipecat 1.7 so run history, exports and the
    # disposition filter keep these values (upstream folded both into END_CALL).
    END_CALL_TOOL_REASON = "end_call_tool"
    USER_QUALIFIED = "user_qualified"
    VOICEMAIL_DETECTED = "voicemail_detected"
    USER_IDLE_MAX_DURATION_EXCEEDED = "user_idle_max_duration_exceeded"
    USER_HANGUP = "user_hangup"
    SYSTEM_CANCELLED = "system_cancelled"
    UNEXPECTED_ERROR = "unexpected_error"
    TRANSFER_CALL = "transfer_call"
    PIPELINE_ERROR = "pipeline_error"


class RealtimeFeedbackType(Enum):
    """Message types for real-time feedback events."""

    USER_TRANSCRIPTION = "rtf-user-transcription"
    BOT_TEXT = "rtf-bot-text"
    FUNCTION_CALL_START = "rtf-function-call-start"
    FUNCTION_CALL_END = "rtf-function-call-end"
    TTFB_METRIC = "rtf-ttfb-metric"
    NODE_TRANSITION = "rtf-node-transition"
    LATENCY_MEASURED = "rtf-latency-measured"
    PIPELINE_ERROR = "rtf-pipeline-error"
    BOT_STARTED_SPEAKING = "rtf-bot-started-speaking"
    BOT_STOPPED_SPEAKING = "rtf-bot-stopped-speaking"
    USER_MUTE_STARTED = "rtf-user-mute-started"
    USER_MUTE_STOPPED = "rtf-user-mute-stopped"
