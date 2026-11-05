# src/voice.py

import threading
import time
from typing import Optional


class VoiceGuide:
    """
    Thread-safe, non-blocking Natural Voice Guidance service for VisionAssist 2.0.
    
    Guarantees:
    - Native Windows SAPI.SpVoice for 100% reliable continuous audio playback.
    - Graceful pyttsx3 fallback.
    - Global speech pacing / debouncing: enforces a calm inter-sentence interval (2.5s - 3.0s)
      to prevent rapid chatter when scene elements fluctuate.
    - Instant silence/interruption support for emergency stop warnings and critical safety hazards.
    - Suppresses duplicate repetitions and jittery alternation.
    - Never blocks the UI or OpenCV frame processing loop.
    """

    def __init__(self, rate: int = 1, cooldown_seconds: float = 3.0, global_pacing_seconds: float = 2.5):
        self.rate = rate
        self.cooldown = cooldown_seconds
        self.global_pacing = global_pacing_seconds

        self._last_spoken_sentence: str = ""
        self._last_spoken_time: float = 0.0
        self._last_request_time: float = 0.0
        self._pending_message: Optional[str] = None
        self._is_speaking: bool = False
        self._lock = threading.Lock()
        self._running = True

        self._speaker = None
        self._engine_type = "none"

        # Launch dedicated background daemon speech worker
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _get_speaker(self):
        """Initialize native Windows SAPI voice or pyttsx3 engine."""
        try:
            import pythoncom
            import win32com.client
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = self.rate
            return speaker, "sapi"
        except Exception:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 165)
                return engine, "pyttsx3"
            except Exception as e:
                print(f"[VoiceGuide Warning] Voice initialization failed: {e}")
                return None, "none"

    def _speech_worker(self) -> None:
        """Background loop continuously servicing speech requests."""
        self._speaker, self._engine_type = self._get_speaker()
        if not self._speaker:
            return

        while self._running:
            msg_to_speak: Optional[str] = None

            with self._lock:
                if self._pending_message:
                    msg_to_speak = self._pending_message
                    self._pending_message = None
                    self._is_speaking = True

            if msg_to_speak:
                try:
                    if self._engine_type == "sapi":
                        self._speaker.Speak(msg_to_speak)
                    elif self._engine_type == "pyttsx3":
                        self._speaker.say(msg_to_speak)
                        self._speaker.runAndWait()
                except Exception as e:
                    print(f"[VoiceGuide Error] Speech playback error: {e}")

                with self._lock:
                    self._last_spoken_sentence = msg_to_speak
                    self._last_spoken_time = time.time()
                    if not self._pending_message:
                        self._is_speaking = False
            else:
                time.sleep(0.04)

    def silence(self) -> None:
        """Immediately interrupt any ongoing speech and purge pending queues."""
        with self._lock:
            self._pending_message = None
            self._is_speaking = False

        if self._speaker:
            try:
                if self._engine_type == "sapi":
                    # SVSFPurgeBeforeSpeak = 2: purge active/pending audio buffer
                    self._speaker.Speak("", 2)
                elif self._engine_type == "pyttsx3":
                    self._speaker.stop()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        """Return whether speech is currently in progress."""
        with self._lock:
            return self._is_speaking or (self._pending_message is not None)

    def speak(self, message: str, force: bool = False, urgent: bool = False) -> None:
        """
        Request spoken guidance for a message.
        
        Args:
            message: Guidance sentence to speak.
            force: If True, immediately interrupts current speech and forces audio output.
            urgent: If True, bypasses non-critical pacing delays.
        """
        if not message or not message.strip():
            return

        clean_msg = message.strip()
        now = time.time()
        msg_lower = clean_msg.lower()
        is_emergency = force or urgent or ("warning" in msg_lower) or ("stop" in msg_lower) or ("caution" in msg_lower)

        if force:
            self.silence()

        with self._lock:
            # 1. Urgent / Emergency messages bypass pacing delays
            if not is_emergency:
                # Rule A: If currently speaking, don't overwrite with non-emergency chatter
                if self._is_speaking:
                    return

                # Rule B: Global pacing cooldown (avoid back-to-back chatter even if sentences differ)
                if (now - self._last_spoken_time) < self.global_pacing:
                    return

                # Rule C: Duplicate sentence suppression cooldown
                if clean_msg == self._last_spoken_sentence and (now - self._last_spoken_time) < self.cooldown:
                    return

            # Set as pending message
            self._pending_message = clean_msg
            self._last_request_time = now

    def stop(self) -> None:
        """Gracefully terminate background thread."""
        self.silence()
        self._running = False