"""AudioAdapter + make_audio_adapter — the dashboard-TTS speak path.

Pure-logic: no robot, no DDS. KovioTtsAdapter.speak is exercised with
subprocess.Popen monkeypatched so we assert the exact argv/env it would launch
without running the real kovio_tts binary.
"""
from __future__ import annotations

import os

import pytest

from kovio.adapters.audio import (
    KovioTtsAdapter,
    LoggingAudioAdapter,
    make_audio_adapter,
)


class _FakeProc:
    def __init__(self, argv, rc=0, out=""):
        self.argv = argv
        self.returncode = rc
        self._out = out
        self.pid = 4242

    def communicate(self, timeout=None):
        return self._out, ""


def _patch_popen(monkeypatch, sink):
    import kovio.adapters.audio as audio_mod

    def fake_popen(argv, **kwargs):
        proc = _FakeProc(argv)
        sink.append((argv, kwargs))
        return proc

    monkeypatch.setattr(audio_mod.subprocess, "Popen", fake_popen)


def test_make_audio_adapter_logger_and_none():
    assert isinstance(make_audio_adapter("logger"), LoggingAudioAdapter)
    assert isinstance(make_audio_adapter("none"), LoggingAudioAdapter)


def test_make_audio_adapter_defaults_to_logger_without_binary(monkeypatch):
    monkeypatch.delenv("KOVIO_TTS_BIN", raising=False)
    monkeypatch.delenv("KOVIO_AUDIO", raising=False)
    assert isinstance(make_audio_adapter(), LoggingAudioAdapter)


def test_make_audio_adapter_missing_binary_falls_back(monkeypatch):
    monkeypatch.setenv("KOVIO_TTS_BIN", "/no/such/kovio_tts")
    assert isinstance(make_audio_adapter(), LoggingAudioAdapter)


def test_make_audio_adapter_kovio_tts_requires_bin(monkeypatch):
    monkeypatch.delenv("KOVIO_TTS_BIN", raising=False)
    with pytest.raises(ValueError):
        make_audio_adapter("kovio_tts")


def test_make_audio_adapter_builds_kovio_tts(monkeypatch, tmp_path):
    binary = tmp_path / "kovio_tts"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("KOVIO_TTS_BIN", str(binary))
    monkeypatch.setenv("KOVIO_TTS_IFACE", "eth0")
    adapter = make_audio_adapter()
    assert isinstance(adapter, KovioTtsAdapter)


def test_speak_launches_binary_with_iface_text_volume(monkeypatch, tmp_path):
    binary = tmp_path / "kovio_tts"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    launches: list = []
    _patch_popen(monkeypatch, launches)

    adapter = KovioTtsAdapter(binary=str(binary), iface="eth0", default_volume=85)
    adapter.speak("hi i am kovios robot", volume=100)

    assert len(launches) == 1
    argv, kwargs = launches[0]
    assert argv == [str(binary), "eth0", "hi i am kovios robot", "100"]
    # env carries LD_LIBRARY_PATH only when a lib_dir was given (none here).
    assert "env" in kwargs


def test_speak_uses_default_volume_and_clamps(monkeypatch, tmp_path):
    launches: list = []
    _patch_popen(monkeypatch, launches)
    adapter = KovioTtsAdapter(binary="/bin/true", iface="eth0", default_volume=85)

    adapter.speak("hello")  # default volume
    adapter.speak("loud", volume=999)  # clamped to 100

    assert launches[0][0][3] == "85"
    assert launches[1][0][3] == "100"


def test_speak_ignores_blank_text(monkeypatch):
    launches: list = []
    _patch_popen(monkeypatch, launches)
    KovioTtsAdapter(binary="/bin/true").speak("   ")
    assert launches == []


def test_speak_sets_ld_library_path_when_lib_dir_given(monkeypatch):
    launches: list = []
    _patch_popen(monkeypatch, launches)
    adapter = KovioTtsAdapter(binary="/bin/true", lib_dir="/opt/kovio/lib")
    adapter.speak("hey")
    env = launches[0][1]["env"]
    assert env["LD_LIBRARY_PATH"].startswith("/opt/kovio/lib")
