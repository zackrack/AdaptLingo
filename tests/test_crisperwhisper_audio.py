import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch


def load_audio_module():
    module_path = Path(__file__).parents[1] / "helpers" / "audio.py"
    spec = importlib.util.spec_from_file_location("audio_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    with patch("transformers.pipeline", return_value=Mock()):
        spec.loader.exec_module(module)
    return module


class TranscribeAudioTests(TestCase):
    def test_uses_crisperwhisper_verbatim_mode(self):
        audio = load_audio_module()
        model = Mock()
        model.transcribe.return_value = SimpleNamespace(text="  Um, hello.  ")

        transcription, audio_path = audio.transcribe_audio("recording.wav", model)

        self.assertEqual(transcription, "Um, hello.")
        self.assertEqual(audio_path, "recording.wav")
        model.transcribe.assert_called_once_with(
            "recording.wav",
            language="en",
            mode="verbatim",
        )

    def test_wraps_transcription_errors(self):
        audio = load_audio_module()
        model = Mock()
        model.transcribe.side_effect = ValueError("invalid audio")

        with self.assertRaisesRegex(RuntimeError, "Failed to transcribe audio"):
            audio.transcribe_audio("recording.wav", model)
