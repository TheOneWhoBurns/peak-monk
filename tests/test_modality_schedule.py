import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from moshi_modality_schedule import make_schedule


def test_audio_control_matches_historical_mixed_order():
    text = [f'text-{i}' for i in range(24)]
    audio = [f'audio-{i}' for i in range(24)]
    original_text, original_audio = list(text), list(audio)
    rng = random.Random(307)
    rng.shuffle(original_text)
    rng.shuffle(original_audio)
    schedule = make_schedule(text, audio, 307, audio_only=True)
    assert [r['id'] for r in schedule] == original_audio
    assert all(r['modality'] == 'audio' for r in schedule)
    assert schedule == make_schedule(text, audio, 307)[1::2]
    wrong_order = list(audio)
    random.Random(307).shuffle(wrong_order)
    assert wrong_order != original_audio  # Catches skipping the original text shuffle.
    assert text == [f'text-{i}' for i in range(24)]


def test_invalid_schedule_fails_closed():
    import pytest
    for text, audio in [([], []), (['t'], ['a', 'b']), (['t', 't'], ['a', 'b']), (['t', 'u'], ['a', 'a'])]:
        with pytest.raises(ValueError):
            make_schedule(text, audio, 307, audio_only=True)
