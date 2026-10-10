"""Keep audio ordering identical when removing the mixed trial's text updates."""
import random


def make_schedule(text_ids, audio_ids, seed, *, audio_only=False):
    if not text_ids or len(text_ids) != len(audio_ids):
        raise ValueError('Require one distinct text and audio episode per pair')
    if len(set(text_ids)) != len(text_ids) or len(set(audio_ids)) != len(audio_ids):
        raise ValueError('Repeated episode in schedule')
    text_order, audio_order = list(text_ids), list(audio_ids)
    rng = random.Random(seed)
    # Preserve both RNG draws even in the audio-only control.
    rng.shuffle(text_order)
    rng.shuffle(audio_order)
    mixed = [dict(modality=kind, id=ident)
             for pair in zip(text_order, audio_order)
             for kind, ident in zip(('text', 'audio'), pair)]
    return [item for item in mixed if item['modality'] == 'audio'] if audio_only else mixed
