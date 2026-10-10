"""Open the preserved development-best text checkpoint after other jobs exit."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = 'adapters/run20-step0032'
DIGEST = 'e0407665c4c7c4ce83046549f6977fe4cfa13c7a8a51a783b541c34146a374ba'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    with (ROOT / ADAPTER / 'adapters.safetensors').open('rb') as saved:
        if hashlib.file_digest(saved, 'sha256').hexdigest() != DIGEST:
            raise ValueError('Saved checkpoint differs from the reviewed version')
    from chat import resolve_inputs
    resolve_inputs(ADAPTER)
    print('Jun — experimental text conversation, running locally.', flush=True)
    print('This checkpoint scored 17/20 on development, but failed longer release checks.', flush=True)
    print('No persona system prompt. Type /reset for a fresh conversation or /quit to exit.', flush=True)
    with (ROOT / 'results' / '.text-model.lock').open('a') as lock:
        waiting = False
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not waiting:
                    print('\nAnother model job is running. Waiting for it to finish before loading Jun.', flush=True)
                    print('Leave this window open. Ctrl-C cancels this launcher only.', flush=True)
                    waiting = True
                if args.check_only:
                    print('Checkpoint verified; waiting is required. No model loaded.', flush=True)
                    return
                time.sleep(2)
    if args.check_only:
        print('Checkpoint verified; model lock available. No model loaded.', flush=True)
        return
    # chat.py independently reacquires the shared lock before any model import.
    # If another job wins the brief handoff, it fails closed instead of overlapping.
    print('\nLoading Jun…', flush=True)
    os.execv(sys.executable, [sys.executable, str(ROOT / 'scripts' / 'chat.py'),
                             '--adapter', ADAPTER])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nLauncher closed. Any training job continues unchanged.')
