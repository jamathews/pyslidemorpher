#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from pyslidemorpher.realtime import _play_audio_loop
from pathlib import Path
import time
import threading

# Test if audio function works
audio_files = list(Path('assets/audio').glob('*.mp3'))
if audio_files:
    print(f'Testing audio with: {audio_files[0]}')
    try:
        import pygame
        pygame.mixer.init()
        thread = threading.Thread(target=_play_audio_loop, args=(audio_files[0],), daemon=True)
        thread.start()
        print('Audio started, playing for 3 seconds...')
        time.sleep(3)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        print('Audio test completed successfully!')
    except Exception as e:
        print(f'Audio test failed: {e}')
else:
    print('No audio files found for testing')