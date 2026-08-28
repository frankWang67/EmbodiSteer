try:
    from pynput.keyboard import Key, KeyCode, Listener
    PYNPUT_IMPORT_ERROR = None
except Exception as exc:  # headless machines can still inspect --help
    PYNPUT_IMPORT_ERROR = exc

    class _Key:
        backspace = object()

    class KeyCode:
        def __init__(self, char=None):
            self.char = char

        def __eq__(self, other):
            return isinstance(other, KeyCode) and self.char == other.char

        def __hash__(self):
            return hash(self.char)

    class Listener:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stop()

        def start(self):
            return self

        def stop(self):
            return None

    Key = _Key()
from collections import defaultdict
from threading import Lock

class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        if PYNPUT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Keyboard input requires a graphical pynput backend. "
                "The release tree can be inspected headlessly, but real "
                "deployment must run with DISPLAY/input permissions."
            ) from PYNPUT_IMPORT_ERROR
        return super().start()
    
    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
    
    def on_release(self, key):
        pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            return events

if __name__ == '__main__':
    import time
    with KeystrokeCounter() as counter:
        try:
            while True:
                print('Space:', counter[Key.space])
                print('q:', counter[KeyCode(char='q')])
                time.sleep(1/60)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)
