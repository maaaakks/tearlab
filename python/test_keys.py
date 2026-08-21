"""Send the full start sequence to Isaac via PostMessage.

Sequence: Enter → Enter → Right → Enter → Enter → Enter
"""
import ctypes
import ctypes.wintypes
import time

from win32_utils import VK_RETURN, VK_RIGHT, send_key

user32 = ctypes.windll.user32

def find_isaac():
    windows = []
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                if "isaac" in buf.value.lower() and "steam" not in buf.value.lower():
                    windows.append((hwnd, buf.value))
        return True
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(cb), 0)
    return windows


# --- Main ---
def main():
    windows = find_isaac()
    if not windows:
        print("No Isaac window found!")
        return 1

    hwnd, title = windows[0]
    print(f"Found: {title} (HWND={hwnd})")

    # The start sequence: Enter Enter Right Enter Enter Enter
    sequence = [
        ("Enter (skip intro)",    VK_RETURN, 1.5),
        ("Enter (sikp title page)",   VK_RETURN, 1.5),
        ("Right (move to middle save file)", VK_RIGHT,  1.5),
        ("Enter (select save file)",       VK_RETURN, 1.5),
        ("Enter (continue/new run)",  VK_RETURN, 0.5),
        ("Enter (confirm character if new run)",        VK_RETURN, 0.3),
    ]

    print("\nSending start sequence in 3 seconds...")
    time.sleep(3.0)

    for label, vk, delay in sequence:
        print(f"  -> {label}")
        send_key(hwnd, vk)
        time.sleep(delay)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
