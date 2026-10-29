"""Record human CS2 screen/input data plus official GSI, without training.

Run this script, focus the game window, and play. Press Ctrl+C in the recorder
terminal when the demonstration is complete. Screen frames are saved as JPEGs;
keyboard and mouse transitions plus frame-aligned input snapshots are written
to inputs.jsonl.
"""

import argparse
import ctypes
from ctypes import wintypes
import json
import time
from datetime import datetime
from pathlib import Path

from cs2.gsi import GSIReceiver
from cs2.state import SCHEMA_VERSION, FEATURE_NAMES, encode_gsi


user32 = ctypes.windll.user32


def _virtual_keys():
    keys = set(range(0x30, 0x3A)) | set(range(0x41, 0x5B))
    keys.update(range(0x60, 0x70))  # numpad digits and operators
    keys.update(range(0x70, 0x88))  # F1-F24
    keys.update(
        {
            0x08, 0x09, 0x0D, 0x10, 0x11, 0x12, 0x13, 0x14, 0x1B, 0x20,
            0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D,
            0x2E, 0x5B, 0x5C, 0x5D,
        }
    )
    keys.update(range(0xA0, 0xA6))  # left/right Shift, Ctrl, Alt
    keys.update(range(0xBA, 0xC1))
    keys.update(range(0xDB, 0xE3))
    return sorted(keys)


def _key_name(vk):
    fixed = {
        0x08: "BACKSPACE", 0x09: "TAB", 0x0D: "ENTER", 0x10: "SHIFT",
        0x11: "CTRL", 0x12: "ALT", 0x13: "PAUSE", 0x14: "CAPSLOCK",
        0x1B: "ESC", 0x20: "SPACE", 0x21: "PAGEUP", 0x22: "PAGEDOWN",
        0x23: "END", 0x24: "HOME", 0x25: "LEFT", 0x26: "UP",
        0x27: "RIGHT", 0x28: "DOWN", 0x2C: "PRINTSCREEN", 0x2D: "INSERT",
        0x2E: "DELETE", 0x5B: "LWIN", 0x5C: "RWIN", 0x5D: "MENU",
        0x6A: "NUM_MULTIPLY", 0x6B: "NUM_ADD", 0x6D: "NUM_SUBTRACT",
        0x6E: "NUM_DECIMAL", 0x6F: "NUM_DIVIDE", 0x90: "NUMLOCK",
        0x91: "SCROLLLOCK", 0xA0: "LSHIFT", 0xA1: "RSHIFT",
        0xA2: "LCTRL", 0xA3: "RCTRL", 0xA4: "LALT", 0xA5: "RALT",
        0xBA: ";", 0xBB: "=", 0xBC: ",", 0xBD: "-", 0xBE: ".",
        0xBF: "/", 0xC0: "`", 0xDB: "[", 0xDC: "\\", 0xDD: "]",
        0xDE: "'", 0xDF: "OEM8", 0xE2: "OEM102",
    }
    if vk in fixed:
        return fixed[vk]
    if 0x30 <= vk <= 0x39 or 0x41 <= vk <= 0x5A:
        return chr(vk)
    if 0x60 <= vk <= 0x69:
        return f"NUM{vk - 0x60}"
    if 0x70 <= vk <= 0x87:
        return f"F{vk - 0x6F}"
    return f"VK_{vk:02X}"


def _pressed(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def _mouse_position():
    point = wintypes.POINT()
    if user32.GetCursorPos(ctypes.byref(point)):
        return point.x, point.y
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record screen, keyboard, and mouse data for a human demonstration."
    )
    parser.add_argument("--output", type=Path, default=Path("demos"),
                        help="Output folder (default: demos). A timestamped subfolder is created.")
    parser.add_argument("--left", type=int, default=0, help="Capture region left edge (default: 0).")
    parser.add_argument("--top", type=int, default=0, help="Capture region top edge (default: 0).")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640).")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480).")
    parser.add_argument("--fps", type=float, default=10.0, help="Saved screen frames per second (default: 10).")
    parser.add_argument("--input-hz", type=float, default=100.0,
                        help="Keyboard/mouse polling rate (default: 100).")
    parser.add_argument("--gsi-port", type=int, default=3000, help="Local GSI listener port (default: 3000).")
    parser.add_argument("--gsi-token", help="Optional auth.token matching the game configuration.")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.fps <= 0 or args.input_hz <= 0:
        parser.error("width, height, fps, and input-hz must be positive")
    return args


def main():
    args = parse_args()
    try:
        import cv2
        import mss
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Screen recording dependencies are missing. Install them with: "
            "python -m pip install -r requirements-cs2.txt"
        ) from error
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output / f"demo_{timestamp}"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    region = {"left": args.left, "top": args.top, "width": args.width, "height": args.height}
    key_codes = _virtual_keys()
    key_names = {vk: _key_name(vk) for vk in key_codes}
    mouse_codes = {0x01: "MOUSE_LEFT", 0x02: "MOUSE_RIGHT", 0x04: "MOUSE_MIDDLE",
                   0x05: "MOUSE_X1", 0x06: "MOUSE_X2"}
    watched_codes = key_codes + list(mouse_codes)
    names = {**key_names, **mouse_codes}

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "capture_region": region,
        "frame_fps": args.fps,
        "input_poll_hz": args.input_hz,
        "frame_format": "RGB JPEG",
        "input_format": "JSONL; relative timestamps use monotonic seconds",
        "coordinate_note": "mouse_x/y are desktop coordinates; capture_x/y are relative to capture region",
        "observation_schema": "cs2-screen-gsi-v1",
        "game_state_source": "Official CS2 Game State Integration HTTP POST",
        "state_schema": SCHEMA_VERSION,
        "state_feature_names": FEATURE_NAMES,
        "mouse_note": "Desktop cursor deltas are diagnostics, not raw relative mouse input or game view angles; do not use them as aim labels.",
        "training_status": "CS2 sequence trainer is not implemented yet.",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Recording directory: {output_dir.resolve()}")
    print(f"Capture region: left={args.left}, top={args.top}, width={args.width}, height={args.height}")
    print("Switch to the game window and play. Press Ctrl+C to stop recording.")

    key_state = {vk: False for vk in watched_codes}
    pressed_since_frame = set()
    start = time.monotonic()
    next_poll = start
    next_frame = start
    frame_index = 0
    previous_mouse = None
    poll_interval = 1.0 / args.input_hz
    frame_interval = 1.0 / args.fps
    try:
        with GSIReceiver(port=args.gsi_port, token=args.gsi_token) as gsi, mss.mss() as screen, (output_dir / "inputs.jsonl").open(
            "w", encoding="utf-8", buffering=1
        ) as inputs:
            while True:
                now = time.monotonic()
                elapsed = now - start

                for vk in watched_codes:
                    current = _pressed(vk)
                    if current != key_state[vk]:
                        key_state[vk] = current
                        if current:
                            pressed_since_frame.add(names[vk])
                        x, y = _mouse_position()
                        inputs.write(json.dumps({
                            "type": "input", "t": round(elapsed, 4),
                            "event": "down" if current else "up", "control": names[vk],
                            "mouse_x": x, "mouse_y": y,
                        }, ensure_ascii=False) + "\n")

                if now >= next_frame:
                    raw = np.asarray(screen.grab(region))
                    rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    frame_name = f"frame_{frame_index:06d}.jpg"
                    if not cv2.imwrite(str(frames_dir / frame_name), bgr,
                                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
                        raise OSError(f"Could not save {frames_dir / frame_name}")
                    # Use only the latest GSI received before capture began.
                    snapshot = gsi.store.at_or_before(now)
                    state = encode_gsi(snapshot, at=now)
                    x, y = _mouse_position()
                    down = sorted(names[vk] for vk, value in key_state.items() if value)
                    inputs.write(json.dumps({
                        "type": "frame", "t": round(elapsed, 4), "frame": f"frames/{frame_name}",
                        "keys_down": down, "mouse_x": x, "mouse_y": y,
                        "pressed_controls": sorted(pressed_since_frame),
                        "capture_x": None if x is None else x - args.left,
                        "capture_y": None if y is None else y - args.top,
                        "desktop_cursor_delta_x": None if x is None or previous_mouse is None else x - previous_mouse[0],
                        "desktop_cursor_delta_y": None if y is None or previous_mouse is None else y - previous_mouse[1],
                        "gsi_sequence": None if snapshot is None else snapshot.sequence,
                        "gsi_received_t": None if snapshot is None else round(snapshot.received_at - start, 4),
                        "gsi_age_seconds": state.age_seconds,
                        "gsi_fresh": state.fresh,
                        "gsi_values": state.values,
                        "gsi_available": state.available,
                        "gsi": None if snapshot is None else snapshot.data,
                    }, ensure_ascii=False) + "\n")
                    previous_mouse = None if x is None or y is None else (x, y)
                    pressed_since_frame.clear()
                    frame_index += 1
                    next_frame += frame_interval
                    if next_frame < now:
                        next_frame = now + frame_interval

                next_poll += poll_interval
                delay = next_poll - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_poll = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        duration = time.monotonic() - start
        with (output_dir / "inputs.jsonl").open("a", encoding="utf-8", buffering=1) as inputs:
            for vk, is_down in key_state.items():
                if is_down:
                    inputs.write(json.dumps({
                        "type": "input", "t": round(duration, 4), "event": "up",
                        "control": names[vk], "mouse_x": None, "mouse_y": None,
                    }, ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(json.dumps({
            "duration_seconds": round(duration, 3),
            "saved_frames": frame_index,
            "stopped_at": datetime.now().astimezone().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Recording finished. Saved {frame_index} frames to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
