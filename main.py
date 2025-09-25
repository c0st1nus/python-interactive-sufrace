# -*- coding: utf-8 -*-
import cv2
import video
import numpy as np
import os
import subprocess
import sys
import time
import ctypes
from typing import Optional, Tuple

# Cross-platform imports                                                                                                                                                                                                                                                                                                                                                                    
try:
    from pynput.mouse import Button, Listener as MouseListener
    from pynput import mouse
    import pynput
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("Warning: pynput not available. Mouse control disabled.")

# Cross-platform screen resolution
def get_screen_resolution():
    try:
        if sys.platform.startswith('win'):
            # Windows: use WinAPI and enable DPI awareness to get true pixels
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        elif sys.platform.startswith('linux'):
            # Try xrandr first
            result = subprocess.run(['xrandr'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ' connected primary ' in line or ' connected ' in line:
                        parts = line.split()
                        for part in parts:
                            if 'x' in part and '+' in part:
                                resolution = part.split('+')[0]
                                width, height = map(int, resolution.split('x'))
                                return width, height
            
            # Fallback to xdpyinfo
            result = subprocess.run(['xdpyinfo'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'dimensions:' in line:
                        dimensions = line.split('dimensions:')[1].split('pixels')[0].strip()
                        width, height = map(int, dimensions.split('x'))
                        return width, height
        
        # Default fallback resolution
        return 1920, 1080
    except Exception as e:
        print(f"Error getting screen resolution: {e}")
        return 1920, 1080

# Cross-platform beep function
def play_beep(frequency=1000, duration=50):
    try:
        if sys.platform.startswith('win'):
            try:
                import winsound
                winsound.Beep(int(frequency), int(duration))
                return
            except Exception:
                pass
        if sys.platform.startswith('linux'):
            # Use system beep or paplay
            try:
                subprocess.run(['paplay', '/usr/share/sounds/alsa/Front_Left.wav'], 
                             capture_output=True, timeout=1)
            except:
                try:
                    os.system('echo "\a"')
                except:
                    pass
    except Exception as e:
        print(f"Beep failed: {e}")

# Cross-platform mouse control
class CrossPlatformMouse:
    def __init__(self):
        self.controller = None
        if PYNPUT_AVAILABLE:
            try:
                self.controller = pynput.mouse.Controller()
            except Exception as e:
                print(f"Mouse controller init failed: {e}")
    
    def set_cursor_pos(self, x, y):
        if self.controller:
            try:
                self.controller.position = (x, y)
            except Exception as e:
                print(f"Set cursor pos failed: {e}")
    
    def mouse_down(self, x, y):
        if self.controller:
            try:
                self.controller.press(pynput.mouse.Button.left)
            except Exception as e:
                print(f"Mouse down failed: {e}")
    
    def mouse_up(self, x, y):
        if self.controller:
            try:
                self.controller.release(pynput.mouse.Button.left)
            except Exception as e:
                print(f"Mouse up failed: {e}")

# Detection tuning constants (for 640x480; adjust if needed)
# Slow/fallback path (HSV + morphology)
V_MIN = 220        # minimal brightness (Value in HSV) for LED
R_MIN = 150        # minimal red channel intensity
DELTA_RG = 55      # red must exceed G and B by this amount
MIN_AREA = 5       # minimal blob area in pixels
MAX_AREA = 800     # maximal blob area (ignore large bright regions)

# Fast path (no HSV, ROI search). More lenient to avoid misses; then verify.
V_FAST_MIN = 200   # use simple max(b,g,r) as brightness gate
FAST_VERIFY_DELTA = 45  # stricter check in a tiny patch around max
ROI_RADIUS = 100   # pixels around last point to search; None -> full frame

# Precompute morphology kernel once
_KERNEL_ELLIPSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Slow fallback cadence: run heavy HSV path only sometimes if tracking is stable
SLOW_FALLBACK_PERIOD = 10  # frames; set 0 to disable periodic fallback

# Click behavior
CLICK_ON_APPEAR_DEFAULT = True  # click when dot appears (rising edge)
CLICK_COOLDOWN_SEC = 0.3        # minimal time between clicks

# Hold behavior (press while visible)
HOLD_WHILE_VISIBLE_DEFAULT = False
HOLD_DELAY_SEC = 0.08            # how long dot must be visible before press
HOLD_RELEASE_GRACE_SEC = 0.12    # grace to avoid chatter on brief dropouts


def _clip_roi(x0: int, y0: int, r: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x0 - r)
    y1 = max(0, y0 - r)
    x2 = min(w, x0 + r)
    y2 = min(h, y0 + r)
    return x1, y1, x2, y2


def find_red_led(img_bgr: np.ndarray, prev: Optional[Tuple[int, int]] = None, allow_slow: bool = True):
    """Fast red LED detection with ROI search and slow fallback.

    Returns: (cx, cy, area, mask_filtered, mask_hsv, mask_rdom, mask_best)
    - mask_* are provided for UI; in fast-path they may be zeros to save CPU.
    """
    h, w = img_bgr.shape[:2]
    zeros = np.zeros((h, w), dtype=np.uint8)

    # ---------- FAST PATH (no HSV): red-dominance + brightness gate ----------
    # ROI around previous point to reduce work
    if prev is not None:
        roi_x1, roi_y1, roi_x2, roi_y2 = _clip_roi(prev[0], prev[1], ROI_RADIUS, w, h)
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
    roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size != 0:
        b, g, r = cv2.split(roi)
        v_approx = np.maximum(np.maximum(b, g), r)
        # red dominance score
        rdom = r.astype(np.int16) - np.maximum(g, b).astype(np.int16)
        # suppress dark areas
        rdom[v_approx < V_FAST_MIN] = -9999
        # find best location
        _, maxVal, _, maxLoc = cv2.minMaxLoc(rdom.astype(np.float32))
        if maxVal > DELTA_RG:
            cx = roi_x1 + maxLoc[0]
            cy = roi_y1 + maxLoc[1]
            # verify in a tiny patch for stability
            px1, py1, px2, py2 = _clip_roi(cx, cy, 4, w, h)
            patch = img_bgr[py1:py2, px1:px2]
            if patch.size:
                pb, pg, pr = cv2.split(patch)
                good = np.mean((pr > (pg + FAST_VERIFY_DELTA)) & (pr > (pb + FAST_VERIFY_DELTA)))
                if good > 0.5:
                    mask_best = zeros.copy()
                    cv2.circle(mask_best, (cx, cy), 3, 255, -1)
                    return cx, cy, 16, zeros, zeros, zeros, mask_best

    # ---------- SLOW FALLBACK: HSV + morphology + contours ----------
    if not allow_slow:
        # Skip heavy path this frame to keep FPS high
        return None, None, 0, zeros, zeros, zeros, zeros
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red hue wraps around: allow lower S/V as some cameras desaturate bright spots
    lower1 = np.array([0, 70, 100], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 70, 100], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    mask_h1 = cv2.inRange(hsv, lower1, upper1)
    mask_h2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(mask_h1, mask_h2)

    # Red dominance mask in BGR: R significantly greater than G and B
    b, g, r = cv2.split(img_bgr)
    mask_rdom = (r > (g + DELTA_RG)) & (r > (b + DELTA_RG)) & (r > R_MIN)
    mask_rdom = mask_rdom.astype(np.uint8) * 255

    # Brightness gate to suppress non-glowing reds
    v = hsv[:, :, 2]
    mask_bright = (v >= V_MIN).astype(np.uint8) * 255

    # Combine (union) then require brightness
    mask = cv2.bitwise_or(mask_hsv, mask_rdom)
    mask = cv2.bitwise_and(mask, mask_bright)

    # Morphology: small open then close to reduce noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_ELLIPSE_5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_ELLIPSE_5, iterations=1)

    # Area-filtered mask (remove too small/large blobs from visualization)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_filtered = np.zeros_like(mask)
    for i in range(1, num):
        area_i = stats[i, cv2.CC_STAT_AREA]
        if MIN_AREA <= area_i <= MAX_AREA:
            mask_filtered[labels == i] = 255

    # Find contours and select small, bright, red-dominant blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = -1
    best_cnt = None
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a < MIN_AREA or a > MAX_AREA:
            continue
        # Score by mean red channel inside contour and compactness
        x, y, ww, hh = cv2.boundingRect(cnt)
        roi = img_bgr[y:y+hh, x:x+ww]
        roi_mask = np.zeros((hh, ww), dtype=np.uint8)
        cv2.drawContours(roi_mask, [cnt - np.array([[x, y]])], -1, 255, -1)
        if roi.size == 0:
            continue
        r_roi = roi[:, :, 2]
        mean_r = cv2.mean(r_roi, mask=roi_mask)[0]
        perim = cv2.arcLength(cnt, True) + 1e-3
        compact = (a / (perim * perim))  # circle ~maximizes compactness
        score = mean_r + 2000 * compact
        if score > best_score:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                best = (cx, cy, a)
                best_score = score
                best_cnt = cnt

    if best is not None:
        cx, cy, area = best
        # Create best-only mask for display
        mask_best = np.zeros_like(mask)
        if best_cnt is not None:
            cv2.drawContours(mask_best, [best_cnt], -1, 255, thickness=-1)
        return cx, cy, area, mask_filtered, mask_hsv, mask_rdom, mask_best

    # Fallback: pick brightest red-dominant pixel
    rdom = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16))
    rdom[r < R_MIN] = -9999
    rdom[v < V_MIN] = -9999
    _, maxVal, _, maxLoc = cv2.minMaxLoc(rdom.astype(np.float32))
    if maxVal > DELTA_RG:
        mask_best = np.zeros_like(mask)
        mask_best[maxLoc[1], maxLoc[0]] = 255
        return maxLoc[0], maxLoc[1], 1, mask_filtered, mask_hsv, mask_rdom, mask_best

    mask_best = np.zeros_like(mask)
    return None, None, 0, mask_filtered, mask_hsv, mask_rdom, mask_best


def try_set_camera_controls(dev_index: int = 0):
    """Optional: reduce auto-exposure/auto-white-balance to make the laser pop.
    Safe no-op if v4l2-ctl is not available or the camera doesn't support it.
    """
    cmds = [
        ["v4l2-ctl", "-d", f"/dev/video{dev_index}", "-c", "exposure_auto=1"],  # manual
        ["v4l2-ctl", "-d", f"/dev/video{dev_index}", "-c", "exposure_absolute=20"],
        ["v4l2-ctl", "-d", f"/dev/video{dev_index}", "-c", "white_balance_temperature_auto=0"],
        ["v4l2-ctl", "-d", f"/dev/video{dev_index}", "-c", "gain_auto=0"],
        ["v4l2-ctl", "-d", f"/dev/video{dev_index}", "-c", "focus_auto=0"],
    ]
    for cmd in cmds:
        try:
            subprocess.run(cmd, capture_output=True, timeout=0.6)
        except Exception:
            break


if __name__ == '__main__':
    
    cv2.namedWindow("original")
    cv2.namedWindow("ir")
    # HighGUI self-test: show a tiny black frame briefly
    try:
        _probe = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.imshow('original', _probe)
        cv2.imshow('ir', _probe[:,:,0])
        cv2.waitKey(50)
    except Exception as e:
        print(f"HighGUI failed to display test frame: {e}")
    #cv2.namedWindow("result")
    
    cap = video.create_capture(0)
    try:
        opened = getattr(cap, 'isOpened', lambda: True)()
        print(f"Camera opened: {opened}")
    except Exception:
        print("Camera opened: unknown (no isOpened)")

    # Validate camera capture
    if cap is None or not hasattr(cap, 'read'):
        print("Error: capture device not initialized.")
        sys.exit(1)
    
    # Initialize mouse controller
    mouse_controller = CrossPlatformMouse()
    
    # Get screen resolution (for pointer mapping only; do NOT set camera to this)
    width, height = get_screen_resolution()
    print(f"Screen resolution: {width}x{height}")

    # Try a modest camera resolution that most webcams support
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # Prefer MJPG to improve USB webcam FPS
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        # Try a higher FPS; camera may clamp to supported value
        cap.set(cv2.CAP_PROP_FPS, 60)
        # Reduce internal buffer to lower latency if backend supports it
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Optional (Linux only): attempt to tame auto-exposure so the laser stands out
    if sys.platform.startswith('linux'):
        try_set_camera_controls(0)

    # Print negotiated camera params for diagnostics
    try:
        negotiated_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        negotiated_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        negotiated_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_val = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((fourcc_val >> 8*i) & 0xFF) for i in range(4)])
        print(f"Camera negotiated: {negotiated_w}x{negotiated_h} @ {negotiated_fps:.1f} FPS, FOURCC={fourcc_str}")
        # If FPS looks capped (~10fps), try to force a smaller mode
        if negotiated_fps and negotiated_fps <= 15.0:
            print("Trying to switch to 320x240 @ 60 FPS for higher framerate...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            try:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            nfps = cap.get(cv2.CAP_PROP_FPS)
            fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fccs = ''.join([chr((fcc >> 8*i) & 0xFF) for i in range(4)])
            print(f"New mode: {nw}x{nh} @ {nfps:.1f} FPS, FOURCC={fccs}")
    except Exception:
        pass

    # Calibration state: collect two corners pointed by the laser (manual capture)
    calibrated = False
    calib_pts = []  # list of (x, y) in camera image
    last_beep_time = 0.0

    # Smoothing for cursor
    smooth_x, smooth_y = None, None
    alpha = 0.4  # smoothing factor

    mask_mode = 0  # 0: filtered combined, 1: HSV-only, 2: red-dominance, 3: best-only
    prev_pt: Optional[tuple] = None
    fast_only = False
    click_enabled = CLICK_ON_APPEAR_DEFAULT
    prev_visible = False
    last_click_time = 0.0
    hold_enabled = HOLD_WHILE_VISIBLE_DEFAULT
    is_down = False
    visible_since: Optional[float] = None
    last_seen_time: float = 0.0
    cv2.setUseOptimized(True)
    # cv2.setNumThreads(0)  # uncomment to test single-thread perf
    t0 = time.perf_counter()
    frames = 0
    while True:
    
        flag, img = cap.read()
        if not flag or img is None or (hasattr(img, 'size') and img.size == 0):
            # No frame from camera; device may be busy or not accessible
            # Avoid processing to prevent cvtColor errors
            ch = cv2.waitKey(5)
            if ch == 27:
                break
            continue
        try:
            # Run slow fallback only periodically or when we lost the track
            allow_slow = (not fast_only) and ((prev_pt is None) or (SLOW_FALLBACK_PERIOD and (frames % SLOW_FALLBACK_PERIOD == 0)))
            cx, cy, area, mask, mask_hsv, mask_rdom, mask_best = find_red_led(img, prev=prev_pt, allow_slow=allow_slow)
            if area > 0:
                prev_pt = (cx, cy)
            else:
                # if fast path misses often, prev will eventually be None and we will search full frame
                prev_pt = None

            # Draw detection for debug
            if area > 0:
                cv2.circle(img, (cx, cy), 6, (0, 255, 255), 2)

            if not calibrated:
                # Show instruction
                cv2.putText(img, 'Calibration: point 2 corners with red LED', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(img, f'Collected: {len(calib_pts)}/2  (SPACE=capture, R=reset, M=mask)', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw current candidate point
                if area >= MIN_AREA:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), 2)

                # Draw collected points
                for p in calib_pts:
                    cv2.circle(img, p, 8, (0, 200, 0), 2)

                if len(calib_pts) >= 2:
                    # Compute bounds from two corners (any order)
                    xs = [calib_pts[0][0], calib_pts[1][0]]
                    ys = [calib_pts[0][1], calib_pts[1][1]]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    # Ensure non-zero rectangle
                    if x_max - x_min >= 20 and y_max - y_min >= 20:
                        calib_rect = (x_min, y_min, x_max, y_max)
                        calibrated = True
                        play_beep(1500, 100)
                        # Show a quick visual cue
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 200, 0), 2)
            else:
                # Tracking: map LED centroid to screen coordinates within calibrated rect
                x_min, y_min, x_max, y_max = calib_rect
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (50, 120, 50), 1)
                if area > 0:
                    # Normalize within rect and map to screen
                    tx = (cx - x_min) / max(1, (x_max - x_min))
                    ty = (cy - y_min) / max(1, (y_max - y_min))
                    tx = min(max(tx, 0.0), 1.0)
                    ty = min(max(ty, 0.0), 1.0)
                    sx = int(tx * width)
                    sy = int(ty * height)

                    # Smooth motion
                    if smooth_x is None:
                        smooth_x, smooth_y = sx, sy
                    else:
                        smooth_x = int(alpha * smooth_x + (1 - alpha) * sx)
                        smooth_y = int(alpha * smooth_y + (1 - alpha) * sy)

                    mouse_controller.set_cursor_pos(smooth_x, smooth_y)
                    # Visibility inside calibrated rect
                    in_rect = (x_min <= cx <= x_max) and (y_min <= cy <= y_max)
                    now = time.perf_counter()
                    visible_now = bool(in_rect and area > 0)

                    # Hold-to-click (mutually exclusive с click-on-appear)
                    if hold_enabled:
                        if visible_now:
                            if visible_since is None:
                                visible_since = now
                            last_seen_time = now
                            if not is_down and (now - visible_since) >= HOLD_DELAY_SEC:
                                mouse_controller.mouse_down(smooth_x, smooth_y)
                                is_down = True
                        else:
                            visible_since = None
                            # release with small grace to avoid chatter
                            if is_down and (now - last_seen_time) >= HOLD_RELEASE_GRACE_SEC:
                                mouse_controller.mouse_up(smooth_x, smooth_y)
                                is_down = False

                    # Click on rising edge when hold is OFF
                    if (not hold_enabled) and click_enabled and (not prev_visible) and visible_now and (now - last_click_time >= CLICK_COOLDOWN_SEC):
                        mouse_controller.mouse_down(smooth_x, smooth_y)
                        mouse_controller.mouse_up(smooth_x, smooth_y)
                        last_click_time = now
                # Note: clicking disabled by default; can be added on key toggle

            # FPS
            frames += 1
            if frames % 10 == 0:
                t1 = time.perf_counter()
                fps = 10.0 / max(1e-6, (t1 - t0))
                t0 = t1
            else:
                fps = None

            if fps is not None:
                cv2.putText(img, f"FPS: {fps:.1f}", (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Show frames
            cv2.imshow('original', img)
            # Show selected mask
            if mask_mode == 0:
                cv2.imshow('ir', mask)
            elif mask_mode == 1:
                cv2.imshow('ir', mask_hsv)
            elif mask_mode == 2:
                cv2.imshow('ir', mask_rdom)
            else:
                cv2.imshow('ir', mask_best)
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            raise

        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break
        if ch in (ord('q'), ord('Q')):
            break
        if ch in (ord('r'), ord('R')):
            calibrated = False
            calib_pts = []
            smooth_x = smooth_y = None
            prev_pt = None
            visible_since = None
            if is_down:
                mouse_controller.mouse_up(smooth_x if smooth_x is not None else 0, smooth_y if smooth_y is not None else 0)
                is_down = False
            prev_visible = False
        if ch in (ord(' '),):  # SPACE to capture
            if not calibrated and area >= MIN_AREA:
                calib_pts.append((cx, cy))
                calib_pts = calib_pts[-2:]  # keep last two
                play_beep(1200, 50)
                # If two points collected, finalize
                if len(calib_pts) == 2:
                    xs = [calib_pts[0][0], calib_pts[1][0]]
                    ys = [calib_pts[0][1], calib_pts[1][1]]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    if x_max - x_min >= 20 and y_max - y_min >= 20:
                        calib_rect = (x_min, y_min, x_max, y_max)
                        calibrated = True
                        play_beep(1500, 100)
        if ch in (ord('m'), ord('M')):
            mask_mode = (mask_mode + 1) % 4
            label = ['filtered', 'hsv', 'r-dom', 'best'][mask_mode]
            print(f"Mask view: {label}")
        if ch in (ord('f'), ord('F')):
            fast_only = not fast_only
            print(f"Fast-only detection: {'ON' if fast_only else 'OFF'}")
        if ch in (ord('c'), ord('C')):
            click_enabled = not click_enabled
            print(f"Click-on-appear: {'ON' if click_enabled else 'OFF'}")
        if ch in (ord('h'), ord('H')):
            hold_enabled = not hold_enabled
            if hold_enabled:
                # To avoid двойных событий, отключаем одиночные клики
                click_enabled = False
            # On toggle off, release if was held
            if not hold_enabled and is_down:
                mouse_controller.mouse_up(smooth_x if smooth_x is not None else 0, smooth_y if smooth_y is not None else 0)
                is_down = False
            print(f"Hold-while-visible: {'ON' if hold_enabled else 'OFF'}")

        # Update visibility state for edge detection (inside rect when calibrated)
        if 'calib_rect' in locals() and calibrated and (area is not None):
            x_min, y_min, x_max, y_max = calib_rect
            prev_visible = bool((area > 0) and (x_min <= cx <= x_max) and (y_min <= cy <= y_max))
        else:
            prev_visible = (area is not None and area > 0)
    # Properly release camera and close windows
    try:
        cap.release()
    except Exception:
        pass
    try:
        # safety release of mouse if still pressed
        if 'is_down' in locals() and is_down:
            mouse_controller.mouse_up(0, 0)
        cv2.destroyAllWindows()
    except Exception:
        pass
    
