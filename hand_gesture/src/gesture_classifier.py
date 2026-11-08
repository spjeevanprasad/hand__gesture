import time
import math
import numpy as np

# Global temporal memory for dynamic distress signal tracking & latching
_HELP_CACHE = {}


def evaluate_hand_gesture(landmarks_21, handedness_str: str):
    """
    Classifies gestures using 21 3D hand landmarks:
    - Signal for Help (Distress and Domestic Violence Hand Signal):
      * Step 1: Palm facing outwards, thumb placed on the palm (4 fingers up)
      * Step 2: "2. Trap thumb" - 4 fingers close down OVER the thumb, trapping it inside!
        (Robust latching prevents the signal from flickering or dropping to 'Fist')
    - Finger Counting (0 to 5 fingers per hand, up to 10 fingers for both hands)
    - Full gesture vocabulary:
      * Palm / Five (5), Four (4), Three (3), Peace / Two (2), Point / One (1), Fist (0)
      * Help (Emergency Distress Signal - Trap Thumb)
      * Like (Thumbs Up), Dislike (Thumbs Down)
      * Call Me, Rock On, OK Sign, Pinch
    
    Returns:
      (gesture_str, conf, finger_count, is_help)
    """
    now = time.time()
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_21])

    wrist = pts[0]
    thumb_tip, thumb_ip, thumb_mcp = pts[4], pts[3], pts[2]
    index_tip, index_pip, index_mcp = pts[8], pts[6], pts[5]
    middle_tip, middle_pip, middle_mcp = pts[12], pts[10], pts[9]
    ring_tip, ring_pip, ring_mcp = pts[16], pts[14], pts[13]
    pinky_tip, pinky_pip, pinky_mcp = pts[20], pts[18], pts[17]

    # Hand scale normalization (wrist to middle MCP)
    hand_size = max(0.01, float(np.linalg.norm(middle_mcp[:2] - wrist[:2])))

    # Normalized extension length from MCP to Tip relative to hand size
    index_ext = float(np.linalg.norm(index_tip[:2] - index_mcp[:2])) / hand_size
    middle_ext = float(np.linalg.norm(middle_tip[:2] - middle_mcp[:2])) / hand_size
    ring_ext = float(np.linalg.norm(ring_tip[:2] - ring_mcp[:2])) / hand_size
    pinky_ext = float(np.linalg.norm(pinky_tip[:2] - pinky_mcp[:2])) / hand_size

    # A finger is extended only if tip is further from MCP than PIP AND total extension >= 0.48 * hand_size
    index_up = (np.linalg.norm(index_tip[:2] - index_mcp[:2]) > 1.15 * np.linalg.norm(index_pip[:2] - index_mcp[:2])) and \
               (np.linalg.norm(index_tip[:2] - wrist[:2]) > np.linalg.norm(index_pip[:2] - wrist[:2])) and \
               (index_ext > 0.48)

    middle_up = (np.linalg.norm(middle_tip[:2] - middle_mcp[:2]) > 1.15 * np.linalg.norm(middle_pip[:2] - middle_mcp[:2])) and \
                (np.linalg.norm(middle_tip[:2] - wrist[:2]) > np.linalg.norm(middle_pip[:2] - wrist[:2])) and \
                (middle_ext > 0.48)

    ring_up = (np.linalg.norm(ring_tip[:2] - ring_mcp[:2]) > 1.15 * np.linalg.norm(ring_pip[:2] - ring_mcp[:2])) and \
              (np.linalg.norm(ring_tip[:2] - wrist[:2]) > np.linalg.norm(ring_pip[:2] - wrist[:2])) and \
              (ring_ext > 0.48)

    pinky_up = (np.linalg.norm(pinky_tip[:2] - pinky_mcp[:2]) > 1.15 * np.linalg.norm(pinky_pip[:2] - pinky_mcp[:2])) and \
               (np.linalg.norm(pinky_tip[:2] - wrist[:2]) > np.linalg.norm(pinky_pip[:2] - wrist[:2])) and \
               (pinky_ext > 0.48)

    fingers_up_count = sum([index_up, middle_up, ring_up, pinky_up])

    # Thumb spatial relationships
    thumb_pinky_dist = float(np.linalg.norm(thumb_tip[:2] - pinky_mcp[:2])) / hand_size
    thumb_index_dist = float(np.linalg.norm(thumb_tip[:2] - index_mcp[:2])) / hand_size
    thumb_wrist_dist = float(np.linalg.norm(thumb_tip[:2] - wrist[:2])) / hand_size
    thumb_ip_wrist_dist = float(np.linalg.norm(thumb_ip[:2] - wrist[:2])) / hand_size

    thumb_upward = (thumb_tip[1] < thumb_mcp[1] - 0.12 * hand_size)
    thumb_downward = (thumb_tip[1] > thumb_mcp[1] + 0.08 * hand_size)

    # Pure Thumbs Up gesture: thumb pointing high up, all 4 fingers folded
    is_pure_thumbs_up = (thumb_upward and fingers_up_count == 0 and thumb_index_dist > 0.38 and thumb_pinky_dist > 0.58)

    # Thumb extended (sticking out to the side or open palm)
    thumb_up = is_pure_thumbs_up or \
               (thumb_pinky_dist > 0.62 and thumb_index_dist > 0.38 and thumb_wrist_dist > thumb_ip_wrist_dist + 0.02) or \
               (thumb_index_dist > 0.48)
    thumb_extended = thumb_up

    # Pinch distance between Thumb Tip and Index Tip
    pinch_dist = float(np.linalg.norm(thumb_tip[:2] - index_tip[:2])) / hand_size

    # -------------------------------------------------------------
    # 🚨 Distress Signal for Help: Step 1 & Step 2 ("2. Trap thumb")
    # -------------------------------------------------------------
    # Step 1: Palm facing outwards, thumb tucked onto palm (4 fingers extended straight up)
    is_step1 = (fingers_up_count >= 3 and not thumb_up and thumb_index_dist < 0.45 and thumb_pinky_dist < 0.58)

    state = _HELP_CACHE.setdefault(handedness_str, {
        'step1_time': 0.0,
        'latched': False,
        'trap_frames': 0
    })

    if is_step1:
        state['step1_time'] = now
    else:
        # If user explicitly opens hand or extends thumb outward, clear latch
        if thumb_up or fingers_up_count >= 3:
            state['step1_time'] = 0.0
            state['latched'] = False
            state['trap_frames'] = 0

    four_fingers_down = (fingers_up_count == 0)
    recent_step1 = (now - state['step1_time'] < 4.0) if state['step1_time'] > 0 else False

    # "2. Trap thumb" (Image 4):
    # - Thumb MUST NOT be sticking out (not thumb_up, thumb_index_dist < 0.46)
    # - Thumb is trapped inside the palm under the curled fingers
    thumb_is_trapped = (not thumb_up and thumb_index_dist < 0.46 and thumb_pinky_dist < 0.55)

    is_static_trap_thumb = False
    if four_fingers_down and thumb_is_trapped:
        # Check if thumb tip is resting inside the palm cavity
        if thumb_tip[2] >= index_pip[2] - 0.02 and thumb_tip[1] > min(index_mcp[1], middle_mcp[1]) - 0.05 * hand_size:
            state['trap_frames'] += 1
            if state['trap_frames'] >= 2:
                is_static_trap_thumb = True
        else:
            state['trap_frames'] = 0

    # Latch Help: Once triggered, it remains solidly locked as long as the fist is held!
    is_trap_thumb = False
    if four_fingers_down and thumb_is_trapped:
        if recent_step1 or is_static_trap_thumb or state['latched']:
            is_trap_thumb = True
            state['latched'] = True  # Hold lock securely
    else:
        if fingers_up_count > 0 or thumb_up:
            state['latched'] = False

    # Exact count of extended fingers on this hand (0 to 5)
    finger_count = int(sum([thumb_up, index_up, middle_up, ring_up, pinky_up]))

    # 1. Trigger "Help" strictly for "2. Trap thumb"
    if is_trap_thumb:
        return "Help", 0.98, 0, True

    # 2. Trigger Step 1: "Help Signal (Tuck Thumb)"
    if is_step1:
        return "Help Signal (Tuck Thumb)", 0.95, finger_count, False

    # -------------------------------------------------------------
    # Standard Hand Gestures & Finger Counts
    # -------------------------------------------------------------
    # 3. Like / Thumbs Up (Thumb extended high up, 4 fingers folded)
    if is_pure_thumbs_up:
        return "Like (Thumbs Up)", 0.95, finger_count, False

    # 4. Dislike / Thumbs Down (Thumb pointing down, 4 fingers folded)
    if thumb_downward and fingers_up_count == 0 and thumb_index_dist > 0.40:
        return "Dislike (Thumbs Down)", 0.95, finger_count, False

    # 5. OK Sign (Thumb and Index tips touching, other 3 fingers extended)
    if pinch_dist < 0.28 and middle_up and ring_up and pinky_up:
        return "OK", 0.92, finger_count, False

    # 6. Pinch Gesture (Thumb and Index close together with index active)
    if pinch_dist < 0.30 and index_up and not (middle_up and ring_up and pinky_up):
        return "Pinch", 0.88, finger_count, False

    # 7. Call Gesture (Thumb and Pinky extended, middle 3 folded)
    if thumb_extended and pinky_up and not index_up and not middle_up and not ring_up:
        return "Call", 0.93, finger_count, False

    # 8. Rock / Horns (Index and Pinky up, Middle and Ring folded)
    if index_up and pinky_up and not middle_up and not ring_up:
        return "Rock", 0.94, finger_count, False

    # 9. Closed Fist (All fingers folded tight, thumb on outside)
    if fingers_up_count == 0 and not thumb_extended:
        return "Fist", 0.92, 0, False

    # 10. Point / One (Index finger only extended)
    if index_up and (not middle_up or middle_ext < 0.65 * index_ext) and not ring_up and not pinky_up:
        return "Point / One", 0.95, finger_count, False

    # 11. Peace / Victory / Two (BOTH Index AND Middle extended)
    if index_up and middle_up and (middle_ext >= 0.70 * index_ext) and not ring_up and not pinky_up:
        avg_finger_dy = ((index_tip[1] - index_mcp[1]) + (middle_tip[1] - middle_mcp[1])) / 2.0
        if avg_finger_dy > 0.03:
            return "Peace Down (Two Down)", 0.95, finger_count, False
        else:
            return "Peace Up (Two Up)", 0.95, finger_count, False

    # 12. Three (Index, Middle, Ring extended)
    if index_up and middle_up and ring_up and not pinky_up:
        return "Three", 0.91, finger_count, False

    # 13. Four (Index, Middle, Ring, Pinky extended, thumb folded)
    if fingers_up_count == 4 and not thumb_extended:
        return "Four", 0.92, finger_count, False

    # 14. Full Open Palm / Five (All 5 extended)
    if fingers_up_count == 4 and thumb_extended:
        return "Palm (Five)", 0.96, 5, False

    # Fallback with finger count
    return f"Hand ({finger_count}F)", 0.70, finger_count, False
