import cv2
import numpy as np

# 21 Hand Connections (MediaPipe standard skeleton)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (0, 17),                                  # Wrist -> Pinky MCP (outer palm edge)
    (0, 9), (0, 13),                          # Wrist -> Middle/Ring MCP (complete palm arch)
]


def draw_hand_skeleton(frame, px_points, hand_color):
    """Draws 21-point neon skeleton with glowing nodes."""
    for p1_i, p2_i in HAND_CONNECTIONS:
        cv2.line(frame, px_points[p1_i], px_points[p2_i], hand_color, 2, cv2.LINE_AA)
    for pt in px_points:
        cv2.circle(frame, pt, 4, hand_color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_gesture_badge(frame, hand_info, occupied_badges):
    """Renders sleek gesture tag and handedness badge above the tracked hand."""
    if len(hand_info) >= 8:
        px_points, handedness_str, gesture_str, g_conf, (hx_min, hy_min, hx_max, hy_max), hand_color, finger_count, is_help = hand_info
    else:
        px_points, handedness_str, gesture_str, g_conf, (hx_min, hy_min, hx_max, hy_max), hand_color = hand_info[:6]
        finger_count = 0
        is_help = (gesture_str == "Help")

    if is_help:
        h_tag = f"[HELP!] {handedness_str} ({int(g_conf * 100)}%)"
        tag_color = (255, 255, 255)
        bg_color = (20, 20, 190)  # Bright Emergency Crimson Red
        border_color = (0, 0, 255)
    else:
        h_tag = f"{handedness_str} | {gesture_str} ({finger_count}F)"
        tag_color = hand_color
        bg_color = (35, 35, 35)
        border_color = hand_color

    (htw, hth), _ = cv2.getTextSize(h_tag, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
    badge_x1 = max(10, hx_min)
    badge_y1 = max(hy_min - hth - 12, 60)
    badge_x2 = badge_x1 + htw + 16
    badge_y2 = badge_y1 + hth + 10

    for ox1, oy1, ox2, oy2 in occupied_badges:
        if not (badge_x2 < ox1 or badge_x1 > ox2 or badge_y2 < oy1 or badge_y1 > oy2):
            badge_y1 = oy2 + 6
            badge_y2 = badge_y1 + hth + 10
    occupied_badges.append((badge_x1, badge_y1, badge_x2, badge_y2))

    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)
    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), border_color, 1)
    cv2.putText(frame, h_tag, (badge_x1 + 8, badge_y1 + hth + 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, tag_color, 1, cv2.LINE_AA)


def draw_hud(frame, num_hands, active_gestures, fps, total_fingers=0, help_active=False, hand_counts_summary=""):
    """
    Renders status banner with:
    - FPS and Hands Tracked
    - Real-time Finger Count (e.g. 10 Fingers / 10 Count)
    - Active gestures
    - Emergency Help Signal distress alert banner
    """
    h, w = frame.shape[:2]
    # Fast in-place alpha blend for HUD bar (avoids full frame.copy())
    frame[:54, :] = (frame[:54, :] * 0.25 + np.array([20, 20, 20], dtype=np.float32) * 0.75).clip(0, 255).astype(np.uint8)

    fps_color = (50, 230, 50) if fps >= 25 else ((0, 215, 255) if fps >= 15 else (50, 50, 240))
    cv2.putText(frame, f"FPS: {fps:.1f}", (14, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, fps_color, 1, cv2.LINE_AA)

    status_txt = f"Hands: {num_hands}"
    cv2.putText(frame, status_txt, (120, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, (50, 220, 220), 1, cv2.LINE_AA)

    # Prominent Finger Counter in Top Bar
    if num_hands > 0:
        if total_fingers == 10:
            count_text = "FINGERS: 10 / 10 (MAX COUNT)"
            count_color = (0, 255, 255)  # Glowing Gold / Yellow
        else:
            count_text = f"FINGERS: {total_fingers} / 10"
            count_color = (50, 240, 50)
        cv2.putText(frame, count_text, (220, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, count_color, 1, cv2.LINE_AA)

    if active_gestures:
        summary = " | ".join(active_gestures)
        cv2.putText(frame, f"Active: {summary}", (14, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Active: Waiting for gestures...", (14, 46),
                    cv2.FONT_HERSHEY_DUPLEX, 0.46, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.putText(frame, "[S] Save  [Q] Quit", (w - 180, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # Floating Finger Counter Widget (Top-Right)
    # -------------------------------------------------------------
    if num_hands > 0:
        card_w = 260
        card_h = 48
        card_x1 = w - card_w - 14
        card_y1 = 62
        card_x2 = card_x1 + card_w
        card_y2 = card_y1 + card_h

        # Fast in-place alpha blend for finger card widget
        roi = frame[card_y1:card_y2, card_x1:card_x2]
        blended = (roi * 0.18 + np.array([25, 25, 25], dtype=np.float32) * 0.82).clip(0, 255).astype(np.uint8)
        frame[card_y1:card_y2, card_x1:card_x2] = blended

        border_c = (0, 215, 255) if total_fingers == 10 else (80, 200, 120)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), border_c, 1)

        if total_fingers == 10:
            banner_title = "10 FINGERS (10 COUNT)"
            t_color = (0, 230, 255)
        else:
            banner_title = f"FINGER COUNT: {total_fingers}"
            t_color = (255, 255, 255)

        cv2.putText(frame, banner_title, (card_x1 + 10, card_y1 + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.54, t_color, 1, cv2.LINE_AA)

        sub_txt = hand_counts_summary if hand_counts_summary else f"Total Extended: {total_fingers}"
        cv2.putText(frame, sub_txt, (card_x1 + 12, card_y1 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # EMERGENCY HELP SIGNAL ALERT BANNER (Only for authentic Help)
    # -------------------------------------------------------------
    if help_active:
        banner_h = 44
        by1 = h - banner_h - 18
        by2 = by1 + banner_h
        bx1 = 40
        bx2 = w - 40

        # Pulsing deep red background (fast in-place blend)
        roi_b = frame[by1:by2, bx1:bx2]
        blended_b = (roi_b * 0.12 + np.array([0, 0, 180], dtype=np.float32) * 0.88).clip(0, 255).astype(np.uint8)
        frame[by1:by2, bx1:bx2] = blended_b
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        help_alert_text = "[!] EMERGENCY DISTRESS SIGNAL: HELP DETECTED! [!]"
        (atw, ath), _ = cv2.getTextSize(help_alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.70, 2)
        text_x = bx1 + (bx2 - bx1 - atw) // 2
        text_y = by1 + (banner_h + ath) // 2

        cv2.putText(frame, help_alert_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
