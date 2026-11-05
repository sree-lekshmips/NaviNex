import cv2
import numpy as np
import math

def generate_test_video(filename="test_sample.mp4", duration_sec=10, fps=25, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    for f in range(total_frames):
        t = f / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Base perspective indoor hallway
        # Ceiling: dark slate
        frame[:int(height * 0.40), :] = [60, 55, 50]
        # Floor: light textured gray (BGR 190, 195, 200)
        frame[int(height * 0.40):, :] = [180, 185, 190]

        # Draw perspective floor lines
        vp_x, vp_y = width // 2, int(height * 0.40)
        for i in range(-5, 6):
            start_x = vp_x + i * 80
            cv2.line(frame, (vp_x, vp_y), (start_x + i * 200, height), (150, 155, 160), 2)

        # Draw left wall
        left_pts = np.array([[0, 0], [int(width * 0.20), vp_y], [int(width * 0.20), height], [0, height]], np.int32)
        cv2.fillPoly(frame, [left_pts], (120, 115, 110))
        # Draw right wall
        right_pts = np.array([[width, 0], [int(width * 0.80), vp_y], [int(width * 0.80), height], [width, height]], np.int32)
        cv2.fillPoly(frame, [right_pts], (120, 115, 110))

        # Scene Sequencing:
        if t < 2.0:
            # 1. Clear Hallway (0 - 2s)
            cv2.putText(frame, "SCENE 1: Clear Hallway", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
            cv2.putText(frame, "Expected: 'Path is clear, continue forward'", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        elif t < 5.0:
            # 2. Person ahead in corridor (2 - 5s)
            progress = (t - 2.0) / 3.0
            person_scale = 0.7 + progress * 0.5
            pw = int(140 * person_scale)
            ph = int(320 * person_scale)
            px = int(width * 0.50 + math.sin(t * 3) * 30 - pw // 2)
            py = int(height * 0.75 - ph)

            # Head
            cv2.circle(frame, (px + pw // 2, py + int(ph * 0.15)), int(pw * 0.22), (200, 180, 160), -1)
            # Torso (Jacket)
            cv2.rectangle(frame, (px + int(pw * 0.15), py + int(ph * 0.30)), (px + int(pw * 0.85), py + int(ph * 0.70)), (40, 50, 160), -1)
            # Legs
            cv2.rectangle(frame, (px + int(pw * 0.20), py + int(ph * 0.70)), (px + int(pw * 0.45), py + ph), (30, 30, 40), -1)
            cv2.rectangle(frame, (px + int(pw * 0.55), py + int(ph * 0.70)), (px + int(pw * 0.80), py + ph), (30, 30, 40), -1)

            cv2.putText(frame, "SCENE 2: Person in Forward Path", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            cv2.putText(frame, "Expected: 'Person detected ahead. Move left/right.'", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        elif t < 7.0:
            # 3. Path opens up again (5 - 7s)
            cv2.putText(frame, "SCENE 3: Path Clear Again", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
            cv2.putText(frame, "Expected: 'Path is clear, continue forward' (Once)", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        else:
            # 4. Approaching Dead-end Wall (7 - 10s)
            wall_progress = (t - 7.0) / 3.0
            wall_y = int(vp_y + wall_progress * 160)
            cv2.rectangle(frame, (int(width * 0.15), wall_y), (int(width * 0.85), height), (85, 80, 75), -1)
            cv2.line(frame, (int(width * 0.15), wall_y), (int(width * 0.85), wall_y), (50, 45, 40), 3)

            cv2.putText(frame, "SCENE 4: Wall Blocking Forward Path", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "Expected: 'Wall detected ahead' / 'Warning. Path blocked'", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        out.write(frame)

    out.release()
    print(f"Generated test video: {filename}")

if __name__ == "__main__":
    generate_test_video("test_sample.mp4")
