import time
import click
import cv2
import inspireface as isf
import numpy as np


def generate_color(id_str):
    """Generate a bright color based on a string ID safely without overflow."""
    # Hash the string to get a consistent integer ID for coloring
    id_int = abs(hash(id_str))

    max_id = 50
    id_val = id_int % max_id

    # Force hue to stay cleanly between 0 and 179 (OpenCV's max Hue value)
    hue = int((id_val * 180 / max_id) % 180)

    # Ensure Saturation and Value stay strictly between 0 and 255
    saturation = int(200 + (55 * id_val) % 55) % 256
    value = int(200 + (55 * id_val) % 55) % 256

    hsv_color = np.uint8([[[hue, saturation, value]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))


@click.command()
@click.argument("source")
@click.option("--show", is_flag=True, help="Display the video stream.")
@click.option("--out", type=str, default=None, help="Path to save processed video.")
def case_face_tracker_from_video(source, show, out):
    """
    Launch face tracking with dynamic re-identification from video source.
    """
    # 1. Enable FACE_RECOGNITION along with interaction features
    opt = (
        isf.HF_ENABLE_NONE | isf.HF_ENABLE_INTERACTION | isf.HF_ENABLE_FACE_RECOGNITION
    )
    session = isf.InspireFaceSession(
        opt, isf.HF_DETECT_MODE_LIGHT_TRACK, max_detect_num=25, detect_pixel_level=320
    )

    # Configure tracking parameters
    session.set_track_mode_smooth_ratio(0.06)
    session.set_track_mode_num_smooth_cache_frame(15)
    session.set_filter_minimum_face_pixel_size(0)
    session.set_track_model_detect_interval(0)

    # 2. In-memory dictionary to store dynamic identities
    runtime_gallery = {}  # Stores { "Person_1": embedding_vector, ... }
    next_id_to_assign = 1

    # Threshold for matching faces (Lowered slightly to 0.50 to align with standard Cosine Similarity)
    SIMILARITY_THRESHOLD = 0.50

    # Open video source
    try:
        source_index = int(source)
        print(f"Using webcam at index {source_index}.")
        cap = cv2.VideoCapture(source_index)
    except ValueError:
        print(f"Opening video file at {source}.")
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Setup video writer if output path provided
    out_video = None
    if out:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = cv2.VideoWriter(out, fourcc, fps, (frame_width, frame_height))
        print(f"Saving video to: {out}")

    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and track faces
        faces = session.face_detection(frame)

        # Execute pipeline for action detection
        exts = session.face_pipeline(frame, faces, isf.HF_ENABLE_INTERACTION)

        # Draw results
        for idx, face in enumerate(faces):
            # 3. Extract the mathematical feature embedding for the current face
            try:
                current_feature = session.face_feature_extract(frame, face)
            except Exception as e:
                current_feature = None

            assigned_id = None
            highest_score = 0.0

            # 4. FIXED: Compare using dynamic attribute checking or direct native math fallback
            if current_feature is not None:
                for custom_id, saved_feature in runtime_gallery.items():
                    try:
                        if hasattr(session, "face_comparison"):
                            score = session.face_comparison(
                                current_feature, saved_feature
                            )
                        elif hasattr(session, "face_compare"):
                            score = session.face_compare(current_feature, saved_feature)
                        else:
                            raise AttributeError
                    except (AttributeError, Exception):
                        # Math fallback: Compute manual Cosine Similarity over vectors
                        feat1 = np.array(current_feature).flatten()
                        feat2 = np.array(saved_feature).flatten()
                        score = float(
                            np.dot(feat1, feat2)
                            / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                        )

                    if score > SIMILARITY_THRESHOLD and score > highest_score:
                        highest_score = score
                        assigned_id = custom_id

                # 5. If no match was found, register them as a new dynamic person
                if assigned_id is None:
                    assigned_id = f"Person_{next_id_to_assign}"
                    runtime_gallery[assigned_id] = current_feature  # Save to RAM cache
                    next_id_to_assign += 1
                    print(f"New identity cached: {assigned_id}")
            else:
                # Fallback if recognition fails completely on a glitchy frame
                assigned_id = f"Unknown_Track_{face.track_id}"

            x1, y1, x2, y2 = face.location
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            angle = face.roll

            # Create rotated bounding box
            rect = ((center[0], center[1]), (size[0], size[1]), angle)
            box = cv2.boxPoints(rect).astype(int)

            # 6. Use the newly assigned dynamic ID string for coloring and display
            color = generate_color(assigned_id)

            # Draw bounding box
            cv2.drawContours(frame, [box], 0, color, 4)

            # Draw landmarks
            lmk = session.get_face_dense_landmark(face)
            for x, y in lmk.astype(int):
                cv2.circle(frame, (x, y), 0, color, 4)

            # Detect and display actions
            if idx < len(exts):
                ext = exts[idx]
                actions = []
                if ext.action_normal:
                    actions.append("Normal")
                if ext.action_jaw_open:
                    actions.append("Jaw Open")
                if ext.action_shake:
                    actions.append("Shake")
                if ext.action_blink:
                    actions.append("Blink")
                if ext.action_head_raise:
                    actions.append("Head Raise")

                # Display actions text
                if actions:
                    action_text = ", ".join(actions)
                    cv2.putText(
                        frame,
                        action_text,
                        (x1, y1 - 50),  # Positioned above tracking ID
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # 7. Display our persistent Re-ID label instead of the volatile track ID
            text = f"ID: {assigned_id} (Internal: {face.track_id})"
            cv2.putText(
                frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Show frame
        if show:
            cv2.imshow("Face Tracker", frame)

            # Fix 1: Explicitly check if the user closed the window using the mouse 'X' button
            if cv2.getWindowProperty("Face Tracker", cv2.WND_PROP_VISIBLE) < 1:
                print("[Window Closed] Shutting down...")
                break

            # Fix 2: Explicitly capture 'q' or 'Q' or Esc (27) keys while the window is focused
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q") or key == 27:
                print("[Key pressed] Shutting down...")
                break

        # Write frame to output video
        if out_video:
            out_video.write(frame)

    # Cleanup
    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()
    print("Released all resources and closed windows.")


if __name__ == "__main__":
    case_face_tracker_from_video()
