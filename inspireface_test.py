import time
import click
import cv2
import inspireface as isf
import numpy as np


def generate_color(id):
    """Generate a bright color based on the given integer ID."""
    if id < 0:
        return (128, 128, 128)  # Gray for invalid ID

    max_id = 50
    id = id % max_id
    hue = int((id * 360 / max_id) % 360)
    saturation = 200 + (55 * id) % 55
    value = 200 + (55 * id) % 55

    hsv_color = np.uint8([[[hue, saturation, value]]])
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))


@click.command()
@click.argument("source")
@click.option("--show", is_flag=True, help="Display the video stream.")
@click.option("--out", type=str, default=None, help="Path to save processed video.")
def case_face_tracker_from_video(source, show, out):
    """
    Launch face tracking from video source.
    Args:
        source: Webcam index (0, 1, ...) or path to video file
        show: Display video window if set
        out: Output video file path if provided
    """
    # Enable interaction features for action detection
    opt = isf.HF_ENABLE_NONE | isf.HF_ENABLE_INTERACTION
    session = isf.InspireFaceSession(
        opt, isf.HF_DETECT_MODE_LIGHT_TRACK, max_detect_num=25, detect_pixel_level=320
    )

    # Configure tracking parameters
    session.set_track_mode_smooth_ratio(0.06)
    session.set_track_mode_num_smooth_cache_frame(15)
    session.set_filter_minimum_face_pixel_size(0)
    session.set_track_model_detect_interval(0)

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
            x1, y1, x2, y2 = face.location
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = (x2 - x1, y2 - y1)
            angle = face.roll

            # Create rotated bounding box
            rect = ((center[0], center[1]), (size[0], size[1]), angle)
            box = cv2.boxPoints(rect).astype(int)

            # Get unique color for this track ID
            color = generate_color(face.track_id)

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
                        (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

            # Display track ID
            text = f"ID: {face.track_id}, Count: {face.track_count}"
            cv2.putText(
                frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        # Show frame
        if show:
            cv2.imshow("Face Tracker", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
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
