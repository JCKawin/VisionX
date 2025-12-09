import cv2
import time

def live_camera_to_low_fps(camera_index=0, target_fps=2, save_output=False, output_path="output_2fps.mp4"):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise IOError("Could not open camera")

    # Get original camera properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup optional video writer
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    # Time between frames for the reduced FPS
    frame_interval = 1.0 / target_fps
    last_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # Only process frames at the target FPS interval
        if (current_time - last_frame_time) >= frame_interval:
            last_frame_time = current_time

            # Display the reduced-FPS stream
            cv2.imshow("2 FPS Stream", frame)

            # Optionally save it
            if save_output:
                out.write(frame)

        # Exit if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()

# Example usage:
live_camera_to_low_fps(target_fps=2, save_output=False)