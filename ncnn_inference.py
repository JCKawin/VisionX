import cv2

def reduce_fps(input_path, output_path, target_fps=2):
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Could not open video file")

    # Get original video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame skipping factor
    skip_factor = int(round(orig_fps / target_fps))
    if skip_factor < 1:
        skip_factor = 1

    # Define video writer (same size, same codec, new FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Write only frames that fit the 10 FPS pattern
        if frame_index % skip_factor == 0:
            out.write(frame)

        frame_index += 1

    cap.release()
    out.release()
    print("Finished. Saved to:", output_path)


# Example usage
reduce_fps("test.mp4", "output_10fps.mp4")
