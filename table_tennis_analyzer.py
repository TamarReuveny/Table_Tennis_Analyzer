import cv2
from ultralytics import YOLO

START_FRAME = 0
END_FRAME = 200  # Set to 0 to process the full video


def analyze_table_tennis_video(input_path="input.mp4", output_path="output_with_detections.mp4"):
    print("Loading model...")
    model = YOLO("yolov8n-pose.pt")
    
    print("Opening video...")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle END_FRAME = 0 → process whole video
    end_frame = END_FRAME if END_FRAME > 0 else total_frames

    # Set starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    # Define VideoWriter with same properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing frames {START_FRAME} to {end_frame}...")

    current_frame = START_FRAME
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("End of video or read error.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        current_frame += 1

    cap.release()
    out.release()
    print(f"Done! Output saved to {output_path}")

if __name__ == "__main__":
    analyze_table_tennis_video()