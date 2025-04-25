import cv2
import csv
from ultralytics import YOLO

START_FRAME = 0
END_FRAME = 200  # Set to 0 to process the full video


def analyze_table_tennis_video(input_path="input.mp4", output_path="output_with_detections.mp4", csv_output="player_positions.csv"):
    model = YOLO("yolo11n-pose.pt")
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
        
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = END_FRAME if END_FRAME > 0 else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Prepare CSV output
    csv_data = []
    csv_data.append(["frame", "player1_position_x", "player1_position_y", "player2_position_x", "player2_position_y"])

    current_frame = START_FRAME
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        player_centers = []

        # Extract bounding boxes
        for box in results[0].boxes.xyxy.cpu().numpy():  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            player_centers.append((center_x, center_y))

        # If we have at least 2 people, get the two with left-most and right-most x values
        if len(player_centers) >= 2:
            # Sort by x value (left to right)
            player_centers.sort(key=lambda p: p[0])
            player1 = player_centers[0]
            player2 = player_centers[1]
            csv_data.append([current_frame + 1, player1[0], player1[1], player2[0], player2[1]])

        current_frame += 1

    cap.release()
    out.release()

    # Save CSV
    with open(csv_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"Done! Video saved to {output_path}")
    print(f"CSV saved to {csv_output}")

if __name__ == "__main__":
    analyze_table_tennis_video()