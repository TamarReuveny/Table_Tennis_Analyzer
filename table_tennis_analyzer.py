# table_tennis_analyzer.py
import cv2
from ultralytics import YOLO
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import math

# --- Constants for Video Processing ---
START_FRAME = 0
END_FRAME = 1000 # Set to 0 to process all frames until the end
INPUT_VIDEO = 'input.mp4'
OUTPUT_VIDEO = 'output_with_detections.mp4'
OUTPUT_CSV = 'player_positions.csv'
PLAYER_Y_THRESHOLD_FACTOR = 0.7 
MIN_X_SEPARATION_FACTOR = 0.1 

# --- Model Selection ---
MODEL_NAME = 'yolov8n-pose.pt'

# --- Constants for Heatmap ---
VIDEO_PATH_FOR_HEATMAP = INPUT_VIDEO
CSV_PATH_FOR_HEATMAP = OUTPUT_CSV
OUTPUT_HEATMAP_PATH = "player_position_heatmap.png"
# --- Heatmap parameters ---
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Heatmap Generation Function ---
def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """
    Generates a heatmap overlay on a video frame using player position data.
    """
    print(f"\n--- Heatmap Generation ---")
    print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}")
    print(f"Params: bins={bins}, alpha={alpha}, cmap='{cmap_name}', interp='{interpolation}', vmax_pctl={vmax_percentile}")

    # --- 1. Read and Process CSV Data ---
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
             print(f"Error: CSV file '{csv_path}' not found or is empty. Cannot generate heatmap.")
             return
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV file '{csv_path}' contains no data rows. Cannot generate heatmap.")
            return

        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        for col in pos_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                print(f"Error: Column '{col}' not found in CSV '{csv_path}'.")
                return

        df_clean = df.dropna(subset=pos_cols)
        all_x = pd.concat([df_clean['player1_position_x'], df_clean['player2_position_x']], ignore_index=True)
        all_y = pd.concat([df_clean['player1_position_y'], df_clean['player2_position_y']], ignore_index=True)

        if all_x.empty or all_y.empty:
            print(f"Error: No valid numeric coordinates found after cleaning CSV '{csv_path}'. Cannot generate heatmap.")
            return
        print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'. Did video processing run successfully?")
        return
    except pd.errors.EmptyDataError:
         print(f"Error: CSV file '{csv_path}' is empty. Cannot generate heatmap.")
         return
    except Exception as e:
        print(f"Error reading/processing CSV '{csv_path}': {e}")
        return

    # --- 2. Get Background Frame ---
    if not os.path.exists(video_path):
        print(f"Error: Video not found '{video_path}'")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame from '{video_path}'")
        cap.release()
        return
    frame_height, frame_width, _ = frame.shape
    background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()
    print(f"Video dimensions (W x H): {frame_width} x {frame_height}")

    # --- 3. Create 2D Histogram Data ---
    try:
        heatmap_data, xedges, yedges = np.histogram2d(
            all_x, all_y,
            bins=bins,
            range=[[0, frame_width], [0, frame_height]]
        )
        heatmap_data = heatmap_data.T # Transpose for imshow compatibility
        print(f"Histogram shape: {heatmap_data.shape}, Max value: {heatmap_data.max():.2f}")
        if heatmap_data.max() == 0:
            print("Warning: Histogram data is all zeros. Heatmap will be blank.")
    except Exception as e:
        print(f"Error during np.histogram2d calculation: {e}")
        return

    # --- 4. Calculate Vmax based on Percentile ---
    non_zero_data = heatmap_data[heatmap_data > 0]
    calculated_vmax = 1.0
    if non_zero_data.size > 0:
        calculated_vmax = np.percentile(non_zero_data, vmax_percentile)
        calculated_vmax = max(calculated_vmax, 1.0) # Ensure vmax is at least 1
        print(f"Using {vmax_percentile}th percentile vmax for color scale: {calculated_vmax:.2f}")
    else:
        print("Warning: No non-zero density data found for vmax calculation. Setting vmax=1.0")

    # --- 5. Modify Colormap for Transparent Zero ---
    try:
        current_cmap = plt.get_cmap(cmap_name).copy()
        cmap_list = current_cmap(np.linspace(0, 1, current_cmap.N))
        cmap_list[0, 3] = 0.0 # Set alpha of the lowest value (0) to transparent
        transparent_cmap = mcolors.ListedColormap(cmap_list)
    except Exception as e:
        print(f"Error modifying colormap '{cmap_name}': {e}")
        transparent_cmap = plt.get_cmap(cmap_name) # Fallback

    # --- 6. Generate Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(background_image, extent=[0, frame_width, frame_height, 0], aspect='auto')

    if heatmap_data.max() > 0:
        heatmap_plot = ax.imshow(
            heatmap_data,
            cmap=transparent_cmap,
            alpha=alpha,
            extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],
            origin='upper',
            interpolation=interpolation,
            aspect='auto',
            vmin=0,
            vmax=calculated_vmax
        )
        # Determine if extend='max' is needed for colorbar
        extend_direction = 'max' if calculated_vmax < heatmap_data.max() else 'neither'
        cbar = fig.colorbar(heatmap_plot, ax=ax, extend=extend_direction)
        cbar.set_label('Density')
    else:
        print("Skipping heatmap overlay as calculated data is empty or all zeros.")

    ax.set_title('Filtered Player Position Heatmap')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # --- 7. Save ---
    try:
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"Filtered heatmap image saved successfully to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving heatmap image: {e}")
    plt.close(fig)


# --- Helper Function for Player Selection (with Debug Prints) ---
# --- Helper Function for Player Selection (Revised Logic) ---
def select_players(all_detected_persons, frame_width, min_x_separation_factor):
    """
    Selects the two players based on detection count and relative positions.
    Prioritizes 3 detections (assumes umpire + 2 players) and 2 detections.

    Args:
        all_detected_persons (list): List of dicts with 'center_x', 'center_y'.
        frame_width (int): Width of the video frame.
        min_x_separation_factor (float): Min horizontal distance factor.

    Returns:
        tuple: (player1_pos, player2_pos) - dicts with 'x', 'y' or empty strings.
    """
    player1_pos = {'x': '', 'y': ''}
    player2_pos = {'x': '', 'y': ''}
    selected_players = []
    num_detected = len(all_detected_persons)
    min_x_distance = frame_width * min_x_separation_factor

    if num_detected == 3:
        # Assume 2 players + 1 umpire/other. Select the two lowest on screen.
        all_detected_persons.sort(key=lambda p: p['center_y'], reverse=True) # Sort Y descending (lowest first)
        p_low1 = all_detected_persons[0]
        p_low2 = all_detected_persons[1]
        # Check their X separation
        x_distance = abs(p_low1['center_x'] - p_low2['center_x'])
        if x_distance >= min_x_distance:
            selected_players = [p_low1, p_low2]

    elif num_detected == 2:
        # Assume it's the two players. Check their X separation.
        p1 = all_detected_persons[0]
        p2 = all_detected_persons[1]
        x_distance = abs(p1['center_x'] - p2['center_x'])
        if x_distance >= min_x_distance:
            selected_players = all_detected_persons
    # Assign player positions only if exactly two were selected
    if len(selected_players) == 2:
        selected_players.sort(key=lambda p: p['center_x']) # Sort by X for P1/P2 consistency
        player1_pos['x'] = int(selected_players[0]['center_x'])
        player1_pos['y'] = int(selected_players[0]['center_y'])
        player2_pos['x'] = int(selected_players[1]['center_x'])
        player2_pos['y'] = int(selected_players[1]['center_y'])

    return player1_pos, player2_pos

# --- Main Processing Function ---
def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame, model_name, y_threshold_factor, min_x_separation_factor):
    """
    Processes video, performs player selection, saves valid frames and data.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at '{input_path}'")
        return None
    print(f"Loading YOLO model: {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        print("Ensure the model file exists and ultralytics is installed.")
        return None

    print(f"Opening video file: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    y_pixel_threshold = frame_height * y_threshold_factor
    min_x_pixel_distance = frame_width * min_x_separation_factor
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Filtering: Y-Threshold < {y_pixel_threshold:.0f}px | Min X-Separation > {min_x_pixel_distance:.0f}px")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened():
        print(f"Error: Could not open video writer for '{output_video_path}'")
        cap.release()
        return None
    print(f"Output video (valid frames only) will be saved to: {output_video_path}")

    csv_file = None
    csv_writer = None # Initialize csv_writer
    try:
        csv_file = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y'])
        print(f"Output CSV (valid frames only) will be saved to: {output_csv_path}")
    except IOError as e:
        print(f"Error opening CSV file '{output_csv_path}': {e}")
        cap.release()
        if out_video.isOpened(): out_video.release()
        if csv_file and not csv_file.closed: csv_file.close()
        return None
    # Ensure csv_writer is available if try block succeeded
    if csv_writer is None:
        print(f"Error: CSV writer could not be initialized for '{output_csv_path}'.")
        cap.release()
        if out_video.isOpened(): out_video.release()
        if csv_file and not csv_file.closed: csv_file.close()
        return None


    frame_count = 0
    processed_frame_count = 0
    valid_frame_count = 0
    start_time = time.time()
    actual_end_frame = end_frame if end_frame > 0 and end_frame <= total_frames else total_frames
    print(f"Processing frames from {start_frame} to {actual_end_frame - 1}...")

    # --- Video Processing Loop ---
    while cap.isOpened():
        if frame_count >= actual_end_frame:
             print(f"\nReached target end frame: {actual_end_frame}")
             break

        success, frame = cap.read()
        if not success:
            print("\nEnd of video stream or error reading frame.")
            break

        if frame_count >= start_frame:
            processed_frame_count += 1
            results = model(frame, verbose=False, classes=[0])

            all_detected_persons = []
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    all_detected_persons.append({'center_x': center_x, 'center_y': center_y})

            player1_pos, player2_pos = select_players(
                all_detected_persons,
                frame_width,
                min_x_separation_factor
            )

            is_valid = bool(player1_pos['x'] and player2_pos['x'])

            if is_valid:
                valid_frame_count += 1
                csv_writer.writerow([
                    frame_count,
                    player1_pos['x'], player1_pos['y'],
                    player2_pos['x'], player2_pos['y']
                ])

                annotated_frame = results[0].plot() # This draws boxes and skeletons

                # 2. Draw our P1/P2 identifiers ON TOP of the YOLO annotations
                p1_coords = (player1_pos['x'], player1_pos['y'])
                p2_coords = (player2_pos['x'], player2_pos['y'])
                # Optional: Make circles slightly larger or different color to stand out
                cv2.circle(annotated_frame, p1_coords, 8, (0, 255, 0), -1) # Green circle P1
                cv2.putText(annotated_frame, 'P1', (p1_coords[0] + 10, p1_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3) # Slightly bigger text
                cv2.circle(annotated_frame, p2_coords, 8, (0, 0, 255), -1) # Red circle P2
                cv2.putText(annotated_frame, 'P2', (p2_coords[0] + 10, p2_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3) # Slightly bigger text

                # Write the frame (now with boxes, skeletons, P1/P2) to the output video
                out_video.write(annotated_frame)

            if processed_frame_count % 100 == 0 and processed_frame_count > 0:
                 elapsed_time = time.time() - start_time
                 fps_current = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
                 print(f"  Processed frame {frame_count} (Analyzed: {processed_frame_count}, Valid: {valid_frame_count}, FPS: {fps_current:.2f})")

        frame_count += 1
    # --- End Video Processing Loop ---

    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---")
    print(f"Frame range processed: {start_frame} to {frame_count - 1}")
    print(f"Total frames analyzed in range: {processed_frame_count}")
    print(f"Valid frames written to output: {valid_frame_count}")
    valid_percentage = (valid_frame_count / processed_frame_count * 100) if processed_frame_count > 0 else 0
    print(f"Percentage of valid frames: {valid_percentage:.2f}%")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average processing FPS: {avg_fps:.2f}")

    # --- Cleanup ---
    cap.release()
    if out_video.isOpened(): out_video.release()
    if csv_file and not csv_file.closed: csv_file.close()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Filtered video: '{output_video_path}', Filtered CSV: '{output_csv_path}'.")
    return frame_width, frame_height

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting Table Tennis Analyzer (with filtering)...")
    dimensions = process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        OUTPUT_CSV,
        START_FRAME,
        END_FRAME,
        MODEL_NAME,
        PLAYER_Y_THRESHOLD_FACTOR,
        MIN_X_SEPARATION_FACTOR
    )

    if dimensions:
        if os.path.exists(CSV_PATH_FOR_HEATMAP) and os.path.getsize(CSV_PATH_FOR_HEATMAP) > 0:
            create_heatmap(
                csv_path=CSV_PATH_FOR_HEATMAP,
                video_path=VIDEO_PATH_FOR_HEATMAP,
                output_image_path=OUTPUT_HEATMAP_PATH
            )
        else:
            print(f"Warning: Filtered CSV file '{CSV_PATH_FOR_HEATMAP}' is missing or empty after processing. Skipping heatmap generation.")
    else:
        print("Video processing failed or returned no dimensions. Skipping heatmap generation.")

    print("\nScript finished.")