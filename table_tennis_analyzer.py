import cv2
from ultralytics import YOLO
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# --- Constants for Video Processing ---
START_FRAME = 0
END_FRAME = 800
INPUT_VIDEO = 'input.mp4'
OUTPUT_VIDEO = 'output_with_detections.mp4'
OUTPUT_CSV = 'player_positions.csv'
PLAYER_Y_THRESHOLD_FACTOR = 0.60 

# --- Model Selection ---
MODEL_NAME = 'yolo11n-pose.pt' # Corrected model name

# --- Constants for Heatmap ---
VIDEO_PATH_FOR_HEATMAP = INPUT_VIDEO
CSV_PATH_FOR_HEATMAP = OUTPUT_CSV
OUTPUT_HEATMAP_PATH = "player_position_heatmap.png" 
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Heatmap Generation Function ---
def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """
    Generates a heatmap overlay on a video frame using player position data.
    Uses transparent zeros and percentile-based color scaling.

    Args:
        csv_path (str): Path to the player_positions.csv file.
        video_path (str): Path to the original video file for background.
        output_image_path (str): Path to save the generated heatmap image.
        bins (int): Number of bins for the 2D histogram.
        alpha (float): Transparency level for the heatmap overlay.
        cmap_name (str): Matplotlib colormap name for the heatmap.
        interpolation (str): Interpolation method for imshow display.
        vmax_percentile (float): Percentile of non-zero density to use for vmax.
    """
    print(f"\n--- Heatmap Generation ---")
    print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}")
    print(f"Params: bins={bins}, alpha={alpha}, cmap='{cmap_name}', interp='{interpolation}', vmax_pctl={vmax_percentile}")

    # --- 1. Read and Process CSV Data ---
    try:
        df = pd.read_csv(csv_path)
        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        for col in pos_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        all_x = pd.concat([df['player1_position_x'].dropna(), df['player2_position_x'].dropna()], ignore_index=True)
        all_y = pd.concat([df['player1_position_y'].dropna(), df['player2_position_y'].dropna()], ignore_index=True)
        if all_x.empty or all_y.empty:
            print("Error: No valid numeric coordinates found in CSV.")
            return
        print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'.")
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
    ret, frame = cap.read() # Read the first frame
    if not ret:
        print(f"Error: Could not read frame from '{video_path}'")
        cap.release()
        return
    frame_height, frame_width, _ = frame.shape
    background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
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
        print("Warning: No non-zero density data found for vmax calculation.")

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
    fig, ax = plt.subplots(1, 1, figsize=(12, 7)) # Adjust figsize if needed
    # Display background
    ax.imshow(background_image, extent=[0, frame_width, frame_height, 0], aspect='auto')

    # Overlay heatmap if data exists
    if heatmap_data.max() > 0:
        heatmap_plot = ax.imshow(
            heatmap_data,
            cmap=transparent_cmap,      # Use map with transparent zero
            alpha=alpha,                # Apply overall alpha
            extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], # Match histogram edges
            origin='upper',
            interpolation=interpolation,# Apply smoothing
            aspect='auto',
            vmin=0,                     # Map 0 to transparent color
            vmax=calculated_vmax        # Use percentile-based max
        )
        # Add colorbar
        cbar = fig.colorbar(heatmap_plot, ax=ax, extend='max')
        cbar.set_label('Density')
    else:
        print("Skipping heatmap overlay as calculated data is empty.")

    # Set titles and labels
    ax.set_title('Combined Player Position Heatmap')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xticks([]) # Hide ticks
    ax.set_yticks([]) # Hide ticks
    plt.tight_layout()

    # --- 7. Save ---
    try:
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"Final heatmap image saved successfully to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving heatmap image: {e}")
    plt.close(fig)


# --- Main Processing Function (Improved Player/Umpire Filtering) ---
def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame, model_name, y_threshold_factor):
    """
    Processes video using YOLO pose detection, saves annotated video and player position CSV
    (filtering umpire based on detection count and Y-coordinate), and returns video dimensions.

    Args:
        input_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file with detections.
        output_csv_path (str): Path to save the filtered player positions CSV file.
        start_frame (int): Frame number to start processing from.
        end_frame (int): Frame number to end processing at (0 for full video).
        model_name (str): Name of the YOLO model to use.
        y_threshold_factor (float): Factor of frame height. Used as fallback filter.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at '{input_path}'")
        return None
    print(f"Loading YOLO model: {model_name}...")
    try:
        # Ensure the correct model name is used here
        model = YOLO(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        print("Please ensure the model name is correct (e.g., 'yolov8n-pose.pt') and ultralytics is installed.")
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
    y_pixel_threshold = frame_height * y_threshold_factor # Calculate threshold for fallback
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Fallback Y-threshold: Persons with center Y < {y_pixel_threshold:.0f} pixels will be filtered out in fallback cases.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {output_video_path}")

    csv_file = None
    try:
        csv_file = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y'])
        print(f"Output CSV will be saved to: {output_csv_path}")
    except IOError as e:
        print(f"Error opening CSV file '{output_csv_path}': {e}")
        cap.release()
        return None

    frame_count = 0
    processed_frame_count = 0
    start_time = time.time()
    actual_end_frame = end_frame if end_frame > 0 else total_frames
    print(f"Processing frames from {start_frame} to {actual_end_frame}...")

    # --- Video Processing Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        if frame_count >= start_frame and (end_frame == 0 or frame_count < end_frame):
            processed_frame_count += 1
            results = model(frame, verbose=False) # Run detection

            player1_pos = {'x': '', 'y': ''}
            player2_pos = {'x': '', 'y': ''}
            all_detected_persons = [] # Store all detected persons with their coords

            # Get all detected persons
            if results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                for i, box in enumerate(boxes):
                    if int(classes[i]) == 0: # Is it a person?
                        xmin, ymin, xmax, ymax = box
                        center_x = (xmin + xmax) / 2
                        center_y = (ymin + ymax) / 2
                        all_detected_persons.append({'center_x': center_x, 'center_y': center_y})

            # --- Player Selection Logic ---
            selected_players = []
            num_detected = len(all_detected_persons)

            if num_detected == 3:
                all_detected_persons.sort(key=lambda p: p['center_y']) # Sort by Y, ascending
                selected_players = all_detected_persons[1:] # Take the lower two
            elif num_detected == 2:
                selected_players = all_detected_persons
            else:
                potential_players = []
                for person in all_detected_persons:
                    if person['center_y'] > y_pixel_threshold:
                        potential_players.append(person)

                if len(potential_players) >= 2:
                    potential_players.sort(key=lambda p: p['center_y'], reverse=True)
                    selected_players = potential_players[:2] # Take the lowest 2
                elif len(potential_players) == 1:
                    selected_players = potential_players # Take the single one

            # Assign P1 and P2 from the selected players
            if len(selected_players) >= 2:
                selected_players.sort(key=lambda p: p['center_x']) # Sort the final two by X
                player1_pos['x'] = int(selected_players[0]['center_x'])
                player1_pos['y'] = int(selected_players[0]['center_y'])
                player2_pos['x'] = int(selected_players[1]['center_x'])
                player2_pos['y'] = int(selected_players[1]['center_y'])
            elif len(selected_players) == 1:
                # If only one player selected (e.g., after fallback)
                player1_pos['x'] = int(selected_players[0]['center_x'])
                player1_pos['y'] = int(selected_players[0]['center_y'])

            # Write ONLY identified player positions to CSV
            csv_writer.writerow([
                frame_count,
                player1_pos['x'], player1_pos['y'],
                player2_pos['x'], player2_pos['y']
            ])

            # Annotate video frame (shows all original detections)
            annotated_frame = results[0].plot()

            # Optional: Draw center points only for identified players
            if player1_pos['x']:
                 cv2.circle(annotated_frame, (player1_pos['x'], player1_pos['y']), 5, (0, 255, 0), -1)
                 cv2.putText(annotated_frame, 'P1', (player1_pos['x']+10, player1_pos['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if player2_pos['x']:
                 cv2.circle(annotated_frame, (player2_pos['x'], player2_pos['y']), 5, (0, 0, 255), -1)
                 cv2.putText(annotated_frame, 'P2', (player2_pos['x']+10, player2_pos['y']), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            out_video.write(annotated_frame)

            if processed_frame_count % 100 == 0:
                 print(f"Processed frame {frame_count} (Total processed: {processed_frame_count})")

        elif frame_count >= actual_end_frame and end_frame != 0:
             print(f"Reached specified end frame: {end_frame}")
             break
        frame_count += 1
    # --- End Video Processing Loop ---

    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---")
    print(f"Total frames processed: {processed_frame_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average processing FPS: {avg_fps:.2f}")

    # --- Cleanup ---
    cap.release()
    out_video.release()
    if csv_file and not csv_file.closed:
        csv_file.close()
    cv2.destroyAllWindows()
    print(f"Video processing complete. Output video saved to '{output_video_path}', positions saved to '{output_csv_path}'.")
    return frame_width, frame_height

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting Table Tennis Analyzer...")
    # Run video processing with the refined player selection logic
    dimensions = process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        OUTPUT_CSV,
        START_FRAME,
        END_FRAME,
        MODEL_NAME, # Use corrected model name
        PLAYER_Y_THRESHOLD_FACTOR # Pass the threshold factor for fallback cases
    )

    # If processing succeeded, generate the heatmap
    if dimensions:
        if os.path.exists(CSV_PATH_FOR_HEATMAP):
            create_heatmap(
                csv_path=CSV_PATH_FOR_HEATMAP,
                video_path=VIDEO_PATH_FOR_HEATMAP,
                output_image_path=OUTPUT_HEATMAP_PATH
                # Uses heatmap parameters defined in constants
            )
        else:
            print(f"Error: CSV file '{CSV_PATH_FOR_HEATMAP}' not found after video processing. Cannot generate heatmap.")
    else:
        print("Video processing failed. Skipping heatmap generation.")

    print("Script finished.")
