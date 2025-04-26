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
import easyocr # Import EasyOCR
import re # Import regex for parsing score

# --- Constants for Video Processing ---
START_FRAME = 1250 # Start processing around 25 seconds (Adjust if needed)
END_FRAME = 3250   # Stop processing around 65 seconds (Adjust if needed)
INPUT_VIDEO = 'input.mp4'
OUTPUT_VIDEO = 'output_with_detections_filtered.mp4' # Video with pose, P1/P2, score overlay
OUTPUT_CSV = 'player_positions.csv'      # Combined position and score data
MIN_X_SEPARATION_FACTOR = 0.10 # Minimum horizontal separation between players

# --- Constants for Score OCR ---
# !!! IMPORTANT: Coordinates derived from user screenshots (Full Resolution 1920x1080) !!!
# Format: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

# Player 1 (Top score line - Numbers only) - Based on 19.32.27.png
ROI_PLAYER1_SCORE = (445, 926, 557, 977)

# Player 2 (Bottom score line - Numbers only) - Based on 19.32.38.png
ROI_PLAYER2_SCORE = (444, 974, 556, 1025)

# --- Model Selection ---
MODEL_NAME = 'yolov8n-pose.pt' # Make sure this model file exists

# --- Constants for Heatmap ---
VIDEO_PATH_FOR_HEATMAP = INPUT_VIDEO
CSV_PATH_FOR_HEATMAP = OUTPUT_CSV # Use the combined CSV
OUTPUT_HEATMAP_PATH = "player_position_heatmap.png" # Desired heatmap name
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Constants for Score Chart ---
OUTPUT_SCORE_CHART_PATH = "score_chart.png" # Desired chart name
CHART_FRAME_SAMPLE_RATE = 50 # Plot score data every N frames (adjust as needed)

# --- Initialize EasyOCR Reader ---
# Done once for efficiency. Downloads models on first run.
try:
    print("Initializing EasyOCR Reader (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if using GPU
    print("EasyOCR Reader initialized.")
    OCR_ENABLED = True
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Score recognition will be disabled.")
    OCR_ENABLED = False
    reader = None # Ensure reader is None if disabled

# --- Heatmap Generation Function ---
def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """Generates a heatmap overlay on a video frame using player position data."""
    print(f"\n--- Heatmap Generation ---")
    print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}")
    print(f"Params: bins={bins}, alpha={alpha}, cmap='{cmap_name}', interp='{interpolation}', vmax_pctl={vmax_percentile}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
             print(f"Error: CSV file '{csv_path}' not found or is empty. Cannot generate heatmap.")
             return
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV file '{csv_path}' contains no data rows. Cannot generate heatmap.")
            return

        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        if not all(col in df.columns for col in pos_cols):
             print(f"Error: One or more position columns missing in '{csv_path}'. Cannot generate heatmap.")
             return

        for col in pos_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')

        df_clean = df.dropna(subset=pos_cols)
        if not df_clean.empty:
            all_x = pd.concat([df_clean['player1_position_x'], df_clean['player2_position_x']], ignore_index=True)
            all_y = pd.concat([df_clean['player1_position_y'], df_clean['player2_position_y']], ignore_index=True)
        else:
             all_x = pd.Series(dtype=float)
             all_y = pd.Series(dtype=float)

        if all_x.empty or all_y.empty:
            print(f"Warning: No valid numeric coordinates found after cleaning CSV '{csv_path}'. Heatmap might be empty.")
        else:
            print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'.")
        return
    except pd.errors.EmptyDataError:
         print(f"Error: CSV file '{csv_path}' is empty.")
         return
    except Exception as e:
        print(f"Error reading/processing CSV '{csv_path}' for heatmap: {e}")
        return

    # --- Get Background Frame ---
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

    # --- Create 2D Histogram Data ---
    if all_x.empty or all_y.empty:
        print("Skipping histogram calculation due to no valid coordinates.")
        heatmap_data = np.zeros((bins, bins))
        xedges = np.linspace(0, frame_width, bins + 1)
        yedges = np.linspace(0, frame_height, bins + 1)
    else:
        try:
            heatmap_data, xedges, yedges = np.histogram2d(
                all_x, all_y, bins=bins, range=[[0, frame_width], [0, frame_height]]
            )
            heatmap_data = heatmap_data.T
            print(f"Histogram shape: {heatmap_data.shape}, Max value: {heatmap_data.max():.2f}")
            if heatmap_data.max() == 0:
                print("Warning: Histogram data is all zeros.")
        except Exception as e:
            print(f"Error during np.histogram2d calculation: {e}")
            return

    # --- Calculate Vmax and Modify Colormap ---
    non_zero_data = heatmap_data[heatmap_data > 0]
    calculated_vmax = 1.0
    if non_zero_data.size > 0:
        calculated_vmax = np.percentile(non_zero_data, vmax_percentile)
        calculated_vmax = max(calculated_vmax, 1.0)
        print(f"Using {vmax_percentile}th percentile vmax for color scale: {calculated_vmax:.2f}")
    else:
        print("Warning: No non-zero density data found for vmax calculation. Setting vmax=1.0")

    try:
        current_cmap = plt.get_cmap(cmap_name).copy()
        cmap_list = current_cmap(np.linspace(0, 1, current_cmap.N))
        cmap_list[0, 3] = 0.0
        transparent_cmap = mcolors.ListedColormap(cmap_list)
    except Exception as e:
        print(f"Error modifying colormap '{cmap_name}': {e}")
        transparent_cmap = plt.get_cmap(cmap_name)

    # --- Generate Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.imshow(background_image, extent=[0, frame_width, frame_height, 0], aspect='auto')

    if heatmap_data.max() > 0:
        heatmap_plot = ax.imshow(
            heatmap_data, cmap=transparent_cmap, alpha=alpha,
            extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]],
            origin='upper', interpolation=interpolation, aspect='auto',
            vmin=0, vmax=calculated_vmax
        )
        extend_direction = 'max' if calculated_vmax < heatmap_data.max() else 'neither'
        cbar = fig.colorbar(heatmap_plot, ax=ax, extend=extend_direction)
        cbar.set_label('Position Density')
    else:
        print("Skipping heatmap overlay as data is empty or all zeros.")

    ax.set_title('Player Position Heatmap')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # --- Save ---
    try:
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap image saved successfully to '{output_image_path}'")
    except Exception as e:
        print(f"Error saving heatmap image: {e}")
    plt.close(fig)


# --- Helper Function for Player Selection ---
def select_players(all_detected_persons, frame_width, min_x_separation_factor):
    """Selects the two players based on detection count and relative positions."""
    player1_pos = {'x': '', 'y': ''}
    player2_pos = {'x': '', 'y': ''}
    selected_players = []
    num_detected = len(all_detected_persons)
    min_x_distance = frame_width * min_x_separation_factor

    if num_detected == 3:
        all_detected_persons.sort(key=lambda p: p['center_y'], reverse=True)
        p_low1 = all_detected_persons[0]
        p_low2 = all_detected_persons[1]
        x_distance = abs(p_low1['center_x'] - p_low2['center_x'])
        if x_distance >= min_x_distance:
            selected_players = [p_low1, p_low2]
    elif num_detected == 2:
        p1 = all_detected_persons[0]
        p2 = all_detected_persons[1]
        x_distance = abs(p1['center_x'] - p2['center_x'])
        if x_distance >= min_x_distance:
            selected_players = all_detected_persons

    if len(selected_players) == 2:
        selected_players.sort(key=lambda p: p['center_x'])
        player1_pos['x'] = int(selected_players[0]['center_x'])
        player1_pos['y'] = int(selected_players[0]['center_y'])
        player2_pos['x'] = int(selected_players[1]['center_x'])
        player2_pos['y'] = int(selected_players[1]['center_y'])

    return player1_pos, player2_pos

# --- Helper Function for Score Parsing ---
def parse_score_text(ocr_results):
    """Parses EasyOCR results to find table tennis score (Games Points)."""
    full_text = " ".join([res[1] for res in ocr_results]).strip()

    # --- Debug Print ---
    if full_text:
        print(f"    Attempting to parse OCR text: '{full_text}'")
    # --- End Debug Print ---

    match = re.search(r'(\d+)\D*(\d+)', full_text)
    if match:
        try:
            games = int(match.group(1))
            points = int(match.group(2))
            return games, points
        except (ValueError, IndexError):
            pass

    numbers = re.findall(r'\d+', full_text)
    if len(numbers) >= 2:
         try:
              games = int(numbers[0])
              points = int(numbers[1])
              return games, points
         except ValueError:
              pass

    return None, None

# --- Function to Create Score Chart ---
def create_score_chart(csv_path, output_chart_path, sample_rate=CHART_FRAME_SAMPLE_RATE):
    """Creates a line chart of calculated scores over time from the CSV."""
    print(f"\n--- Score Chart Generation ---")
    print(f"CSV: {csv_path}, Output: {output_chart_path}, SampleRate: {sample_rate}")

    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
             print(f"Error: CSV file '{csv_path}' not found or is empty.")
             return
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Error: CSV file '{csv_path}' contains no data rows.")
            return

        score_cols = ['p1_calc_score', 'p2_calc_score', 'frame']
        if not all(col in df.columns for col in score_cols):
             print(f"Error: Required score columns {score_cols} missing in '{csv_path}'.")
             return

        df['p1_calc_score'] = pd.to_numeric(df['p1_calc_score'], errors='coerce')
        df['p2_calc_score'] = pd.to_numeric(df['p2_calc_score'], errors='coerce')
        df_scores = df.dropna(subset=['p1_calc_score', 'p2_calc_score'])

        if df_scores.empty:
            print("Error: No valid score data found in CSV after cleaning. Cannot generate chart.")
            return

        if sample_rate <= 0: sample_rate = 1
        df_sampled = df_scores.iloc[::sample_rate, :]
        if df_sampled.empty and not df_scores.empty:
             print(f"Warning: No data points remain after applying sample rate {sample_rate}. Plotting first/last points.")
             df_sampled = df_scores.iloc[[0, -1]] if len(df_scores) >= 2 else df_scores

        if df_sampled.empty:
            print("Error: Still no data to plot. Skipping chart.")
            return

        print(f"Plotting {len(df_sampled)} data points for score chart.")

        plt.figure(figsize=(15, 7))
        plt.plot(df_sampled['frame'], df_sampled['p1_calc_score'], label='Player 1 Score', marker='.', linestyle='-', markersize=4)
        plt.plot(df_sampled['frame'], df_sampled['p2_calc_score'], label='Player 2 Score', marker='.', linestyle='-', markersize=4)
        plt.xlabel(f"Frame Number (Sampled)")
        plt.ylabel("Calculated Score (Games*10 + Points)")
        plt.title("Calculated Table Tennis Score Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(output_chart_path, dpi=150)
        print(f"Score chart saved successfully to '{output_chart_path}'")
        plt.close()

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_path}'.")
    except pd.errors.EmptyDataError:
         print(f"Error: CSV file '{csv_path}' is empty.")
    except Exception as e:
        print(f"Error generating score chart: {e}")

# --- Main Processing Function (with Periodic OCR) ---
def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame, model_name, min_x_separation_factor, ocr_reader, p1_roi, p2_roi):
    """
    Processes video for player positions and scores (periodic OCR), saves data and annotated video.
    """
    # --- Basic Setup and Checks --- (Same as before)
    if not os.path.exists(input_path):
        print(f"Error: Input video file not found at '{input_path}'")
        return None
    ocr_is_active = isinstance(ocr_reader, easyocr.Reader)
    if not ocr_is_active and OCR_ENABLED:
        print("Warning: OCR was enabled but Reader is not available. Score processing skipped.")

    print(f"Loading YOLO model: {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

    print(f"Opening video file: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        print("Warning: Could not determine video FPS. Assuming 30 FPS for OCR interval.")
        fps = 30 # Provide a default if FPS is not available

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_x_pixel_distance = frame_width * min_x_separation_factor

    # Calculate OCR frame interval
    OCR_CHECK_INTERVAL_SECONDS = 3
    ocr_frame_interval = int(OCR_CHECK_INTERVAL_SECONDS * fps)
    ocr_frame_interval = max(1, ocr_frame_interval) # Ensure interval is at least 1

    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Filtering: Prioritizing 2/3 detections | Min X-Separation > {min_x_pixel_distance:.0f}px")
    if ocr_is_active:
        print(f"Score ROIs: P1={p1_roi}, P2={p2_roi}")
        print(f"Running score OCR check every {OCR_CHECK_INTERVAL_SECONDS} seconds (approx. every {ocr_frame_interval} frames).")
    else:
        print("Score OCR reader not active.")

    # --- Output Setup --- (Same as before)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened():
        print(f"Error: Could not open video writer for '{output_video_path}'"); cap.release(); return None
    print(f"Output video (annotated valid frames) will be saved to: {output_video_path}")

    csv_file = None
    csv_writer = None
    try:
        csv_file = open(output_csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y',
            'p1_games_read', 'p1_points_read', 'p1_calc_score', 'p2_games_read', 'p2_points_read', 'p2_calc_score'
        ])
        print(f"Output CSV (positions and scores) will be saved to: {output_csv_path}")
    except IOError as e:
        print(f"Error opening CSV file '{output_csv_path}': {e}")
        # Cleanup resources before returning
        cap.release() # Close video capture
        if out_video.isOpened(): # Check if video writer is open
            out_video.release() # Close video writer
        # No need to close csv_file here as it failed to open
        return None # Exit the function

    if csv_writer is None: # Check if writer initialization failed (less likely now)
        print(f"Error: CSV writer could not be initialized.")
        # Cleanup resources before returning
        cap.release()
        if out_video.isOpened():
            out_video.release()
        # Close file only if it was opened but writer init failed
        if csv_file and not csv_file.closed:
            csv_file.close()
        return None # Exit the function
    # --- Processing Loop Init --- (Same as before)
    frame_count = 0
    processed_frame_count = 0
    valid_player_frame_count = 0
    ocr_success_count = 0
    start_time = time.time()
    actual_end_frame = end_frame if end_frame > 0 and end_frame <= total_frames else total_frames
    print(f"Processing frames from {start_frame} to {actual_end_frame - 1}...")
    last_p1_score_str = "P1: - -"
    last_p2_score_str = "P2: - -"
    saved_roi_count = 0
    max_debug_rois = 10

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

            # --- Player Detection (Runs every frame) ---
            results = model(frame, verbose=False, classes=[0])
            all_detected_persons = []
            # ... (code to extract detected persons - same as before) ...
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    all_detected_persons.append({'center_x': center_x, 'center_y': center_y})

            player1_pos, player2_pos = select_players(
                all_detected_persons, frame_width, min_x_separation_factor
            )
            is_valid_player_frame = bool(player1_pos['x'] and player2_pos['x'])

            # --- Score OCR (Run periodically) ---
            p1_games, p1_points, p1_calc = '', '', ''
            p2_games, p2_points, p2_calc = '', '', ''
            score_read_success = False
            # Keep using last known score for overlay by default
            current_p1_str = last_p1_score_str
            current_p2_str = last_p2_score_str

            # ---> Check if it's time to run OCR <---
            if ocr_is_active and frame_count % ocr_frame_interval == 0:
                print(f"--- Running OCR on frame {frame_count} ---") # Log OCR attempt

                x1_p1, y1_p1, x2_p1, y2_p1 = p1_roi
                x1_p2, y1_p2, x2_p2, y2_p2 = p2_roi
                y1_p1, y2_p1 = max(0, y1_p1), min(frame_height, y2_p1)
                x1_p1, x2_p1 = max(0, x1_p1), min(frame_width, x2_p1)
                y1_p2, y2_p2 = max(0, y1_p2), min(frame_height, y2_p2)
                x1_p2, x2_p2 = max(0, x1_p2), min(frame_width, x2_p2)

                # --- Save ROI images for debugging (Still useful for first few OCR attempts) ---
                if is_valid_player_frame and saved_roi_count < max_debug_rois:
                    print(f"  DEBUG: Attempting to save ROI images for frame {frame_count} (Count: {saved_roi_count+1}/{max_debug_rois})")
                    # ... (rest of the ROI saving code - same as before) ...
                    try:
                        roi1_saved, roi2_saved = False, False
                        processed_roi1_to_save, processed_roi2_to_save = None, None
                        thresh_val_dbg, thresh_type_dbg = 150, cv2.THRESH_BINARY # Or tuned values
                        if y2_p1 > y1_p1 and x2_p1 > x1_p1:
                             roi1_img_orig = frame[y1_p1:y2_p1, x1_p1:x2_p1]
                             cv2.imwrite(f"debug_roi1_orig_frame{frame_count}.png", roi1_img_orig)
                             gray1 = cv2.cvtColor(roi1_img_orig, cv2.COLOR_BGR2GRAY)
                             ret1, processed_roi1_to_save = cv2.threshold(gray1, thresh_val_dbg, 255, thresh_type_dbg)
                             cv2.imwrite(f"debug_roi1_processed_frame{frame_count}.png", processed_roi1_to_save)
                             roi1_saved = True
                        if y2_p2 > y1_p2 and x2_p2 > x1_p2:
                             roi2_img_orig = frame[y1_p2:y2_p2, x1_p2:x2_p2]
                             cv2.imwrite(f"debug_roi2_orig_frame{frame_count}.png", roi2_img_orig)
                             gray2 = cv2.cvtColor(roi2_img_orig, cv2.COLOR_BGR2GRAY)
                             ret2, processed_roi2_to_save = cv2.threshold(gray2, thresh_val_dbg, 255, thresh_type_dbg)
                             cv2.imwrite(f"debug_roi2_processed_frame{frame_count}.png", processed_roi2_to_save)
                             roi2_saved = True
                        if roi1_saved or roi2_saved: saved_roi_count += 1
                    except Exception as e_write: print(f" Warning: Could not write debug ROI image: {e_write}")
                # --- End Debug ROI Save ---


                # --- Preprocessing ROIs for OCR (Keep enabled, adjust values if needed) ---
                processed_roi1, processed_roi2 = None, None
                thresh_val = 150 # Keep tuning this if needed
                threshold_type = cv2.THRESH_BINARY # Or THRESH_BINARY_INV

                if y2_p1 > y1_p1 and x2_p1 > x1_p1:
                     roi1_img = frame[y1_p1:y2_p1, x1_p1:x2_p1]
                     gray1 = cv2.cvtColor(roi1_img, cv2.COLOR_BGR2GRAY)
                     ret1, processed_roi1 = cv2.threshold(gray1, thresh_val, 255, threshold_type)
                if y2_p2 > y1_p2 and x2_p2 > x1_p2:
                     roi2_img = frame[y1_p2:y2_p2, x1_p2:x2_p2]
                     gray2 = cv2.cvtColor(roi2_img, cv2.COLOR_BGR2GRAY)
                     ret2, processed_roi2 = cv2.threshold(gray2, thresh_val, 255, threshold_type)
                # --- End Preprocessing ---


                # Run OCR on processed ROIs
                ocr1_results, ocr2_results = [], []
                if processed_roi1 is not None:
                    try: ocr1_results = ocr_reader.readtext(processed_roi1)
                    except Exception as e_ocr: print(f" Warning: OCR ROI1 failed: {e_ocr}")
                if processed_roi2 is not None:
                    try: ocr2_results = ocr_reader.readtext(processed_roi2)
                    except Exception as e_ocr: print(f" Warning: OCR ROI2 failed: {e_ocr}")

                # Parse results
                p1_games, p1_points = parse_score_text(ocr1_results)
                p2_games, p2_points = parse_score_text(ocr2_results)

                # Process if both scores parsed correctly
                if p1_games is not None and p2_games is not None:
                    score_read_success = True
                    ocr_success_count += 1 # Increment overall success count
                    p1_calc = p1_games * 10 + p1_points
                    p2_calc = p2_games * 10 + p2_points
                    # Update strings for overlay with CURRENTLY read score
                    current_p1_str = f"P1: {p1_games} - {p1_points}"
                    current_p2_str = f"P2: {p2_games} - {p2_points}"
                    last_p1_score_str = current_p1_str # Update last known good
                    last_p2_score_str = current_p2_str
                # If OCR failed this time, current_pX_str keeps last known value

            # --- End Periodic OCR block ---


            # --- Write CSV Data (Every Frame) ---
            csv_writer.writerow([
                frame_count,
                player1_pos['x'] if is_valid_player_frame else '', player1_pos['y'] if is_valid_player_frame else '',
                player2_pos['x'] if is_valid_player_frame else '', player2_pos['y'] if is_valid_player_frame else '',
                # Write score only if read successfully in this specific periodic check
                p1_games if score_read_success else '', p1_points if score_read_success else '', p1_calc if score_read_success else '',
                p2_games if score_read_success else '', p2_points if score_read_success else '', p2_calc if score_read_success else ''
            ])

            # --- Annotation and Video Output (Only for Valid Player Frames) ---
            if is_valid_player_frame:
                valid_player_frame_count += 1
                annotated_frame = results[0].plot()
                # ... (Draw P1/P2 markers - same as before) ...
                p1_coords = (player1_pos['x'], player1_pos['y'])
                p2_coords = (player2_pos['x'], player2_pos['y'])
                cv2.circle(annotated_frame, p1_coords, 8, (0, 255, 0), -1)
                cv2.putText(annotated_frame, 'P1', (p1_coords[0] + 10, p1_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.circle(annotated_frame, p2_coords, 8, (0, 0, 255), -1)
                cv2.putText(annotated_frame, 'P2', (p2_coords[0] + 10, p2_coords[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


                # Overlay score (uses last known good score stored in current_pX_str)
                if ocr_is_active:
                    score_display_y = 50; score_display_x = 20
                    font_scale = 1.2; font_thickness = 3; font_color = (255, 255, 0) # Cyan
                    cv2.putText(annotated_frame, current_p1_str, (score_display_x, score_display_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                    cv2.putText(annotated_frame, current_p2_str, (score_display_x, score_display_y + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

                out_video.write(annotated_frame)

            # --- Progress Update --- (Same as before)
            if processed_frame_count % 100 == 0 and processed_frame_count > 0:
                 elapsed_time = time.time() - start_time
                 fps_current = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
                 ocr_rate = (ocr_success_count / (processed_frame_count / ocr_frame_interval) * 100) if processed_frame_count >= ocr_frame_interval else 0 # Approx OCR success rate per attempt
                 print(f"  Frame {frame_count} (Processed: {processed_frame_count}, Players OK: {valid_player_frame_count}, OCR OK: {ocr_success_count} [{ocr_rate:.1f}% per check], FPS: {fps_current:.2f})")


        frame_count += 1
    # --- End Video Processing Loop ---

    # --- Final Summary and Cleanup --- (Same as before)
    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---")
    print(f"Frame range processed: {start_frame} to {frame_count - 1}")
    print(f"Total frames analyzed: {processed_frame_count}")
    print(f"Valid player frames written: {valid_player_frame_count}")
    print(f"Successful score reads (periodic checks): {ocr_success_count}") # Updated label
    valid_percentage = (valid_player_frame_count / processed_frame_count * 100) if processed_frame_count > 0 else 0
    # Calculate overall OCR success rate based on number of checks made
    num_ocr_checks = processed_frame_count // ocr_frame_interval
    ocr_overall_success_rate = (ocr_success_count / num_ocr_checks * 100) if num_ocr_checks > 0 else 0
    print(f"Percentage valid player frames: {valid_percentage:.2f}%")
    print(f"Percentage successful score reads (of checks): {ocr_overall_success_rate:.2f}%") # Updated label
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average processing FPS: {avg_fps:.2f}") # Will be higher now

    cap.release()
    if out_video.isOpened(): out_video.release()
    if csv_file and not csv_file.closed: csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nVideo processing complete.")
    print(f"Output video: '{output_video_path}'")
    print(f"Output CSV: '{output_csv_path}'")
    return frame_width, frame_height

# --- (The rest of the code: create_heatmap, create_score_chart, if __name__ == "__main__": etc. remain the same) ---

# Example of the __main__ block (ensure it calls the updated process_video correctly)
if __name__ == "__main__":
    print("Starting Table Tennis Analyzer (with Periodic Score OCR)...") # Updated title

    dimensions = process_video(
        INPUT_VIDEO,
        OUTPUT_VIDEO,
        OUTPUT_CSV,
        START_FRAME,
        END_FRAME,
        MODEL_NAME,
        MIN_X_SEPARATION_FACTOR,
        reader,                 # Pass the initialized EasyOCR reader (or None)
        ROI_PLAYER1_SCORE,      # Pass player 1 score ROI
        ROI_PLAYER2_SCORE       # Pass player 2 score ROI
    )

    # --- Post-processing ---
    if dimensions:
        print("\n--- Starting Post-Processing ---")
        if os.path.exists(CSV_PATH_FOR_HEATMAP) and os.path.getsize(CSV_PATH_FOR_HEATMAP) > 0:
            create_heatmap(
                csv_path=CSV_PATH_FOR_HEATMAP,
                video_path=VIDEO_PATH_FOR_HEATMAP,
                output_image_path=OUTPUT_HEATMAP_PATH
            )
        else:
            print(f"\nWarning: Data CSV file '{CSV_PATH_FOR_HEATMAP}' is missing or empty. Skipping heatmap generation.")

        if OCR_ENABLED:
             if os.path.exists(CSV_PATH_FOR_HEATMAP) and os.path.getsize(CSV_PATH_FOR_HEATMAP) > 0:
                create_score_chart(
                    csv_path=CSV_PATH_FOR_HEATMAP,
                    output_chart_path=OUTPUT_SCORE_CHART_PATH,
                    sample_rate=CHART_FRAME_SAMPLE_RATE
                )
             else:
                 print(f"\nWarning: Data CSV file '{CSV_PATH_FOR_HEATMAP}' is missing or empty. Skipping score chart generation.")
        else:
            print("\nSkipping score chart generation as OCR was disabled.")

    else:
        print("\nVideo processing failed or returned no dimensions. Skipping post-processing.")

    print("\nScript finished.")
    print("Reminder: Check/Adjust OCR Preprocessing threshold (thresh_val) if needed.")