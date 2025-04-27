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
import easyocr # Import EasyOCR
import re # Import regex for parsing score

# --- Constants for Video Processing ---
START_FRAME = 15000 # Start frame (adjust as needed)
END_FRAME = 20000   # End frame (adjust as needed, 0 for full video)
INPUT_VIDEO = 'input.mp4'
OUTPUT_VIDEO = 'output_with_detections.mp4'
OUTPUT_CSV = 'player_positions.csv'
MIN_X_SEPARATION_FACTOR = 0.10

# --- Constants for Score OCR ---
# Coordinates derived from user screenshots (Full Resolution 1920x1080)
ROI_PLAYER1_SCORE = (445, 926, 557, 977) # Player 1 (Top)
ROI_PLAYER2_SCORE = (445, 974, 557, 1025) # Player 2 (Bottom)
OCR_FRAME_INTERVAL = 5 # Run OCR check every 5 frames <<<--- UPDATED

# --- Model Selection ---
MODEL_NAME = 'yolo11n-pose.pt'

# --- Constants for Heatmap ---
VIDEO_PATH_FOR_HEATMAP = INPUT_VIDEO
CSV_PATH_FOR_HEATMAP = OUTPUT_CSV
OUTPUT_HEATMAP_PATH = "player_position_heatmap.png"
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Constants for Score Chart ---
OUTPUT_SCORE_CHART_PATH = "score_chart.png"
CHART_FRAME_SAMPLE_RATE = 5 # How often to plot points on the chart
CHART_PLOT_ALL_THRESHOLD = 100 # If fewer valid scores than this, plot all

# --- Initialize EasyOCR Reader ---
try:
    print("Initializing EasyOCR Reader (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=False)
    print("EasyOCR Reader initialized.")
    OCR_ENABLED = True
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Score recognition will be disabled.")
    OCR_ENABLED = False
    reader = None

# --- Heatmap Generation Function ---
def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """Generates a heatmap overlay on a video frame using player position data."""
    print(f"\n--- Heatmap Generation ---"); print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}"); print(f"Params: bins={bins}, alpha={alpha}, cmap='{cmap_name}', interp='{interpolation}', vmax_pctl={vmax_percentile}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return
        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        if not all(col in df.columns for col in pos_cols): print(f"Error: One or more position columns missing in '{csv_path}'."); return
        for col in pos_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df_clean = df.dropna(subset=pos_cols)
        if not df_clean.empty: all_x = pd.concat([df_clean['player1_position_x'], df_clean['player2_position_x']], ignore_index=True); all_y = pd.concat([df_clean['player1_position_y'], df_clean['player2_position_y']], ignore_index=True)
        else: all_x, all_y = pd.Series(dtype=float), pd.Series(dtype=float)
        if all_x.empty or all_y.empty: print(f"Warning: No valid numeric coordinates found after cleaning CSV '{csv_path}'. Heatmap might be empty.")
        else: print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")
    except Exception as e: print(f"Error reading/processing CSV '{csv_path}' for heatmap: {e}"); return
    if not os.path.exists(video_path): print(f"Error: Video not found '{video_path}'"); return
    cap = cv2.VideoCapture(video_path);
    if not cap.isOpened(): print(f"Error: Could not open video '{video_path}'"); return
    ret, frame = cap.read()
    if not ret: print(f"Error: Could not read frame from '{video_path}'"); cap.release(); return
    frame_height, frame_width, _ = frame.shape; background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); cap.release()
    print(f"Video dimensions (W x H): {frame_width} x {frame_height}")
    if all_x.empty or all_y.empty:
        print("Skipping histogram calculation due to no valid coordinates."); heatmap_data = np.zeros((bins, bins)); xedges = np.linspace(0, frame_width, bins + 1); yedges = np.linspace(0, frame_height, bins + 1)
    else:
        try:
            heatmap_data, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[0, frame_width], [0, frame_height]])
            heatmap_data = heatmap_data.T; print(f"Histogram shape: {heatmap_data.shape}, Max value: {heatmap_data.max():.2f}")
            if heatmap_data.max() == 0: print("Warning: Histogram data is all zeros.")
        except Exception as e: print(f"Error during np.histogram2d calculation: {e}"); return
    non_zero_data = heatmap_data[heatmap_data > 0]; calculated_vmax = 1.0
    if non_zero_data.size > 0: calculated_vmax = np.percentile(non_zero_data, vmax_percentile); calculated_vmax = max(calculated_vmax, 1.0); print(f"Using {vmax_percentile}th percentile vmax for color scale: {calculated_vmax:.2f}")
    else: print("Warning: No non-zero density data found for vmax calculation. Setting vmax=1.0")
    try: current_cmap = plt.get_cmap(cmap_name).copy(); cmap_list = current_cmap(np.linspace(0, 1, current_cmap.N)); cmap_list[0, 3] = 0.0; transparent_cmap = mcolors.ListedColormap(cmap_list)
    except Exception as e: print(f"Error modifying colormap '{cmap_name}': {e}"); transparent_cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7)); ax.imshow(background_image, extent=[0, frame_width, frame_height, 0], aspect='auto')
    if heatmap_data.max() > 0:
        heatmap_plot = ax.imshow(heatmap_data, cmap=transparent_cmap, alpha=alpha, extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], origin='upper', interpolation=interpolation, aspect='auto', vmin=0, vmax=calculated_vmax)
        extend_direction = 'max' if calculated_vmax < heatmap_data.max() else 'neither'; cbar = fig.colorbar(heatmap_plot, ax=ax, extend=extend_direction); cbar.set_label('Position Density')
    else: print("Skipping heatmap overlay as data is empty or all zeros.")
    ax.set_title('Player Position Heatmap'); ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_xticks([]); ax.set_yticks([]); plt.tight_layout()
    try: plt.savefig(output_image_path, dpi=150, bbox_inches='tight'); print(f"Heatmap image saved successfully to '{output_image_path}'")
    except Exception as e: print(f"Error saving heatmap image: {e}")
    plt.close(fig)

# --- Helper Function for Player Selection ---
def select_players(all_detected_persons, frame_width, min_x_separation_factor):
    """Selects the two players based on detection count and relative positions."""
    player1_pos={'x':'','y':''}; player2_pos={'x':'','y':''}; selected_players=[]; num_detected=len(all_detected_persons); min_x_distance=frame_width*min_x_separation_factor
    if num_detected == 3:
        all_detected_persons.sort(key=lambda p: p['center_y'], reverse=True); p_low1, p_low2 = all_detected_persons[0], all_detected_persons[1]
        if abs(p_low1['center_x'] - p_low2['center_x']) >= min_x_distance: selected_players = [p_low1, p_low2]
    elif num_detected == 2:
        p1, p2 = all_detected_persons[0], all_detected_persons[1]
        if abs(p1['center_x'] - p2['center_x']) >= min_x_distance: selected_players = all_detected_persons
    if len(selected_players) == 2:
        selected_players.sort(key=lambda p: p['center_x'])
        player1_pos['x'],player1_pos['y'] = int(selected_players[0]['center_x']),int(selected_players[0]['center_y'])
        player2_pos['x'],player2_pos['y'] = int(selected_players[1]['center_x']),int(selected_players[1]['center_y'])
    return player1_pos, player2_pos

# --- Helper Function for Score Parsing (Handles single digit as Games Won) ---
def parse_score_text(ocr_results):
    """
    Parses EasyOCR results from a 2-ROI setup.
    Tries to find two numbers (Games Points).
    If only one number is found, assumes it's Games Won and Points are 0.
    Returns (games_won, current_points) or (None, None).
    """
    full_text = " ".join([res[1] for res in ocr_results]).strip()
    if full_text:
        print(f"    Attempting to parse OCR text: '{full_text}'") # Debug print

    # 1. Try to find two numbers first
    match = re.search(r'(\d+)\D*(\d+)', full_text)
    if match:
        try:
            games = int(match.group(1))
            points = int(match.group(2))
            # print(f"      Found two numbers: ({games}, {points})") # Optional Debug
            return games, points
        except (ValueError, IndexError):
            pass # Failed conversion, proceed to check for single digit

    # 2. If two numbers not found, check for exactly one number
    numbers = re.findall(r'\d+', full_text)
    if len(numbers) == 1:
        try:
            single_digit = int(numbers[0])
            # Assume single digit is Games Won, Points are 0
            # print(f"      Found one number: {single_digit}, assuming Games Won.") # Optional Debug
            return single_digit, 0
        except ValueError:
            pass # Failed conversion

    # 3. If neither two nor exactly one number found
    # print(f"      Could not parse score from '{full_text}'") # Optional Debug
    return None, None
# --- Function to Create Score Chart (with Outlier Filtering and Improved Sampling Logic) ---
def create_score_chart(csv_path, output_chart_path, sample_rate=CHART_FRAME_SAMPLE_RATE, plot_all_threshold=CHART_PLOT_ALL_THRESHOLD):
    """Creates a line chart of calculated scores over time from the CSV, filtering out single-frame outliers."""
    print(f"\n--- Score Chart Generation ---")
    print(f"CSV: {csv_path}, Output: {output_chart_path}, SampleRate: {sample_rate}, PlotAllThreshold: {plot_all_threshold}")
    try:
        # --- Read and Initial Clean ---
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return
        score_cols = ['p1_calc_score', 'p2_calc_score', 'frame']
        if not all(col in df.columns for col in score_cols): print(f"Error: Required score columns {score_cols} missing in '{csv_path}'."); return

        df['p1_calc_score'] = pd.to_numeric(df['p1_calc_score'], errors='coerce')
        df['p2_calc_score'] = pd.to_numeric(df['p2_calc_score'], errors='coerce')
        # Keep only rows where BOTH scores are valid numbers for comparison
        df_scores = df.dropna(subset=['p1_calc_score', 'p2_calc_score']).copy()
        if df_scores.empty: print("Error: No valid score data found in CSV after cleaning. Cannot generate chart."); return
        print(f"Initial valid score rows found: {len(df_scores)}")

        # --- <<< NEW: Outlier Filtering Logic >>> ---
        # Calculate previous and next score values using shift
        df_scores['p1_prev'] = df_scores['p1_calc_score'].shift(1)
        df_scores['p1_next'] = df_scores['p1_calc_score'].shift(-1)
        df_scores['p2_prev'] = df_scores['p2_calc_score'].shift(1)
        df_scores['p2_next'] = df_scores['p2_calc_score'].shift(-1)

        # Identify outliers for Player 1: Previous and next are the same, but current is different
        p1_outlier_condition = (
            (df_scores['p1_prev'] == df_scores['p1_next']) &
            (df_scores['p1_calc_score'] != df_scores['p1_prev']) # or != p1_next
        )
        # Identify outliers for Player 2: Previous and next are the same, but current is different
        p2_outlier_condition = (
            (df_scores['p2_prev'] == df_scores['p2_next']) &
            (df_scores['p2_calc_score'] != df_scores['p2_prev']) # or != p2_next
        )

        # Combine conditions: a row is an outlier if EITHER player's score is an outlier in that row
        is_outlier = p1_outlier_condition | p2_outlier_condition

        # Keep only the rows that are NOT outliers
        df_filtered = df_scores[~is_outlier].copy() # Use ~ for negation
        num_removed = len(df_scores) - len(df_filtered)
        if num_removed > 0:
            print(f"Removed {num_removed} single-frame outlier score reading(s).")
        else:
            print("No single-frame outliers detected for removal.")
        # --- <<< End of Outlier Filtering Logic >>> ---

        # --- Sampling and Plotting (using df_filtered) ---
        num_valid_scores = len(df_filtered); plot_label_suffix = ""; df_sampled = pd.DataFrame()
        if num_valid_scores == 0: print("Error: Zero valid scores remaining after filtering."); return
        elif num_valid_scores < plot_all_threshold:
            print(f"Found {num_valid_scores} valid score points after filtering (<{plot_all_threshold}). Plotting all points.")
            df_sampled = df_filtered # Plot all filtered points
            plot_label_suffix = " (All Filtered Points)"
        else:
            if sample_rate <= 0: sample_rate = 1
            # Sample from the filtered data
            df_sampled = df_filtered.iloc[::sample_rate, :]
            # Ensure at least first/last points if sampling drastically reduces points
            if len(df_sampled) < 2 and num_valid_scores >= 2 :
                 print(f"Warning: Only {len(df_sampled)} points remain after sample rate {sample_rate} on filtered data. Plotting first/last filtered points.")
                 # Ensure we take from the filtered dataframe
                 df_sampled = df_filtered.iloc[[0, -1]]
                 plot_label_suffix = " (First/Last Filtered Points)"
            elif not df_sampled.empty:
                plot_label_suffix = f" (Sampled approx. every {sample_rate} valid filtered reads)"

        if df_sampled.empty: print("Error: No data points selected for plotting after filtering/sampling. Skipping chart."); return
        print(f"Plotting {len(df_sampled)} data points for score chart.")

        plt.figure(figsize=(15, 7))
        # Plot using the original score columns from the sampled data
        plt.plot(df_sampled['frame'], df_sampled['p1_calc_score'], label='Player 1 Score', marker='.', linestyle='-', markersize=4)
        plt.plot(df_sampled['frame'], df_sampled['p2_calc_score'], label='Player 2 Score', marker='.', linestyle='-', markersize=4)
        plt.xlabel(f"Frame Number{plot_label_suffix}"); plt.ylabel("Calculated Score (Games*10 + Points)"); plt.title("Calculated Table Tennis Score Over Time (Outliers Filtered)")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(output_chart_path, dpi=150); print(f"Score chart saved successfully to '{output_chart_path}'"); plt.close()

    except Exception as e:
        print(f"Error generating score chart: {e}")

# --- Main Processing Function ---
def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame, model_name, min_x_separation_factor, ocr_reader, p1_roi, p2_roi, ocr_interval=OCR_FRAME_INTERVAL):
    """ Processes video for player positions and scores (periodic OCR), saves data and annotated video. """
    if not os.path.exists(input_path): print(f"Error: Input video file not found at '{input_path}'"); return None
    ocr_is_active = isinstance(ocr_reader, easyocr.Reader)
    if not ocr_is_active and OCR_ENABLED: print("Warning: OCR was enabled but Reader is not available.")

    print(f"Loading YOLO model: {model_name}...")
    try: model = YOLO(model_name); print("Model loaded successfully.")
    except Exception as e: print(f"Error loading model '{model_name}': {e}"); return None

    print(f"Opening video file: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): print(f"Error: Could not open video file {input_path}"); return None
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0: print("Warning: Could not determine video FPS. Assuming 30 FPS."); fps = 30
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); min_x_pixel_distance=frame_width*min_x_separation_factor; ocr_frame_interval = max(1, ocr_interval)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")
    print(f"Filtering: Prioritizing 2/3 detections | Min X-Separation > {min_x_pixel_distance:.0f}px")
    if ocr_is_active: print(f"Score ROIs: P1={p1_roi}, P2={p2_roi}"); print(f"Running score OCR check every {ocr_frame_interval} frames.")
    else: print("Score OCR reader not active.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened(): print(f"Error: Could not open video writer for '{output_video_path}'"); cap.release(); return None
    print(f"Output video (annotated valid frames) will be saved to: {output_video_path}")

    # --- Corrected CSV Opening Block ---
    csv_file, csv_writer = None, None
    try:
        csv_file = open(output_csv_path, 'w', newline=''); csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y', 'p1_games_read', 'p1_points_read', 'p1_calc_score', 'p2_games_read', 'p2_points_read', 'p2_calc_score'])
        print(f"Output CSV (positions and scores) will be saved to: {output_csv_path}")
    except IOError as e:
        print(f"Error opening CSV file '{output_csv_path}': {e}")
        cap.release()
        if out_video.isOpened(): out_video.release()
        return None
    if csv_writer is None:
        print(f"Error: CSV writer could not be initialized.")
        cap.release()
        if out_video.isOpened(): out_video.release()
        if csv_file and not csv_file.closed: csv_file.close()
        return None
    # --- End Corrected CSV Opening Block ---

    frame_count, processed_frame_count, valid_player_frame_count, ocr_success_count = 0, 0, 0, 0
    start_time = time.time(); actual_end_frame = end_frame if end_frame > 0 and end_frame <= total_frames else total_frames
    print(f"Processing frames from {start_frame} to {actual_end_frame - 1}...")
    last_p1_score_str, last_p2_score_str = "P1: - -", "P2: - -"
    # saved_roi_count, max_debug_rois = 0, 10 # Keep commented out if not needed

    while cap.isOpened(): # Main loop
        if frame_count >= actual_end_frame: print(f"\nReached target end frame: {actual_end_frame}"); break
        success, frame = cap.read()
        if not success: print("\nEnd of video stream or error reading frame."); break

        if frame_count >= start_frame:
            processed_frame_count += 1; results = model(frame, verbose=False, classes=[0])
            all_detected_persons = []
            if results[0].boxes is not None:
                for box in results[0].boxes.xyxy.cpu().numpy(): all_detected_persons.append({'center_x':(box[0]+box[2])/2, 'center_y':(box[1]+box[3])/2})
            player1_pos, player2_pos = select_players(all_detected_persons, frame_width, min_x_separation_factor)
            is_valid_player_frame = bool(player1_pos['x'] and player2_pos['x'])

            p1_games_csv,p1_points_csv,p1_calc_csv = '','',''; p2_games_csv,p2_points_csv,p2_calc_csv = '','',''
            current_p1_str, current_p2_str = last_p1_score_str, last_p2_score_str

            if ocr_is_active and frame_count % ocr_frame_interval == 0: # Periodic OCR Check
                x1_p1,y1_p1,x2_p1,y2_p1=p1_roi; x1_p2,y1_p2,x2_p2,y2_p2=p2_roi
                y1_p1,y2_p1=max(0,y1_p1),min(frame_height,y2_p1); x1_p1,x2_p1=max(0,x1_p1),min(frame_width,x2_p1)
                y1_p2,y2_p2=max(0,y1_p2),min(frame_height,y2_p2); x1_p2,x2_p2=max(0,x1_p2),min(frame_width,x2_p2)

                # --- (Save Debug ROIs - Currently Commented Out) ---
                # if is_valid_player_frame and saved_roi_count < max_debug_rois:
                #     try:
                #        ... (code to save ROIs) ...
                #        saved_roi_count += 1
                #     except Exception as e_write: print(f"  Warning: Could not write debug ROI: {e_write}")
                # --- End Debug ROI Save ---

                processed_roi1,processed_roi2=None,None; thresh_val=200; threshold_type=cv2.THRESH_BINARY # Preprocessing setup

                if y2_p1 > y1_p1 and x2_p1 > x1_p1: roi1_img=frame[y1_p1:y2_p1,x1_p1:x2_p1]; gray1=cv2.cvtColor(roi1_img,cv2.COLOR_BGR2GRAY); ret1,processed_roi1=cv2.threshold(gray1,thresh_val,255,threshold_type)
                if y2_p2 > y1_p2 and x2_p2 > x1_p2: roi2_img=frame[y1_p2:y2_p2,x1_p2:x2_p2]; gray2=cv2.cvtColor(roi2_img,cv2.COLOR_BGR2GRAY); ret2,processed_roi2=cv2.threshold(gray2,thresh_val,255,threshold_type)

                ocr1_results,ocr2_results = [],[]; allow_list = '0123456789 ' # Run OCR with allowlist
                if processed_roi1 is not None:
                    try: ocr1_results = ocr_reader.readtext(processed_roi1, allowlist=allow_list)
                    except Exception as e_ocr: print(f"  Warning: OCR ROI1 failed: {e_ocr}")
                if processed_roi2 is not None:
                    try: ocr2_results = ocr_reader.readtext(processed_roi2, allowlist=allow_list)
                    except Exception as e_ocr: print(f"  Warning: OCR ROI2 failed: {e_ocr}")

                p1_games,p1_points=parse_score_text(ocr1_results); p2_games,p2_points=parse_score_text(ocr2_results) # Parse
                if p1_games is not None and p2_games is not None: # Check success
                    ocr_success_count+=1; p1_calc=p1_games*10+p1_points; p2_calc=p2_games*10+p2_points
                    p1_games_csv,p1_points_csv,p1_calc_csv=p1_games,p1_points,p1_calc # Assign for CSV
                    p2_games_csv,p2_points_csv,p2_calc_csv=p2_games,p2_points,p2_calc # Assign for CSV
                    current_p1_str=f"P1: {p1_games} - {p1_points}"; current_p2_str=f"P2: {p2_games} - {p2_points}" # Update display string
                    last_p1_score_str=current_p1_str; last_p2_score_str=current_p2_str # Update last known good

            csv_writer.writerow([ frame_count, player1_pos['x'] if is_valid_player_frame else '', player1_pos['y'] if is_valid_player_frame else '', player2_pos['x'] if is_valid_player_frame else '', player2_pos['y'] if is_valid_player_frame else '', p1_games_csv, p1_points_csv, p1_calc_csv, p2_games_csv, p2_points_csv, p2_calc_csv ]) # Write CSV row

            if is_valid_player_frame: # Annotate and write video frame
                valid_player_frame_count += 1; annotated_frame = results[0].plot()
                p1_coords=(player1_pos['x'],player1_pos['y']); p2_coords=(player2_pos['x'],player2_pos['y'])
                cv2.circle(annotated_frame,p1_coords,8,(0,255,0),-1); cv2.putText(annotated_frame,'P1',(p1_coords[0]+10,p1_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)
                cv2.circle(annotated_frame,p2_coords,8,(0,0,255),-1); cv2.putText(annotated_frame,'P2',(p2_coords[0]+10,p2_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
                if ocr_is_active: # Overlay score
                    score_display_y,score_display_x=50,20; font_scale,font_thickness,font_color=1.2,3,(255,255,0)
                    cv2.putText(annotated_frame,current_p1_str,(score_display_x,score_display_y),cv2.FONT_HERSHEY_SIMPLEX,font_scale,font_color,font_thickness)
                    cv2.putText(annotated_frame,current_p2_str,(score_display_x,score_display_y+40),cv2.FONT_HERSHEY_SIMPLEX,font_scale,font_color,font_thickness)
                out_video.write(annotated_frame)

            if processed_frame_count % 100 == 0 and processed_frame_count > 0: # Progress update
                 elapsed_time = time.time() - start_time; fps_current = processed_frame_count / elapsed_time if elapsed_time > 0 else 0
                 num_ocr_checks_so_far = (processed_frame_count // ocr_frame_interval) if ocr_frame_interval > 0 else processed_frame_count
                 ocr_rate = (ocr_success_count / num_ocr_checks_so_far * 100) if num_ocr_checks_so_far > 0 else 0
                 print(f"  Frame {frame_count} (Processed: {processed_frame_count}, Players OK: {valid_player_frame_count}, OCR OK: {ocr_success_count} [{ocr_rate:.1f}% per check], FPS: {fps_current:.2f})")

        frame_count += 1
    # --- End Video Processing Loop ---

    # --- Final Summary and Cleanup ---
    end_time = time.time(); processing_time = end_time - start_time; avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---"); print(f"Frame range processed: {start_frame} to {frame_count - 1}"); print(f"Total frames analyzed: {processed_frame_count}")
    print(f"Valid player frames written: {valid_player_frame_count}"); print(f"Successful score reads (periodic checks): {ocr_success_count}")
    valid_percentage=(valid_player_frame_count/processed_frame_count*100) if processed_frame_count > 0 else 0
    num_ocr_checks = processed_frame_count // ocr_frame_interval if ocr_frame_interval > 0 else 0
    ocr_overall_success_rate=(ocr_success_count/num_ocr_checks*100) if num_ocr_checks > 0 else 0
    print(f"Percentage valid player frames: {valid_percentage:.2f}%"); print(f"Percentage successful score reads (of checks): {ocr_overall_success_rate:.2f}%")
    print(f"Processing time: {processing_time:.2f} seconds"); print(f"Average processing FPS: {avg_fps:.2f}")
    cap.release();
    if out_video.isOpened(): out_video.release()
    if csv_file and not csv_file.closed: csv_file.close()
    cv2.destroyAllWindows()
    print(f"\nVideo processing complete."); print(f"Output video: '{output_video_path}'"); print(f"Output CSV: '{output_csv_path}'")
    return frame_width, frame_height

# --- Script Execution ---
if __name__ == "__main__":
    print("Starting Table Tennis Analyzer (with Periodic Score OCR)...")
    dimensions = process_video(
        INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_CSV,
        START_FRAME, END_FRAME, MODEL_NAME,
        MIN_X_SEPARATION_FACTOR,
        reader,                 # Pass EasyOCR reader
        ROI_PLAYER1_SCORE,      # Pass player 1 ROI
        ROI_PLAYER2_SCORE,      # Pass player 2 ROI
        ocr_interval=OCR_FRAME_INTERVAL # Pass the interval
    )

    # --- Post-processing ---
    if dimensions:
        print("\n--- Starting Post-Processing ---")
        if os.path.exists(CSV_PATH_FOR_HEATMAP) and os.path.getsize(CSV_PATH_FOR_HEATMAP) > 0:
            create_heatmap(csv_path=CSV_PATH_FOR_HEATMAP, video_path=VIDEO_PATH_FOR_HEATMAP, output_image_path=OUTPUT_HEATMAP_PATH)
        else: print(f"\nWarning: Data CSV file '{CSV_PATH_FOR_HEATMAP}' is missing or empty. Skipping heatmap generation.")

        if OCR_ENABLED:
             if os.path.exists(CSV_PATH_FOR_HEATMAP) and os.path.getsize(CSV_PATH_FOR_HEATMAP) > 0:
                create_score_chart(csv_path=CSV_PATH_FOR_HEATMAP, output_chart_path=OUTPUT_SCORE_CHART_PATH, sample_rate=CHART_FRAME_SAMPLE_RATE, plot_all_threshold=CHART_PLOT_ALL_THRESHOLD)
             else: print(f"\nWarning: Data CSV file '{CSV_PATH_FOR_HEATMAP}' is missing or empty. Skipping score chart generation.")
        else: print("\nSkipping score chart generation as OCR was disabled.")
    else: print("\nVideo processing failed or returned no dimensions. Skipping post-processing.")

    print("\nScript finished.")
    print(f"Reminder: OCR is now checked every {OCR_FRAME_INTERVAL} frames. Adjust if needed.")
    print("          Check/Adjust OCR Preprocessing threshold (thresh_val) if accuracy is low.")