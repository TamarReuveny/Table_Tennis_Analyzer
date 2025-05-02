# table_tennis_analyzer_streamlit.py
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
import streamlit as st # Import Streamlit
import tempfile # To handle temporary file for uploaded video
import traceback # For detailed error logging

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(layout="wide", page_title="Table Tennis Analyzer")

# --- Constants for Video Processing ---
DEFAULT_START_FRAME = 0
DEFAULT_END_FRAME = 0
MIN_X_SEPARATION_FACTOR = 0.10

# --- Constants for Score OCR (Combined Method Only) ---
ROI_SCORE_AREA = (441, 929, 560, 1024) # Combined area (x1, y1, x2, y2)
OCR_MAG_RATIO = 4                     # Default Magnification ratio
DEFAULT_OCR_INTERVAL = 5

# --- Model Selection ---
MODEL_NAME = 'yolo11n-pose.pt' # Using nano pose model

# --- Constants for Heatmap ---
OUTPUT_HEATMAP_FILENAME = "streamlit_heatmap.png"
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Constants for Score Chart ---
OUTPUT_SCORE_CHART_FILENAME = "streamlit_score_chart.png"
CHART_FRAME_SAMPLE_RATE = 5
CHART_PLOT_ALL_THRESHOLD = 100

# --- Initialize Session State Variables ---
# This ensures these variables persist across reruns triggered by widget interactions
if 'analysis_complete' not in st.session_state:
    st.session_state['analysis_complete'] = False
if 'heatmap_path' not in st.session_state:
    st.session_state['heatmap_path'] = None
if 'score_chart_path' not in st.session_state:
    st.session_state['score_chart_path'] = None
if 'processed_video_path' not in st.session_state:
    st.session_state['processed_video_path'] = None
if 'csv_path' not in st.session_state:
    st.session_state['csv_path'] = None
if 'last_uploaded_id' not in st.session_state: # Stores ID of the last successfully processed file
    st.session_state['last_uploaded_id'] = None
if 'ocr_was_enabled' not in st.session_state: # Store if OCR was enabled for the last run
    st.session_state['ocr_was_enabled'] = False


def preprocess_image_for_ocr(image):
    """Applies preprocessing (grayscale, Otsu's thresholding) to an image ROI for OCR."""
    # import cv2 # Already imported globally
    if image is None: return None
    if len(image.shape) == 2: gray = image # Already grayscale
    elif len(image.shape) == 3 and image.shape[2] == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: print("Warning: Unexpected image format in preprocess_image_for_ocr"); return None
    try:
        # Apply inverted binary thresholding with Otsu's method
        thresh_val, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception as e_thresh: print(f"Error during thresholding: {e_thresh}"); return None
    return thresh_img

# --- Initialize EasyOCR Reader (runs once when script starts) ---
@st.cache_resource # Cache the reader object
def load_easyocr_reader():
    """Loads the EasyOCR reader, returns the reader instance or None."""
    reader_instance = None
    ocr_init_success = False
    try:
        print("Initializing EasyOCR Reader...")
        reader_instance = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR Reader initialized successfully.")
        ocr_init_success = True
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
    return reader_instance, ocr_init_success

reader, OCR_ENABLED = load_easyocr_reader()
# Display error only once if it failed
if not OCR_ENABLED and 'ocr_error_shown' not in st.session_state:
    st.error("EasyOCR initialization failed. Score recognition will be disabled.", icon="🚨")
    st.session_state['ocr_error_shown'] = True


# --- Core Analysis Functions ---

def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """Generates a heatmap overlay on a video frame using player position data."""
    print(f"\n--- Heatmap Generation ---"); print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return False
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return False
        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        if not all(col in df.columns for col in pos_cols): print(f"Error: One or more position columns missing in '{csv_path}'."); return False
        df[pos_cols] = df[pos_cols].apply(pd.to_numeric, errors='coerce') # Convert relevant columns
        df_clean = df.dropna(subset=pos_cols)
        if not df_clean.empty: all_x = pd.concat([df_clean['player1_position_x'], df_clean['player2_position_x']], ignore_index=True); all_y = pd.concat([df_clean['player1_position_y'], df_clean['player2_position_y']], ignore_index=True)
        else: all_x, all_y = pd.Series(dtype=float), pd.Series(dtype=float)
        if all_x.empty or all_y.empty: print(f"Warning: No valid numeric coordinates found after cleaning CSV '{csv_path}'."); return False
        else: print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")
    except Exception as e: print(f"Error reading/processing CSV '{csv_path}' for heatmap: {e}"); return False
    if not os.path.exists(video_path): print(f"Error: Video not found for heatmap background '{video_path}'"); return False
    cap = cv2.VideoCapture(video_path);
    if not cap.isOpened(): print(f"Error: Could not open video '{video_path}' for heatmap background"); return False
    ret, frame = cap.read()
    if not ret: print(f"Error: Could not read frame from '{video_path}' for heatmap background"); cap.release(); return False
    frame_height, frame_width, _ = frame.shape; background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); cap.release()
    print(f"Video dimensions (W x H): {frame_width} x {frame_height}")
    if all_x.empty or all_y.empty:
        print("Skipping histogram calculation due to no valid coordinates."); heatmap_data = np.zeros((bins, bins)); xedges = np.linspace(0, frame_width, bins + 1); yedges = np.linspace(0, frame_height, bins + 1)
    else:
        try:
            heatmap_data, xedges, yedges = np.histogram2d(all_x, all_y, bins=bins, range=[[0, frame_width], [0, frame_height]])
            heatmap_data = heatmap_data.T; print(f"Histogram shape: {heatmap_data.shape}, Max value: {heatmap_data.max():.2f}")
            if heatmap_data.max() == 0: print("Warning: Histogram data is all zeros.")
        except Exception as e: print(f"Error during np.histogram2d calculation: {e}"); return False
    non_zero_data = heatmap_data[heatmap_data > 0]; calculated_vmax = 1.0
    if non_zero_data.size > 0: calculated_vmax = np.percentile(non_zero_data, vmax_percentile); calculated_vmax = max(calculated_vmax, 1.0); print(f"Using {vmax_percentile}th percentile vmax for color scale: {calculated_vmax:.2f}")
    else: print("Warning: No non-zero density data for vmax calculation. Setting vmax=1.0")
    try: current_cmap = plt.get_cmap(cmap_name).copy(); cmap_list = current_cmap(np.linspace(0, 1, current_cmap.N)); cmap_list[0, 3] = 0.0; transparent_cmap = mcolors.ListedColormap(cmap_list)
    except Exception as e: print(f"Error modifying colormap '{cmap_name}': {e}"); transparent_cmap = plt.get_cmap(cmap_name)
    fig, ax = plt.subplots(1, 1, figsize=(12, 7)); ax.imshow(background_image, extent=[0, frame_width, frame_height, 0], aspect='auto')
    if heatmap_data.max() > 0:
        heatmap_plot = ax.imshow(heatmap_data, cmap=transparent_cmap, alpha=alpha, extent=[xedges[0], xedges[-1], yedges[-1], yedges[0]], origin='upper', interpolation=interpolation, aspect='auto', vmin=0, vmax=calculated_vmax)
        extend_direction = 'max' if calculated_vmax < heatmap_data.max() else 'neither'; cbar = fig.colorbar(heatmap_plot, ax=ax, extend=extend_direction); cbar.set_label('Position Density')
    else: print("Skipping heatmap overlay as data is empty or all zeros.")
    ax.set_title('Player Position Heatmap'); ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_xticks([]); ax.set_yticks([]); plt.tight_layout()
    try: plt.savefig(output_image_path, dpi=150, bbox_inches='tight'); print(f"Heatmap image saved successfully to '{output_image_path}'")
    except Exception as e: print(f"Error saving heatmap image: {e}"); return False
    plt.close(fig)
    return True

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

def parse_combined_score_area(ocr_results_list):
    """
    Parses EasyOCR flat list results like ['0', '6', '0', '4'].
    Assumes order is P1_Games, P1_Points, P2_Games, P2_Points.
    Includes fallback for lists with 2 elements (assumes points only).
    Returns (p1_games, p1_points, p2_games, p2_points) or Nones if parsing fails.
    """
    p1_games, p1_points, p2_games, p2_points = None, None, None, None
    print(f"    Attempting to parse flat list: {ocr_results_list}") # Debug input

    if isinstance(ocr_results_list, list): # Check if input is a list
        num_elements = len(ocr_results_list)

        # --- Primary Logic: Expect 4 numbers ---
        if num_elements == 4:
            try:
                p1_games = int(ocr_results_list[0])
                p1_points = int(ocr_results_list[1])
                p2_games = int(ocr_results_list[2])
                p2_points = int(ocr_results_list[3])
                print(f"    Parsed 4 numbers: P1G={p1_games}, P1P={p1_points}, P2G={p2_games}, P2P={p2_points}")
                return p1_games, p1_points, p2_games, p2_points
            except (ValueError, IndexError) as e:
                print(f"    Error converting 4 numbers to int: {e}")
                return None, None, None, None

        # --- Fallback Logic: Expect 2 numbers (Assume points, games=0) ---
        elif num_elements == 2:
            try:
                p1_games = int(ocr_results_list[0])
                p1_points = 0
                p2_games = int(ocr_results_list[1])
                p2_points = 0
                print(f"    Fallback: Parsed 2 numbers as GAMES: P1={p1_games}-{p1_points}, P2={p2_games}-{p2_points}")
                return p1_games, p1_points, p2_games, p2_points
            except (ValueError, IndexError) as e:
                print(f"    Error converting 2 numbers (fallback): {e}")
                return None, None, None, None

        # --- Fallback Logic: Expect 3 numbers (e.g., '0','6','04' - try parsing last element) ---
        elif num_elements == 3:
            try:
                # Try to parse the last element for two digits
                numbers_last = re.findall(r'\d+', ocr_results_list[2])
                if len(numbers_last) >= 2:
                    p1_games = int(ocr_results_list[0])
                    p1_points = int(ocr_results_list[1])
                    p2_games = int(numbers_last[0])
                    p2_points = int(numbers_last[1])
                    print(f"    Fallback: Parsed 3 elements like G1, P1, G2P2: P1={p1_games}-{p1_points}, P2={p2_games}-{p2_points}")
                    return p1_games, p1_points, p2_games, p2_points
                else: # If last element has only one digit, try common G1, P1, P2 format
                     p1_games = int(ocr_results_list[0])
                     p1_points = int(ocr_results_list[1])
                     p2_games = int(ocr_results_list[2])
                     p2_points = 0 # Assume 0 points for P2
                     print(f"    Fallback: Parsed 3 elements like G1, P1, G2: P1={p1_games}-{p1_points}, P2={p2_games}-{p2_points}")
                     return p1_games, p1_points, p2_games, p2_points
            except (ValueError, IndexError) as e:
                 print(f"    Error converting 3 elements (fallback): {e}")
                 return None, None, None, None

        else:
             print(f"    Parse failed: Unexpected number of elements found ({num_elements}). List: {ocr_results_list}")
             return None, None, None, None
    else:
         print(f"    Parse failed: Input was not a list ({type(ocr_results_list)}).")
         return None, None, None, None

def create_score_chart(csv_path, output_chart_path, sample_rate=CHART_FRAME_SAMPLE_RATE, plot_all_threshold=CHART_PLOT_ALL_THRESHOLD):
    """Creates a line chart of calculated scores over time from the CSV, filtering out single-frame outliers."""
    print(f"\n--- Score Chart Generation ---"); print(f"CSV: {csv_path}, Output: {output_chart_path}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return False
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return False
        score_cols = ['p1_calc_score', 'p2_calc_score', 'frame']
        if not all(col in df.columns for col in score_cols): print(f"Error: Required score columns {score_cols} missing."); return False
        df['p1_calc_score'] = pd.to_numeric(df['p1_calc_score'], errors='coerce')
        df['p2_calc_score'] = pd.to_numeric(df['p2_calc_score'], errors='coerce')
        df_scores = df.dropna(subset=['p1_calc_score', 'p2_calc_score']).copy()
        if df_scores.empty: print("Error: No valid score data found."); return False
        print(f"Initial valid score rows: {len(df_scores)}")
        # Outlier Filtering
        df_scores['p1_prev'] = df_scores['p1_calc_score'].shift(1); df_scores['p1_next'] = df_scores['p1_calc_score'].shift(-1)
        df_scores['p2_prev'] = df_scores['p2_calc_score'].shift(1); df_scores['p2_next'] = df_scores['p2_calc_score'].shift(-1)
        p1_outlier = (df_scores['p1_prev'] == df_scores['p1_next']) & (df_scores['p1_calc_score'] != df_scores['p1_prev'])
        p2_outlier = (df_scores['p2_prev'] == df_scores['p2_next']) & (df_scores['p2_calc_score'] != df_scores['p2_prev'])
        df_filtered = df_scores[~(p1_outlier | p2_outlier)].copy()
        num_removed = len(df_scores) - len(df_filtered)
        if num_removed > 0: print(f"Removed {num_removed} single-frame outlier(s).")
        else: print("No single-frame outliers removed.")
        # Sampling
        num_valid = len(df_filtered); suffix = ""; df_sampled = pd.DataFrame()
        if num_valid == 0: print("Error: Zero valid scores after filtering."); return False
        elif num_valid < plot_all_threshold: df_sampled, suffix = df_filtered, " (All Filtered Points)"
        else:
            sample_rate = max(1, sample_rate)
            df_sampled = df_filtered.iloc[::sample_rate, :]
            if len(df_sampled) < 2 and num_valid >= 2 : df_sampled, suffix = df_filtered.iloc[[0, -1]], " (First/Last Filtered Points)"
            elif not df_sampled.empty: suffix = f" (Sampled ~every {sample_rate} reads)"
        if df_sampled.empty: print("Error: No points selected for plotting."); return False
        print(f"Plotting {len(df_sampled)} points.")
        # Plotting
        plt.figure(figsize=(15, 7))
        plt.plot(df_sampled['frame'], df_sampled['p1_calc_score'], label='Player 1 Score', marker='.', linestyle='-', markersize=4)
        plt.plot(df_sampled['frame'], df_sampled['p2_calc_score'], label='Player 2 Score', marker='.', linestyle='-', markersize=4)
        plt.xlabel(f"Frame Number{suffix}"); plt.ylabel("Calculated Score (Games*10 + Points)"); plt.title("Calculated Score Over Time (Outliers Filtered)")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(output_chart_path, dpi=150); print(f"Score chart saved to '{output_chart_path}'"); plt.close()
        return True
    except FileNotFoundError: print(f"Error: CSV '{csv_path}' not found."); return False
    except Exception as e: print(f"Error generating score chart: {e}"); return False


# --- !! MODIFIED !! process_video Function Signature (Combined ROI Only) ---
def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame,
                  model_name, min_x_separation_factor,
                  ocr_reader_instance, ocr_interval,
                  # Takes combined ROI and mag ratio directly
                  score_roi, ocr_mag_ratio,
                  progress_bar=None):
    """ Processes video using ONLY the COMBINED SCORE ROI method. """
    if not os.path.exists(input_path): st.error(f"Input video file not found: {input_path}"); return None

    ocr_is_active = isinstance(ocr_reader_instance, easyocr.Reader)
    if ocr_is_active: print("OCR Reader is active for this run.")
    else: print("OCR Reader not available. Score recognition disabled.")

    print(f"Loading YOLO model: {model_name}...")
    try: model = YOLO(model_name); print("Model loaded successfully.")
    except Exception as e: st.error(f"Error loading YOLO model '{model_name}': {e}"); return None

    print(f"Opening video file: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): st.error(f"Error opening video file: {input_path}"); return None

    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0: print("Warning: Could not determine video FPS. Assuming 30 FPS."); fps = 30
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); min_x_pixel_distance=frame_width*min_x_separation_factor; ocr_frame_interval = max(1, ocr_interval)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened(): st.error(f"Error opening video writer for '{output_video_path}'"); cap.release(); return None
    print(f"Output video will be saved to: {output_video_path}")

    try:
        with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y', 'p1_games_read', 'p1_points_read', 'p1_calc_score', 'p2_games_read', 'p2_points_read', 'p2_calc_score'])
            print(f"Output CSV will be saved to: {output_csv_path}")

            frame_count, processed_frame_count, valid_player_frame_count, ocr_success_count = 0, 0, 0, 0
            start_time = time.time();
            actual_end_frame = total_frames
            if end_frame > 0 and end_frame < total_frames: actual_end_frame = end_frame + 1
            start_frame = min(start_frame, actual_end_frame - 1) if actual_end_frame > 0 else 0
            print(f"Processing frames from {start_frame} to {actual_end_frame - 1}...")
            last_p1_score_str, last_p2_score_str = "P1: - -", "P2: - -"
            total_frames_to_process = actual_end_frame - start_frame

            # --- Main Processing Loop ---
            while cap.isOpened():
                if frame_count >= actual_end_frame: break
                success, frame = cap.read()
                if not success: break

                if frame_count >= start_frame:
                    processed_frame_count += 1
                    results = model(frame, verbose=False, classes=[0]) # Detect only persons
                    all_detected_persons = [{'center_x': (box[0] + box[2]) / 2, 'center_y': (box[1] + box[3]) / 2} for box in results[0].boxes.xyxy.cpu().numpy()] if results[0].boxes is not None else []
                    player1_pos, player2_pos = select_players(all_detected_persons, frame_width, min_x_separation_factor)
                    is_valid_player_frame = bool(player1_pos['x'] and player2_pos['x'])

                    p1_games_csv, p1_points_csv, p1_calc_csv = '', '', ''
                    p2_games_csv, p2_points_csv, p2_calc_csv = '', '', ''
                    current_p1_str, current_p2_str = last_p1_score_str, last_p2_score_str

                    # --- Periodic OCR Check (Combined Area Method ONLY) ---
                    if ocr_is_active and score_roi is not None and frame_count % ocr_frame_interval == 0:
                        x1, y1, x2, y2 = score_roi
                        y1, y2 = max(0, y1), min(frame_height, y2)
                        x1, x2 = max(0, x1), min(frame_width, x2)

                        processed_score_roi = None
                        if y2 > y1 and x2 > x1:
                            score_roi_img = frame[y1:y2, x1:x2]
                            processed_score_roi = preprocess_image_for_ocr(score_roi_img)

                        ocr_results_list = []
                        if processed_score_roi is not None:
                            try:
                                ocr_results_list = ocr_reader_instance.readtext(
                                    processed_score_roi, allowlist='0123456789 ', detail=0, mag_ratio=ocr_mag_ratio)
                                print(f"  [Frame {frame_count}] Raw OCR Result (Combined): {ocr_results_list}")
                            except Exception as e_ocr: print(f"  Warning: OCR Combined ROI failed: {e_ocr}")

                        p1_games, p1_points, p2_games, p2_points = parse_combined_score_area(ocr_results_list)

                        if p1_games is not None and p2_games is not None:
                            print(f"  [Frame {frame_count}] Combined OCR Parse Success: P1={p1_games}-{p1_points}, P2={p2_games}-{p2_points}")
                            ocr_success_count += 1
                            p1_calc = p1_games * 10 + p1_points; p2_calc = p2_games * 10 + p2_points
                            p1_games_csv, p1_points_csv, p1_calc_csv = p1_games, p1_points, p1_calc
                            p2_games_csv, p2_points_csv, p2_calc_csv = p2_games, p2_points, p2_calc
                            current_p1_str = f"P1: {p1_games} - {p1_points}"
                            current_p2_str = f"P2: {p2_games} - {p2_points}"
                            last_p1_score_str = current_p1_str; last_p2_score_str = current_p2_str
                        else:
                            print(f"  [Frame {frame_count}] Combined OCR Parse Failed.")
                            p1_games_csv, p1_points_csv, p1_calc_csv = '', '', ''
                            p2_games_csv, p2_points_csv, p2_calc_csv = '', '', ''
                    # --- End of OCR Check Block ---

                    csv_writer.writerow([ frame_count, player1_pos['x'] if is_valid_player_frame else '', player1_pos['y'] if is_valid_player_frame else '', player2_pos['x'] if is_valid_player_frame else '', player2_pos['y'] if is_valid_player_frame else '', p1_games_csv, p1_points_csv, p1_calc_csv, p2_games_csv, p2_points_csv, p2_calc_csv ])

                    if is_valid_player_frame:
                        valid_player_frame_count += 1
                        annotated_frame = results[0].plot()
                        p1_coords=(player1_pos['x'],player1_pos['y']); p2_coords=(player2_pos['x'],player2_pos['y'])
                        cv2.circle(annotated_frame,p1_coords,8,(0,255,0),-1); cv2.putText(annotated_frame,'P1',(p1_coords[0]+10,p1_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)
                        cv2.circle(annotated_frame,p2_coords,8,(0,0,255),-1); cv2.putText(annotated_frame,'P2',(p2_coords[0]+10,p2_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
                        if ocr_is_active: # Use last known good score strings
                            score_display_y, score_display_x = 50, 20; font_scale, font_thickness, font_color = 1.2, 3, (255, 255, 0)
                            cv2.putText(annotated_frame, current_p1_str, (score_display_x, score_display_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                            cv2.putText(annotated_frame, current_p2_str, (score_display_x, score_display_y + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                        out_video.write(annotated_frame)

                    # --- Update Progress Bar ---
                    if progress_bar is not None and total_frames_to_process > 0:
                        progress_value = processed_frame_count / total_frames_to_process
                        if processed_frame_count % 20 == 0 or frame_count == actual_end_frame - 1 :
                             progress_bar.progress(min(progress_value, 1.0), text=f"Processing Frame {frame_count}/{actual_end_frame - 1}")

                frame_count += 1
            # --- End Loop ---

    except IOError as e:
        st.error(f"Error accessing CSV file '{output_csv_path}': {e}")
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out_video' in locals() and out_video.isOpened(): out_video.release()
        cv2.destroyAllWindows(); return None
    except Exception as e:
        st.error(f"An unexpected error occurred during video processing: {e}")
        st.text_area("Traceback", traceback.format_exc(), height=200)
        if 'cap' in locals() and cap.isOpened(): cap.release()
        if 'out_video' in locals() and out_video.isOpened(): out_video.release()
        cv2.destroyAllWindows(); return None

    # --- Final Summary and Cleanup ---
    end_time = time.time(); processing_time = end_time - start_time; avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---"); print(f"Frame range processed: {start_frame} to {frame_count - 1}"); print(f"Total frames analyzed: {processed_frame_count}")
    print(f"Valid player frames written: {valid_player_frame_count}"); print(f"Successful score reads (periodic checks): {ocr_success_count}")
    print(f"Processing time: {processing_time:.2f} seconds"); print(f"Average processing FPS: {avg_fps:.2f}")
    cap.release(); out_video.release(); cv2.destroyAllWindows()
    print("\nVideo processing finished.")
    return frame_width, frame_height


# --- Streamlit App Implementation ---

st.title("🏓 Table Tennis Analyzer")
st.write("Upload a video file to detect players, read scores (optional), and generate analysis outputs.")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

# Define fixed output paths relative to the script directory
# Ensure output directory exists
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "output_files")
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "streamlit_output_video.mp4")
output_csv_path = os.path.join(output_dir, "streamlit_output_data.csv")
output_heatmap_path = os.path.join(output_dir, OUTPUT_HEATMAP_FILENAME)
output_score_chart_path = os.path.join(output_dir, OUTPUT_SCORE_CHART_FILENAME)

# Variable to hold temp input path
input_video_path = None

if uploaded_file is not None:
    uploaded_file_id = uploaded_file.file_id + uploaded_file.name + str(uploaded_file.size)
    if st.session_state.get('last_uploaded_id') != uploaded_file_id:
        print("New file detected or first upload, resetting analysis state.")
        st.session_state['analysis_complete'] = False
        st.session_state['heatmap_path'] = None
        st.session_state['score_chart_path'] = None
        st.session_state['processed_video_path'] = None
        st.session_state['csv_path'] = None

    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            input_video_path = tmp_file.name
            print(f"Uploaded file saved temporarily to: {input_video_path}")
        st.success(f"Successfully uploaded: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.1f} MB)")
        if not st.session_state.get('analysis_complete', False): st.video(input_video_path)
    except Exception as e_upload:
         st.error(f"Error handling uploaded file: {e_upload}")
         input_video_path = None # Ensure path is None on error

    # --- Parameters Setup (in Sidebar) ---
    # Display sidebar only if file was successfully handled
    if input_video_path:
        with st.sidebar:
            st.header("⚙️ Analysis Parameters")
            start_f = st.number_input("Start Frame", value=DEFAULT_START_FRAME, min_value=0, step=1, help="Frame number to start analysis from (0 for beginning).")
            end_f = st.number_input("End Frame (0 = End)", value=DEFAULT_END_FRAME, min_value=0, step=1, help="Frame number to end analysis (0 processes until the end).")
            st.divider()
            st.subheader("OCR Settings")
            ocr_int = st.number_input("OCR Check Interval (frames)", value=DEFAULT_OCR_INTERVAL, min_value=1, step=1, help="Check score using OCR every N frames.")
            ocr_mag_r = st.number_input("OCR Magnification Ratio", value=OCR_MAG_RATIO, min_value=1, max_value=10, step=1, help="Magnification for EasyOCR.")
            ocr_tooltip = "Enable automatic score reading using EasyOCR." if reader else "EasyOCR failed to initialize, OCR disabled."
            use_ocr = st.checkbox("Enable Score Recognition (OCR)", value=OCR_ENABLED, disabled=(reader is None), help=ocr_tooltip)
            # --- !! REMOVED !! OCR Method Radio Button ---
            st.divider()
            run_button_clicked = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

        # --- Analysis Execution ---
        if run_button_clicked:
            st.info("Analysis started... Please wait.", icon="⏳")
            st.session_state['analysis_complete'] = False
            st.session_state['ocr_was_enabled'] = use_ocr

            progress_bar_placeholder = st.empty()
            progress_bar = progress_bar_placeholder.progress(0, text="Starting...")

            try:
                # --- !! MODIFIED !! Call to process_video (Combined Method Only) ---
                process_args = {
                    "input_path": input_video_path,
                    "output_video_path": output_video_path,
                    "output_csv_path": output_csv_path,
                    "start_frame": start_f,
                    "end_frame": end_f,
                    "model_name": MODEL_NAME,
                    "min_x_separation_factor": MIN_X_SEPARATION_FACTOR,
                    "ocr_reader_instance": reader if use_ocr else None,
                    "ocr_interval": ocr_int,
                    "score_roi": ROI_SCORE_AREA,      # Always pass combined ROI
                    "ocr_mag_ratio": ocr_mag_r,       # Pass selected mag ratio
                    "progress_bar": progress_bar
                }
                dimensions = process_video(**process_args)

            except Exception as e:
                 st.error(f"An error occurred during analysis setup or call: {e}")
                 st.text_area("Traceback", traceback.format_exc(), height=200)
                 dimensions = None
            finally:
                 progress_bar_placeholder.empty()

            # --- Run Post-processing and Store Results in Session State ---
            if dimensions:
                st.success("Video processing finished!", icon="✅")
                heatmap_success = False; score_chart_success = False

                with st.spinner("Generating heatmap..."):
                    # Use the temporary input path for create_heatmap's video_path argument
                    heatmap_success = create_heatmap(output_csv_path, input_video_path, output_heatmap_path)
                if heatmap_success: st.success("Heatmap generated!", icon="🗺️")
                else: st.warning("Heatmap generation failed or skipped.", icon="⚠️")

                if use_ocr:
                    with st.spinner("Generating score chart..."):
                         score_chart_success = create_score_chart(output_csv_path, output_score_chart_path)
                    if score_chart_success: st.success("Score chart generated!", icon="📊")
                    else: st.warning("Score chart generation failed or skipped.", icon="⚠️")
                else: st.info("Score chart generation skipped as OCR was disabled.")

                st.session_state['analysis_complete'] = True
                st.session_state['heatmap_path'] = output_heatmap_path if heatmap_success and os.path.exists(output_heatmap_path) else None
                st.session_state['score_chart_path'] = output_score_chart_path if score_chart_success and os.path.exists(output_score_chart_path) else None
                st.session_state['processed_video_path'] = output_video_path if os.path.exists(output_video_path) else None
                st.session_state['csv_path'] = output_csv_path if os.path.exists(output_csv_path) else None
                st.session_state['last_uploaded_id'] = uploaded_file_id

            else:
                st.error("Analysis failed during video processing.", icon="❌")
                st.session_state['analysis_complete'] = False

            # Clean up the temporary file AFTER analysis and post-processing
            if input_video_path and os.path.exists(input_video_path):
                 try: os.unlink(input_video_path); print(f"Temp file deleted: {input_video_path}")
                 except Exception as e_unlink: print(f"Warning: Could not delete temp file {input_video_path}: {e_unlink}")

# --- Display Results Section (Checks Session State) ---
if st.session_state.get('analysis_complete', False):
    st.divider()
    st.header("📊 Analysis Results")
    col1, col2 = st.columns(2)
    # Display Heatmap
    with col1:
        heatmap_path_state = st.session_state.get('heatmap_path')
        if heatmap_path_state and os.path.exists(heatmap_path_state):
            try:
                st.image(heatmap_path_state, caption="Player Position Heatmap")
                with open(heatmap_path_state, "rb") as file: st.download_button("Download Heatmap", file, os.path.basename(heatmap_path_state), "image/png", key="dl_heatmap")
            except Exception as e_img: st.error(f"Error displaying heatmap: {e_img}")
        else: st.write("Heatmap could not be generated or found.") # More informative message
    # Display Score Chart
    with col2:
        score_chart_path_state = st.session_state.get('score_chart_path')
        ocr_enabled_last_run = st.session_state.get('ocr_was_enabled', False) # Get state from last run
        if score_chart_path_state and os.path.exists(score_chart_path_state):
            try:
                st.image(score_chart_path_state, caption="Score Chart")
                with open(score_chart_path_state, "rb") as file: st.download_button("Download Score Chart", file, os.path.basename(score_chart_path_state), "image/png", key="dl_score_chart")
            except Exception as e_img: st.error(f"Error displaying score chart: {e_img}")
        elif ocr_enabled_last_run: st.write("Score Chart could not be generated or found.") # More informative message
        # else: st.write("Score Chart skipped (OCR was disabled).") # Removed to avoid redundancy if chart path is None

    st.divider()
    st.subheader("Download Processed Files")
    # Download Processed Video
    processed_video_path_state = st.session_state.get('processed_video_path')
    if processed_video_path_state and os.path.exists(processed_video_path_state):
         try:
             with open(processed_video_path_state, "rb") as file: st.download_button("Download Processed Video", file, os.path.basename(processed_video_path_state), "video/mp4", key="dl_video")
         except Exception as e_dl: st.error(f"Error preparing video download: {e_dl}")
    else: st.warning("Processed video file not found.", icon="⚠️")
    # Download CSV Data
    csv_path_state = st.session_state.get('csv_path')
    if csv_path_state and os.path.exists(csv_path_state):
         try:
             with open(csv_path_state, "rb") as file: st.download_button("Download Player Data (CSV)", file, os.path.basename(csv_path_state), "text/csv", key="dl_csv")
         except Exception as e_dl: st.error(f"Error preparing CSV download: {e_dl}")
    else: st.warning("CSV data file not found.", icon="⚠️")

# Message if no file is uploaded OR analysis not complete
elif uploaded_file is None: # Show only if no file is ever uploaded
     st.info("⬆️ Upload a video file using the button above to start the analysis.")

# --- Sidebar Footer ---
st.sidebar.divider()
st.sidebar.caption("Analyzer App")
