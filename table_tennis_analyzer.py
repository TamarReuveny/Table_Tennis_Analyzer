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

st.set_page_config(layout="wide", page_title="Table Tennis Analyzer")

# --- Constants for Video Processing ---
# DEFAULT VALUES - User can override via Streamlit interface
DEFAULT_START_FRAME = 0
DEFAULT_END_FRAME = 0
# INPUT_VIDEO will come from Streamlit's file uploader
# OUTPUT_VIDEO, OUTPUT_CSV etc will be defined later based on input/fixed names
MIN_X_SEPARATION_FACTOR = 0.10

# --- Constants for Score OCR ---
ROI_PLAYER1_SCORE = (445, 926, 557, 977) # Player 1 (Top) ROI (x1, y1, x2, y2)
ROI_PLAYER2_SCORE = (445, 974, 557, 1025) # Player 2 (Bottom) ROI (x1, y1, x2, y2)
DEFAULT_OCR_INTERVAL = 5

# --- Model Selection ---
MODEL_NAME = 'yolo11n-pose.pt'

# --- Constants for Heatmap ---
OUTPUT_HEATMAP_FILENAME = "streamlit_heatmap.png" # Fixed output filename
HEATMAP_BINS = 30
HEATMAP_ALPHA = 0.7
HEATMAP_CMAP = 'jet'
INTERPOLATION = 'gaussian'
VMAX_PERCENTILE = 95

# --- Constants for Score Chart ---
OUTPUT_SCORE_CHART_FILENAME = "streamlit_score_chart.png" # Fixed output filename
CHART_FRAME_SAMPLE_RATE = 5
CHART_PLOT_ALL_THRESHOLD = 100

# --- Initialize EasyOCR Reader (runs once when script starts) ---
# Use @st.cache_resource to initialize the reader only once
@st.cache_resource
def load_easyocr_reader():
    """Loads the EasyOCR reader, returns the reader instance or None."""
    reader_instance = None
    ocr_init_success = False
    try:
        print("Initializing EasyOCR Reader (this may take a moment)...")
        # Consider gpu=True if you have a compatible GPU and setup
        reader_instance = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR Reader initialized successfully.")
        ocr_init_success = True
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("Score recognition will be disabled.")
        st.error(f"EasyOCR initialization failed: {e}. Score recognition disabled.", icon="🚨")
    return reader_instance, ocr_init_success

reader, OCR_ENABLED = load_easyocr_reader()

# --- Core Analysis Functions (Mostly unchanged, removed status_callback) ---
# --- Added optional progress bar update to process_video ---

def create_heatmap(csv_path, video_path, output_image_path, bins=HEATMAP_BINS, alpha=HEATMAP_ALPHA, cmap_name=HEATMAP_CMAP, interpolation=INTERPOLATION, vmax_percentile=VMAX_PERCENTILE):
    """Generates a heatmap overlay on a video frame using player position data."""
    print(f"\n--- Heatmap Generation ---"); print(f"CSV: {csv_path}, Video: {video_path}, Output: {output_image_path}"); print(f"Params: bins={bins}, alpha={alpha}, cmap='{cmap_name}', interp='{interpolation}', vmax_pctl={vmax_percentile}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return False
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return False
        pos_cols = ['player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y']
        if not all(col in df.columns for col in pos_cols): print(f"Error: One or more position columns missing in '{csv_path}'."); return False
        for col in pos_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df_clean = df.dropna(subset=pos_cols)
        if not df_clean.empty: all_x = pd.concat([df_clean['player1_position_x'], df_clean['player2_position_x']], ignore_index=True); all_y = pd.concat([df_clean['player1_position_y'], df_clean['player2_position_y']], ignore_index=True)
        else: all_x, all_y = pd.Series(dtype=float), pd.Series(dtype=float)
        if all_x.empty or all_y.empty: print(f"Warning: No valid numeric coordinates found after cleaning CSV '{csv_path}'. Heatmap might be empty.")
        else: print(f"Found {len(all_x)} valid coordinate pairs for heatmap.")
    except Exception as e: print(f"Error reading/processing CSV '{csv_path}' for heatmap: {e}"); return False
    if not os.path.exists(video_path): print(f"Error: Video not found '{video_path}'"); return False
    cap = cv2.VideoCapture(video_path);
    if not cap.isOpened(): print(f"Error: Could not open video '{video_path}'"); return False
    ret, frame = cap.read()
    if not ret: print(f"Error: Could not read frame from '{video_path}'"); cap.release(); return False
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
    except Exception as e: print(f"Error saving heatmap image: {e}"); return False
    plt.close(fig)
    return True # Indicate success

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
        selected_players.sort(key=lambda p: p['center_x']) # Sort by X: Left is P1, Right is P2
        player1_pos['x'],player1_pos['y'] = int(selected_players[0]['center_x']),int(selected_players[0]['center_y'])
        player2_pos['x'],player2_pos['y'] = int(selected_players[1]['center_x']),int(selected_players[1]['center_y'])
    return player1_pos, player2_pos

def parse_score_text(ocr_results):
    """
    Parses EasyOCR results from a score ROI.
    Tries to find two numbers (Games Points).
    If only one number is found, assumes it's Games Won and Points are 0.
    Returns (games_won, current_points) or (None, None).
    """
    full_text = " ".join([res[1] for res in ocr_results]).strip()
    # Don't print in streamlit app unless necessary for debugging
    # if full_text: print(f"    Attempting to parse OCR text: '{full_text}'")

    match = re.search(r'(\d+)\D*(\d+)', full_text)
    if match:
        try: return int(match.group(1)), int(match.group(2))
        except: pass

    numbers = re.findall(r'\d+', full_text)
    if len(numbers) == 1:
        try: return int(numbers[0]), 0
        except: pass

    return None, None

def create_score_chart(csv_path, output_chart_path, sample_rate=CHART_FRAME_SAMPLE_RATE, plot_all_threshold=CHART_PLOT_ALL_THRESHOLD):
    """Creates a line chart of calculated scores over time from the CSV, filtering out single-frame outliers."""
    print(f"\n--- Score Chart Generation ---")
    print(f"CSV: {csv_path}, Output: {output_chart_path}, SampleRate: {sample_rate}, PlotAllThreshold: {plot_all_threshold}")
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0: print(f"Error: CSV file '{csv_path}' not found or is empty."); return False
        df = pd.read_csv(csv_path);
        if df.empty: print(f"Error: CSV file '{csv_path}' contains no data rows."); return False
        score_cols = ['p1_calc_score', 'p2_calc_score', 'frame']
        if not all(col in df.columns for col in score_cols): print(f"Error: Required score columns {score_cols} missing in '{csv_path}'."); return False

        df['p1_calc_score'] = pd.to_numeric(df['p1_calc_score'], errors='coerce')
        df['p2_calc_score'] = pd.to_numeric(df['p2_calc_score'], errors='coerce')
        df_scores = df.dropna(subset=['p1_calc_score', 'p2_calc_score']).copy()
        if df_scores.empty: print("Error: No valid score data found in CSV after cleaning. Cannot generate chart."); return False
        print(f"Initial valid score rows found: {len(df_scores)}")

        # Outlier Filtering Logic
        df_scores['p1_prev'] = df_scores['p1_calc_score'].shift(1); df_scores['p1_next'] = df_scores['p1_calc_score'].shift(-1)
        df_scores['p2_prev'] = df_scores['p2_calc_score'].shift(1); df_scores['p2_next'] = df_scores['p2_calc_score'].shift(-1)
        p1_outlier_condition = (df_scores['p1_prev'] == df_scores['p1_next']) & (df_scores['p1_calc_score'] != df_scores['p1_prev'])
        p2_outlier_condition = (df_scores['p2_prev'] == df_scores['p2_next']) & (df_scores['p2_calc_score'] != df_scores['p2_prev'])
        is_outlier = p1_outlier_condition | p2_outlier_condition
        df_filtered = df_scores[~is_outlier].copy()
        num_removed = len(df_scores) - len(df_filtered)
        if num_removed > 0: print(f"Removed {num_removed} single-frame outlier score reading(s).")
        else: print("No single-frame outliers detected for removal.")

        # Sampling and Plotting
        num_valid_scores = len(df_filtered); plot_label_suffix = ""; df_sampled = pd.DataFrame()
        if num_valid_scores == 0: print("Error: Zero valid scores remaining after filtering."); return False
        elif num_valid_scores < plot_all_threshold:
            print(f"Found {num_valid_scores} valid score points after filtering (<{plot_all_threshold}). Plotting all points.")
            df_sampled = df_filtered; plot_label_suffix = " (All Filtered Points)"
        else:
            if sample_rate <= 0: sample_rate = 1
            df_sampled = df_filtered.iloc[::sample_rate, :]
            if len(df_sampled) < 2 and num_valid_scores >= 2 :
                 print(f"Warning: Only {len(df_sampled)} points remain after sample rate {sample_rate}. Plotting first/last filtered points.")
                 df_sampled = df_filtered.iloc[[0, -1]]; plot_label_suffix = " (First/Last Filtered Points)"
            elif not df_sampled.empty:
                plot_label_suffix = f" (Sampled approx. every {sample_rate} valid filtered reads)"

        if df_sampled.empty: print("Error: No data points selected for plotting after filtering/sampling. Skipping chart."); return False
        print(f"Plotting {len(df_sampled)} data points for score chart.")

        plt.figure(figsize=(15, 7))
        plt.plot(df_sampled['frame'], df_sampled['p1_calc_score'], label='Player 1 Score', marker='.', linestyle='-', markersize=4)
        plt.plot(df_sampled['frame'], df_sampled['p2_calc_score'], label='Player 2 Score', marker='.', linestyle='-', markersize=4)
        plt.xlabel(f"Frame Number{plot_label_suffix}"); plt.ylabel("Calculated Score (Games*10 + Points)"); plt.title("Calculated Table Tennis Score Over Time (Outliers Filtered)")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(output_chart_path, dpi=150); print(f"Score chart saved successfully to '{output_chart_path}'"); plt.close()
        return True # Indicate success

    except Exception as e:
        print(f"Error generating score chart: {e}")
        return False

def process_video(input_path, output_video_path, output_csv_path, start_frame, end_frame,
                  model_name, min_x_separation_factor,
                  ocr_reader_instance, p1_roi, p2_roi, ocr_interval,
                  progress_bar=None): # Optional Streamlit progress bar
    """
    Processes video for player positions and scores (periodic OCR), saves data and annotated video.
    Updates the optional Streamlit progress bar.
    Returns video dimensions (width, height) on success, None on failure.
    """
    if not os.path.exists(input_path):
        st.error(f"Input video file not found: {input_path}")
        return None

    ocr_is_active = isinstance(ocr_reader_instance, easyocr.Reader)
    if not ocr_is_active:
         print("OCR Reader not available. Score recognition disabled.")
         # Don't show error in streamlit if checkbox was unchecked intentionally

    print(f"Loading YOLO model: {model_name}...")
    try:
        model = YOLO(model_name)
        print("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading YOLO model '{model_name}': {e}")
        return None

    print(f"Opening video file: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {input_path}")
        return None

    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0: print("Warning: Could not determine video FPS. Assuming 30 FPS."); fps = 30
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); min_x_pixel_distance=frame_width*min_x_separation_factor; ocr_frame_interval = max(1, ocr_interval)
    print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out_video.isOpened():
        st.error(f"Error opening video writer for '{output_video_path}'")
        cap.release(); return None
    print(f"Output video will be saved to: {output_video_path}")

    try:
        with open(output_csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y', 'p1_games_read', 'p1_points_read', 'p1_calc_score', 'p2_games_read', 'p2_points_read', 'p2_calc_score'])
            print(f"Output CSV will be saved to: {output_csv_path}")

            frame_count, processed_frame_count, valid_player_frame_count, ocr_success_count = 0, 0, 0, 0
            start_time = time.time(); actual_end_frame = end_frame if end_frame > 0 and end_frame <= total_frames else total_frames
            print(f"Processing frames from {start_frame} to {actual_end_frame - 1}...")
            last_p1_score_str, last_p2_score_str = "P1: - -", "P2: - -"
            
            # --- Main Processing Loop ---
            while cap.isOpened():
                if frame_count >= actual_end_frame: break
                success, frame = cap.read()
                if not success: break

                if frame_count >= start_frame:
                    processed_frame_count += 1
                    results = model(frame, verbose=False, classes=[0])
                    all_detected_persons = []
                    if results[0].boxes is not None:
                        for box in results[0].boxes.xyxy.cpu().numpy(): all_detected_persons.append({'center_x':(box[0]+box[2])/2, 'center_y':(box[1]+box[3])/2})
                    player1_pos, player2_pos = select_players(all_detected_persons, frame_width, min_x_separation_factor)
                    is_valid_player_frame = bool(player1_pos['x'] and player2_pos['x'])

                    p1_games_csv, p1_points_csv, p1_calc_csv = '', '', ''
                    p2_games_csv, p2_points_csv, p2_calc_csv = '', '', ''
                    current_p1_str, current_p2_str = last_p1_score_str, last_p2_score_str

                    if ocr_is_active and frame_count % ocr_frame_interval == 0:
                        x1_p1,y1_p1,x2_p1,y2_p1=p1_roi; x1_p2,y1_p2,x2_p2,y2_p2=p2_roi
                        y1_p1,y2_p1=max(0,y1_p1),min(frame_height,y2_p1); x1_p1,x2_p1=max(0,x1_p1),min(frame_width,x2_p1)
                        y1_p2,y2_p2=max(0,y1_p2),min(frame_height,y2_p2); x1_p2,x2_p2=max(0,x1_p2),min(frame_width,x2_p2)
                        processed_roi1, processed_roi2 = None, None
                        thresh_val=200; threshold_type=cv2.THRESH_BINARY

                        if y2_p1 > y1_p1 and x2_p1 > x1_p1:
                           roi1_img=frame[y1_p1:y2_p1,x1_p1:x2_p1]; gray1=cv2.cvtColor(roi1_img,cv2.COLOR_BGR2GRAY); _,processed_roi1=cv2.threshold(gray1,thresh_val,255,threshold_type)
                        if y2_p2 > y1_p2 and x2_p2 > x1_p2:
                           roi2_img=frame[y1_p2:y2_p2,x1_p2:x2_p2]; gray2=cv2.cvtColor(roi2_img,cv2.COLOR_BGR2GRAY); _,processed_roi2=cv2.threshold(gray2,thresh_val,255,threshold_type)

                        ocr1_results, ocr2_results = [], []
                        allow_list = '0123456789 '
                        if processed_roi1 is not None:
                            try: ocr1_results = ocr_reader_instance.readtext(processed_roi1, allowlist=allow_list)
                            except Exception as e_ocr: print(f"  Warning: OCR ROI1 failed: {e_ocr}")
                        if processed_roi2 is not None:
                            try: ocr2_results = ocr_reader_instance.readtext(processed_roi2, allowlist=allow_list)
                            except Exception as e_ocr: print(f"  Warning: OCR ROI2 failed: {e_ocr}")

                        p1_games, p1_points = parse_score_text(ocr1_results)
                        p2_games, p2_points = parse_score_text(ocr2_results)
                        if p1_games is not None and p2_games is not None:
                            ocr_success_count += 1
                            p1_calc = p1_games * 10 + p1_points; p2_calc = p2_games * 10 + p2_points
                            p1_games_csv, p1_points_csv, p1_calc_csv = p1_games, p1_points, p1_calc
                            p2_games_csv, p2_points_csv, p2_calc_csv = p2_games, p2_points, p2_calc
                            current_p1_str = f"P1: {p1_games} - {p1_points}"
                            current_p2_str = f"P2: {p2_games} - {p2_points}"
                            last_p1_score_str = current_p1_str; last_p2_score_str = current_p2_str

                    csv_writer.writerow([ frame_count, player1_pos['x'] if is_valid_player_frame else '', player1_pos['y'] if is_valid_player_frame else '', player2_pos['x'] if is_valid_player_frame else '', player2_pos['y'] if is_valid_player_frame else '', p1_games_csv, p1_points_csv, p1_calc_csv, p2_games_csv, p2_points_csv, p2_calc_csv ])

                    if is_valid_player_frame:
                        valid_player_frame_count += 1
                        annotated_frame = results[0].plot()
                        p1_coords=(player1_pos['x'],player1_pos['y']); p2_coords=(player2_pos['x'],player2_pos['y'])
                        cv2.circle(annotated_frame,p1_coords,8,(0,255,0),-1); cv2.putText(annotated_frame,'P1',(p1_coords[0]+10,p1_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)
                        cv2.circle(annotated_frame,p2_coords,8,(0,0,255),-1); cv2.putText(annotated_frame,'P2',(p2_coords[0]+10,p2_coords[1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)
                        if ocr_is_active:
                            score_display_y, score_display_x = 50, 20; font_scale, font_thickness, font_color = 1.2, 3, (255, 255, 0)
                            cv2.putText(annotated_frame, current_p1_str, (score_display_x, score_display_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                            cv2.putText(annotated_frame, current_p2_str, (score_display_x, score_display_y + 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)
                        out_video.write(annotated_frame)

                    # --- Update Progress Bar ---
                    if progress_bar is not None and actual_end_frame > start_frame:
                        # Update every few frames to avoid too many updates
                        if processed_frame_count % 10 == 0:
                            progress_value = (frame_count - start_frame) / (actual_end_frame - start_frame)
                            progress_bar.progress(min(progress_value, 1.0), text=f"Processing Frame {frame_count}/{actual_end_frame}")

                frame_count += 1
            # --- End Loop ---
            # Final progress bar update
            if progress_bar is not None:
                 progress_bar.progress(1.0, text="Video Processing Complete")

    except IOError as e:
        st.error(f"Error accessing CSV file '{output_csv_path}': {e}")
        cap.release(); out_video.release(); cv2.destroyAllWindows()
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during video processing: {e}")
        import traceback
        traceback.print_exc()
        cap.release(); out_video.release(); cv2.destroyAllWindows()
        return None

    # --- Final Summary and Cleanup ---
    end_time = time.time(); processing_time = end_time - start_time; avg_fps = processed_frame_count / processing_time if processing_time > 0 else 0
    print("\n--- Processing Summary ---"); print(f"Frame range processed: {start_frame} to {frame_count - 1}"); print(f"Total frames analyzed: {processed_frame_count}")
    print(f"Valid player frames written: {valid_player_frame_count}"); print(f"Successful score reads (periodic checks): {ocr_success_count}")
    # ... (rest of summary printing) ...
    print(f"Processing time: {processing_time:.2f} seconds"); print(f"Average processing FPS: {avg_fps:.2f}")

    cap.release(); out_video.release(); cv2.destroyAllWindows()
    print("\nVideo processing finished.")
    return frame_width, frame_height


# --- Streamlit App Implementation ---

st.title("🏓 Table Tennis Analyzer")
st.write("Upload a video file to detect players, read scores (optional), and generate analysis outputs.")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_video_path = tmp_file.name # Path to the temporary file
        print(f"Uploaded file saved temporarily to: {input_video_path}")

    st.success(f"Successfully uploaded: **{uploaded_file.name}**")
    st.video(input_video_path) # Show the uploaded video

    # --- Parameters Setup (in Sidebar) ---
    with st.sidebar:
        st.header("⚙️ Analysis Parameters")

        start_f = st.number_input("Start Frame", value=DEFAULT_START_FRAME, min_value=0, step=1,
                                  help="Frame number to start analysis from (0 for beginning).")
        end_f = st.number_input("End Frame (0 = End)", value=DEFAULT_END_FRAME, min_value=0, step=1,
                                help="Frame number to end analysis (0 processes until the end of the video).")
        ocr_int = st.number_input("OCR Check Interval (frames)", value=DEFAULT_OCR_INTERVAL, min_value=1, step=1,
                                  help="Check score using OCR every N frames.")

        # Disable OCR checkbox if reader failed to load
        ocr_tooltip = "Enable automatic score reading using EasyOCR." if reader else "EasyOCR failed to initialize, OCR disabled."
        use_ocr = st.checkbox("Enable Score Recognition (OCR)", value=OCR_ENABLED, disabled=(reader is None),
                              help=ocr_tooltip)

        st.divider()
        run_button_clicked = st.button("▶️ Run Analysis", type="primary", use_container_width=True)

    # --- Analysis Execution and Results Display ---
    if run_button_clicked:
        st.info("Analysis started... Please wait. Processing time depends on video length and parameters.", icon="⏳")

        # Define output file paths (use fixed names for simplicity in Streamlit)
        output_base = os.path.splitext(uploaded_file.name)[0] # Use original base name for outputs
        output_video_path = f"{output_base}_analyzed.mp4"
        output_csv_path = f"{output_base}_data.csv"
        output_heatmap_path = OUTPUT_HEATMAP_FILENAME # Use fixed name defined above
        output_score_chart_path = OUTPUT_SCORE_CHART_FILENAME # Use fixed name

        # Placeholder for progress bar
        progress_bar = st.progress(0, text="Starting...")

        # --- Run Video Processing ---
        try:
             dimensions = process_video(
                 input_path=input_video_path,
                 output_video_path=output_video_path,
                 output_csv_path=output_csv_path,
                 start_frame=start_f,
                 end_frame=end_f,
                 model_name=MODEL_NAME,
                 min_x_separation_factor=MIN_X_SEPARATION_FACTOR,
                 ocr_reader_instance=reader, # Pass the loaded reader
                 p1_roi=ROI_PLAYER1_SCORE,
                 p2_roi=ROI_PLAYER2_SCORE,
                 ocr_interval=ocr_int,
                 progress_bar=progress_bar # Pass the progress bar object
             )
        except Exception as e:
             st.error(f"An error occurred during analysis: {e}")
             import traceback
             st.text_area("Traceback", traceback.format_exc(), height=200)
             dimensions = None # Ensure analysis stops

        # --- Run Post-processing and Display Results ---
        if dimensions:
            st.success("Video processing finished!", icon="✅")

            heatmap_success = False
            score_chart_success = False

            # Heatmap
            with st.spinner("Generating heatmap..."):
                heatmap_success = create_heatmap(
                    csv_path=output_csv_path,
                    video_path=input_video_path, # Use temp path for background frame
                    output_image_path=output_heatmap_path
                )
            if heatmap_success: st.success("Heatmap generated!", icon="🗺️")
            else: st.warning("Heatmap generation failed or was skipped.", icon="⚠️")

            # Score Chart (if OCR was enabled)
            if use_ocr:
                with st.spinner("Generating score chart..."):
                     score_chart_success = create_score_chart(
                         csv_path=output_csv_path,
                         output_chart_path=output_score_chart_path
                     )
                if score_chart_success: st.success("Score chart generated!", icon="📊")
                else: st.warning("Score chart generation failed or was skipped.", icon="⚠️")
            else:
                st.info("Score chart generation skipped as OCR was disabled.")

            # Display Results Section
            st.divider()
            st.header("📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                if heatmap_success and os.path.exists(output_heatmap_path):
                    st.image(output_heatmap_path, caption="Player Position Heatmap")
                    with open(output_heatmap_path, "rb") as file:
                        st.download_button(
                             label="Download Heatmap",
                             data=file,
                             file_name=os.path.basename(output_heatmap_path), # Use the actual saved filename
                             mime="image/png"
                         )
                else:
                    st.write("Heatmap not available.")

            with col2:
                 if score_chart_success and os.path.exists(output_score_chart_path):
                    st.image(output_score_chart_path, caption="Score Chart")
                    with open(output_score_chart_path, "rb") as file:
                         st.download_button(
                              label="Download Score Chart",
                              data=file,
                              file_name=os.path.basename(output_score_chart_path), # Use the actual saved filename
                              mime="image/png"
                          )
                 elif use_ocr:
                    st.write("Score Chart not available.")
                 else:
                    st.write("Score Chart skipped (OCR disabled).")

            st.divider()
            st.subheader("Download Processed Files")

            # Provide download buttons for video and CSV
            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name=os.path.basename(output_video_path), # Use the actual saved filename
                        mime="video/mp4"
                    )
            else:
                 st.warning("Processed video file not found.", icon="⚠️")

            if os.path.exists(output_csv_path):
                 with open(output_csv_path, "rb") as file:
                     st.download_button(
                         label="Download Player Data (CSV)",
                         data=file,
                         file_name=os.path.basename(output_csv_path), # Use the actual saved filename
                         mime="text/csv"
                     )
            else:
                 st.warning("CSV data file not found.", icon="⚠️")


        else:
            st.error("Analysis could not be completed due to errors during video processing.", icon="❌")

        # Clean up the temporary uploaded file after processing
        if os.path.exists(input_video_path):
             os.unlink(input_video_path)
             print(f"Temporary file deleted: {input_video_path}")

else:
    st.info("⬆️ Upload a video file using the button above to start the analysis.")

st.sidebar.divider()
st.sidebar.caption("Developed with Streamlit")