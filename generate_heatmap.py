import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter

csv_path = "player_positions.csv"
video_frame_path = "input.mp4"
output_image = "player_position_heatmap.png"
resolution = (1920, 1080)

def generate_player_heatmap(csv_file, output_image, resolution, video_frame_path):
    heatmap_data = np.zeros((resolution[1], resolution[0])) # Initialize heatmap with zeros

    try:
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader) # Skip header row
            for i, row in enumerate(reader):
                # Ensure correct number of columns is read (frame, p1x, p1y, p2x, p2y)
                if len(row) >= 5:
                    try:
                        # Convert position strings to integers
                        # Ensure values are within integer range if they might be large
                        _, p1x, p1y, p2x, p2y = map(int, row[:5])

                        # Clamp coordinates to be within the resolution bounds
                        p1x = max(0, min(p1x, resolution[0] - 1))
                        p1y = max(0, min(p1y, resolution[1] - 1))
                        p2x = max(0, min(p2x, resolution[0] - 1))
                        p2y = max(0, min(p2y, resolution[1] - 1))

                        # Increment the count for each player's position
                        heatmap_data[p1y, p1x] += 1
                        heatmap_data[p2y, p2x] += 1
                    except ValueError as e:
                        print(f"Skipping row {i+2} due to invalid value: {row} - {e}")
                    except IndexError as e:
                         print(f"Skipping row {i+2} due to incorrect number of columns: {row} - {e}")
                else:
                    print(f"Skipping row {i+2} due to incorrect format (expected at least 5 columns): {row}")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return


    print(f"Max heatmap value (pre-filter): {np.max(heatmap_data)}")
    print(f"Non-zero values (pre-filter): {np.count_nonzero(heatmap_data)}")

    sigma_value = 40
    heatmap_data = gaussian_filter(heatmap_data, sigma=sigma_value)

    # Get the maximum value after filtering
    max_filtered_value = np.max(heatmap_data)
    print(f"Max filtered heatmap value: {max_filtered_value}")

    # Load the video frame to use as background
    cap = cv2.VideoCapture(video_frame_path)
    ret, frame = cap.read()
    cap.release()

    # If frame loading failed, create a white background image
    if not ret or frame is None:
        print(f"Warning: Could not load video frame from {video_frame_path}. Using a white background.")
        frame = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
    else:
        # Ensure frame is the correct size and convert from BGR to RGB
        frame = cv2.resize(frame, resolution)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mask_threshold = 0.01
    mask = np.ma.masked_where(heatmap_data < mask_threshold, heatmap_data)

    plt.figure(figsize=(12, 7))
    plt.imshow(frame) # Display the video frame (or background)

    heatmap_vmin = mask_threshold # Start heatmap visualization from the mask threshold
    heatmap_vmax = max_filtered_value # The maximum value determines the top of the color scale

    # Ensure vmin is less than or equal to vmax to prevent ValueError
    if heatmap_vmax < heatmap_vmin:
        print("Warning: Max filtered heatmap value is extremely low or equal to mask threshold. Heatmap may not be visible or show little variation.")
        heatmap_vmax = heatmap_vmin if heatmap_vmax < heatmap_vmin else heatmap_vmax

    plt.imshow(mask, cmap='jet', interpolation='bilinear', alpha=0.6, vmin=0, vmax=max_filtered_value)
    plt.colorbar(label='Density')
    plt.title('Combined Player Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    try:
        plt.savefig(output_image, bbox_inches='tight')
        print(f"Heatmap saved to: {output_image}")
    except Exception as e:
        print(f"Error saving the figure: {e}")


    plt.show() # Display the plot window

if __name__ == "__main__":
    generate_player_heatmap(csv_path, output_image, resolution, video_frame_path)