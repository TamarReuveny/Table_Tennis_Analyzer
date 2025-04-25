import cv2
from ultralytics import YOLO

def analyze_table_tennis_image(image_path):
    model = YOLO("yolov8n-pose.pt")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    results = model(img)
    annotated_img = results[0].plot()
    cv2.imshow("Detected Players and Poses", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = "table-tennis.png"
    analyze_table_tennis_image(image_file)
