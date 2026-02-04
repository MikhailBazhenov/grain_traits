# Image processing for detection objects on transillunination plate (for images exact combination)

def detect_n_objects(image_path, n, model_path = r"yolo12_wheat_grain.pt"):
        
        import numpy as np        
        import cv2
        from ultralytics import YOLO
        from sort_detection_boxes import sort_detection_boxes
        
        # load the model
        model = YOLO(model_path)
        
        img = cv2.imread(image_path)
        img_copy = np.array(img)
        results = model(image_path, verbose=False)

        detections = results[0].cpu().boxes.data.numpy()
        dim_max_array = []
        conf_array = []

        for x1, y1, x2, y2, conf, cls in detections:
                if cls in [0, 2]:
                        conf_array.append(float(conf))
                width = abs(x2 - x1)
                length = abs(y2 - y1)
                dim_max = max(width, length)
                dim_max_array.append(dim_max)

        tolerance = int(np.mean(dim_max_array))
        conf_threshold = np.mean(sorted(conf_array, reverse=True)[n - 1: n])
        detections_filtered = []

        for x1, y1, x2, y2, conf, cls in detections:
                if conf >= conf_threshold:
                        detections_filtered.append([x1, y1, x2, y2, conf, cls])
        
        detections = np.array(detections_filtered)
        detections = sort_detection_boxes(detections, tolerance)
        
        # Making picture for checking correctness of detections
        counter = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.2
        font_thickness = 5

        for x1, y1, x2, y2, conf, cls in detections:
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if cls == 0:
                        counter += 1
                        text_color = (255, 255, 0)
                        cv2.circle(img, center = (center_x, center_y), radius=15, color=(0, 255, 0), thickness=-1)
                        cv2.putText(img, str(counter), (center_x + 25, center_y), font, font_scale, text_color, font_thickness)
                elif cls == 2:
                        counter += 1
                        text_color = (255, 0, 0)
                        cv2.circle(img, center = (center_x, center_y), radius=15, color=(0, 0, 255), thickness=-1)
                        cv2.putText(img, str(counter), (center_x+25, center_y), font, font_scale, text_color, font_thickness)
                elif cls == 1 and conf > 0.6:
                        cv2.circle(img, center = (center_x, center_y), radius=15, color=(0, 255, 255), thickness=-1)
                        
        return detections, img


if __name__ == "__main__":
        image_path = r'grain_11.12.2025_B39-1-1_ostatok7.68\Spectral_Cube\image410.jpg'
        detections, image = detect_n_objects(image_path, 100)
        plt.imshow(image)
        print(detections)
