# Уменьшение размера изображений

def resize_images(folder, scale):

    import os
    import cv2
    import numpy as np
    
    if not os.path.exists(folder + '\\Spectral_Cube_original'):
        os.rename(folder + '\\Spectral_Cube',  folder + '\\Spectral_Cube_original')
    
    # Paths
    input_dir = folder + "\\Spectral_Cube_original"
    output_dir = folder + "\\Spectral_Cube"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    extensions = (".png", ".jpg", ".jpeg")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read image
            with open(input_path, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8)

            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            #img = cv2.imread(input_path)
            
            if img is None:
                print(f"Warning: could not read {filename}")
                print(input_path)
                continue

            # Original size
            h, w = img.shape[:2]

            # New size
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # BGR → Grayscale
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

            # Гарантируем 8 бит (обычно уже uint8, но на всякий случай)
            gray = gray.astype(np.uint8)

            # Save resized image
            # cv2.imwrite(output_path, resized_img)
            
            # Get file extension (e.g. .png or .jpg)
            ext = os.path.splitext(output_path)[1]

            # Encode image to memory buffer
            success, encoded_img = cv2.imencode(ext, gray)

            if not success:
                raise RuntimeError("Image encoding failed")

            # Write buffer to file (Unicode-safe)
            with open(output_path, "wb") as f:
                f.write(encoded_img.tobytes())
            
    return(input_dir, output_dir)

if __name__ == '__main__':
    
    for folder in folder_list:
        resize_images(folder=main_folder + '\\' + folder, scale=0.5)
