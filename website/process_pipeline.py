from .utils import extract_embedding
import os
import io
from datetime import datetime
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from mtcnn import MTCNN

# ##### Create an instance of the ImageProcessor class #####
# from process_pipeline import ImageProcessor
# processor = ImageProcessor()

# ##### Process the image #####
# image_path = "path/to/image.jpg"
# face = processor.process_image(image_path)

# ##### Save the processed image #####
# output_directory = "path/to/output/directory"
# filename = "processed_face.jpg"
# processor.save_image(face, output_directory, filename)

# ##### Process the dataset #####
# input_directory = "path/to/input/directory"
# output_directory = "path/to/output/directory"
# processor.process_dataset(input_directory, output_directory)


class ImageProcessor:

    def __init__(self):
        self.face_detector = MTCNN(min_face_size=20, 
                              scale_factor=0.709, 
                              steps_threshold=[0.6, 0.7, 0.7])


    def process_image(self, image_input):
        """
        Process a single image to detect, crop, align, and resize the face.

        Args:
            image_input (str/FileStorage): The path to the image file OR a FileStorage object.

        Returns:
            numpy.ndarray: The processed face as a numpy array, or None if no face could be detected or aligned.

        """
        # Check if the input is a file path (string) or a FileStorage object
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Unable to load image: {image_input}")
        else:
            image = Image.open(image_input)
            image = np.array(image)

        # Convert the image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        detection_results = self.face_detector.detect_faces(image)
        if len(detection_results) == 0:
            print(f"No face detected in image {image_input}")
            return None

        # Crop the face from the image
        x, y, width, height = detection_results[0]['box']
        face = image[y:y+height, x:x+width]

        # Detect facial landmarks in the face
        landmarks = self.face_detector.detect_faces(face)
        if len(landmarks) == 0:
            print(f"Unable to align face in image {image_input}")
            return None

        # Calculate the center and angle for rotation
        left_eye = landmarks[0]['keypoints']['left_eye']
        right_eye = landmarks[0]['keypoints']['right_eye']
        center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Rotate the face to align the eyes horizontally
        aligned_face = cv2.warpAffine(face, rotation_matrix, (face.shape[1], face.shape[0]), flags=cv2.INTER_CUBIC)

        # Calculate the scaling factor and resize the face
        face_height, face_width, _ = aligned_face.shape
        if face_height > face_width:
            scaling_factor = 160 / face_height
        else:
            scaling_factor = 160 / face_width
        resized_face = cv2.resize(aligned_face, None, fx=scaling_factor, fy=scaling_factor)

         # Pad the resized face to make it square
        padded_face = np.zeros((160, 160, 3), dtype=np.uint8)
        x_offset = (padded_face.shape[1] - resized_face.shape[1]) // 2
        y_offset = (padded_face.shape[0] - resized_face.shape[0]) // 2
        padded_face[y_offset:y_offset+resized_face.shape[0], x_offset:x_offset+resized_face.shape[1]] = resized_face

        return padded_face



    def save_image(self, image, output_directory, filename):
        # Take a np.array image and save it to the output directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        face_filename = os.path.join(output_directory, filename)
        cv2.imwrite(face_filename, image)

        print(f"Aligned face has been saved in the '{output_directory}' directory as {filename}.")
        print(f"Path to saved new image: {output_directory}\\{filename}")



    def process_dataset(self, input_directory, output_directory):
        # Iterate over all directories and files in the input directory
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    person_name = os.path.basename(os.path.dirname(image_path))
                    person_output_dir = os.path.join(output_directory, person_name)
                    if not os.path.exists(person_output_dir):
                        os.makedirs(person_output_dir)

                    count = len(os.listdir(person_output_dir))
                    filename = f"{person_name}_{count:02}.jpg"
                    face_filename = os.path.join(person_output_dir, filename)
                    # Process the image and save it if it hasn't been processed yet
                    if not os.path.exists(face_filename):
                        image = self.process_image(image_path)
                        if image is not None:
                            self.save_image(image, person_output_dir, filename)

#################################################################################

    def process_user_uploaded_image(self, uploaded_image, processor, output_directory):
        uploaded_image_processed = processor.process_image(image_input=uploaded_image)
        if uploaded_image_processed is None:
            return None
        pil_image = Image.fromarray((uploaded_image_processed).astype(np.uint8))
        pil_image = pil_image.convert("RGB")
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format="JPEG")
        filename = datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"
        processor.save_image(uploaded_image_processed, output_directory, filename)
        return output_directory + "/" + filename

    def process_video(self, image, processor, output_directory):
        image_processed = processor.process_image(image_input=image)
        if image_processed is None:
            return None
        pil_image = Image.fromarray((image_processed).astype(np.uint8))
        pil_image = pil_image.convert("RGB")
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format="JPEG")
        filename = "screenshot_processed.jpg"
        processor.save_image(image_processed, output_directory, filename)
        return output_directory + "/" + filename


    def extract_user_uploaded_embeddings(self, image_path, embedder):
        uploaded_image_embeddings = extract_embedding(image_path, embedder)
        feature_names = [str(i) for i in range(0, 512)]
        df_embeddings = pd.DataFrame(uploaded_image_embeddings)
        df_embeddings.columns = feature_names
        return df_embeddings

    
    def make_prediction(self, df_embeddings, top_model):
        prediction = top_model.predict(df_embeddings)
        prediction = prediction[0]
        df_embeddings['prediction'] = prediction
        return prediction, df_embeddings
