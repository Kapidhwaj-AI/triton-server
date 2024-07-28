import json

import dlib
import face_recognition_models
from face_recognition.api import _raw_face_landmarks
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):
        
        self.model_config = model_config = json.loads(args["model_config"])

        face_recognition_model = face_recognition_models.face_recognition_model_location()
        self.face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

        embeddings_config = pb_utils.get_output_config_by_name(
            model_config, "embeddings"
        )

        self.embeddings_dtype = pb_utils.triton_string_to_numpy(
            embeddings_config["data_type"]
        )


    def execute(self, requests):

        responses = []
        for request in requests:

            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            face_locations_tensor = pb_utils.get_input_tensor_by_name(request, "face_locations")
            frame = image_tensor.as_numpy()
            face_locations = face_locations_tensor.as_numpy()

            embeddings = self.get_batch_encodings(frame, face_locations)

            embeddings_tensor = pb_utils.Tensor(
                "embeddings", np.array(embeddings).astype(self.embeddings_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    embeddings_tensor
                ]
            )
            responses.append(inference_response)

        return responses
    

    def get_batch_encodings(self, face_image, known_face_locations, num_jitters=1, model="small"):
        try:
            print("method = get_batch_encodings, status = started")
            dlib_vector = dlib.full_object_detections()
            raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
            dlib_vector.extend(raw_landmarks)
            embeddings = np.array(self.face_encoder.compute_face_descriptor(face_image, dlib_vector, num_jitters))
            print("method = get_batch_encodings, status = completed")
            return embeddings
        except Exception as e:
            print(e)
            return np.array([])
