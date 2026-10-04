import os
import json
import cv2
import random
from google.genai import errors
from google.genai import types
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.logger import get_logger
from src.custom_exeption import CustomExepton
load_dotenv()

logger = get_logger(__name__)

class GeminiVLM:
    """
    Gemini VLM wrapper for elderlyperson video analysis.

    Supports:
    - Single image-path analysis
    - OpenCV frame analysis
    - Multi-frame temporal analysis
    - Text-based reasoning
    - Structured JSON responses
    """

    def __init__(self, model=os.getenv("GEMINI_MODEL")):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

    # ---------------------------------------------------------
    # Analyze an image using an image file path
    # ---------------------------------------------------------
    try:
        def analyze_image(self, image_path, prompt):

            image_bytes = self._load_image(
                image_path
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=self._get_mime_type(
                            image_path
                        )
                    ),
                    prompt
                ]
            )

            logger.info(f"================Image analyzis data===============")
            logger.info(f"Image Analyze details: {self._parse_json(response.text)}")
            logger.info(f"==================================================")

            return self._parse_json(
                response.text
            )
            
    except Exception as e:
        raise CustomExepton(f"Error while analyzing the image",e)
    # ---------------------------------------------------------
    # Analyze an OpenCV frame directly
    # ---------------------------------------------------------

    def analyze_frame(self, frame, prompt):

        success, encoded_image = cv2.imencode(
            ".jpg",
            frame
        )
        # max_retries = 4
        
        if not success:
            raise ValueError(
                "Failed to encode OpenCV frame."
            )

        image_bytes = encoded_image.tobytes()

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ]
        )

        return self._parse_json(
            response.text
        )

    # ---------------------------------------------------------
    # Analyze multiple image files
    # ---------------------------------------------------------

    def analyze_frames(
        self,
        image_paths,
        prompt
    ):

        contents = [
            prompt
        ]
        try:
            for image_path in image_paths:

                image_bytes = self._load_image(
                    image_path
                )

                contents.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=self._get_mime_type(
                            image_path
                        )
                    )
                )

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents
            )

            logger.info(f"================Analyze frames===============")
            logger.info(f"Image Analyze details: {self._parse_json(response.text)}")
            logger.info(f"===============================")

            return self._parse_json(
                response.text
            )
        except Exception as e:
            raise CustomExepton(f"Error while analyzing frames",e)

    # ---------------------------------------------------------
    # Analyze text
    # ---------------------------------------------------------

    def analyze_text(self, prompt):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return self._parse_json(
            response.text
        )

    # ---------------------------------------------------------
    # Load image from disk
    # ---------------------------------------------------------

    def _load_image(self, image_path):

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with open(image_path, "rb") as f:
            return f.read()

    # ---------------------------------------------------------
    # Determine image MIME type
    # ---------------------------------------------------------

    def _get_mime_type(self, image_path):

        extension = os.path.splitext(
            image_path
        )[1].lower()

        if extension in [
            ".jpg",
            ".jpeg"
        ]:
            return "image/jpeg"

        if extension == ".png":
            return "image/png"

        if extension == ".webp":
            return "image/webp"

        raise ValueError(
            f"Unsupported image format: {extension}"
        )

    # ---------------------------------------------------------
    # Parse Gemini JSON response
    # ---------------------------------------------------------

    def _parse_json(self, text):

        if not text:
            return {
                "error":
                    "Gemini returned an empty response"
            }

        text = text.strip()

        # Remove Markdown JSON code block
        if text.startswith("```json"):

            text = text[
                len("```json"):
            ]

        elif text.startswith("```"):

            text = text[
                len("```"):
            ]

        if text.endswith("```"):

            text = text[
                :-3
            ]

        text = text.strip()

        # Try direct JSON parsing
        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        # Try extracting JSON object
        start = text.find("{")
        end = text.rfind("}")

        if (
            start != -1
            and end != -1
        ):

            json_text = text[
                start:end + 1
            ]

            try:

                return json.loads(
                    json_text
                )

            except json.JSONDecodeError:
                pass

        # Return raw response for debugging
        return {
            "error":
                "Invalid JSON returned by Gemini",

            "raw_response":
                text
        }
