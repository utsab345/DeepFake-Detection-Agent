# deepfake_detector.py
import argparse
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np

# optional face cropping (fast/simple) using face_recognition if available
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except Exception:
    FACE_REC_AVAILABLE = False

class DeepfakeDetector:
    def __init__(self, model_name="dima806/deepfake_vs_real_image_detection"):
        """
        Initialize the DeepfakeDetector with a pretrained Hugging Face model.
        Default model: dima806/deepfake_vs_real_image_detection (recommended).
        """
        print(f"[info] Loading model: {model_name} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[info] Using device: {self.device}")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("[info] Model loaded successfully.")
        except Exception as e:
            print(f"[error] Error loading model '{model_name}': {e}")
            raise

    def _crop_face_if_possible(self, pil_image):
        """
        If face_recognition is installed, detect faces and return the largest face crop.
        Otherwise, return the center crop (square) as a fallback.
        """
        if FACE_REC_AVAILABLE:
            # convert to numpy RGB
            img_np = np.array(pil_image)
            boxes = face_recognition.face_locations(img_np, model="hog")  # or 'cnn' if installed & desired
            if boxes:
                # boxes are (top, right, bottom, left)
                # choose largest box
                areas = [(b[2]-b[0])*(b[1]-b[3]) for b in boxes]
                idx = int(np.argmax(areas))
                top, right, bottom, left = boxes[idx]
                # pad a little
                hpad = int(0.15 * (bottom - top))
                wpad = int(0.15 * (right - left))
                top = max(0, top - hpad)
                left = max(0, left - wpad)
                bottom = min(pil_image.height, bottom + hpad)
                right = min(pil_image.width, right + wpad)
                return pil_image.crop((left, top, right, bottom))
            # if no faces found, fall through to center crop
        # center square crop fallback
        w, h = pil_image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        return pil_image.crop((left, top, left + side, top + side))

    def predict(self, image_path, use_face_crop=True):
        """
        Predict whether an image is Real or Fake.
        Returns dict: {label, confidence, all_scores}
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found"}

        try:
            image = Image.open(image_path).convert("RGB")
            if use_face_crop:
                cropped = self._crop_face_if_possible(image)
            else:
                cropped = image

            inputs = self.processor(images=cropped, return_tensors="pt")
            # move tensors to device properly
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_idx = int(logits.argmax(-1).item())
                label = self.model.config.id2label[pred_idx]
                score = float(probs[0, pred_idx].item())

                all_scores = {
                    self.model.config.id2label[i]: float(probs[0, i].item())
                    for i in range(len(self.model.config.id2label))
                }

            return {"label": label, "confidence": score, "all_scores": all_scores}
        except Exception as e:
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection System")
    parser.add_argument("image_path", type=str, help="Path to the image to analyze")
    parser.add_argument("--model", type=str, default="dima806/deepfake_vs_real_image_detection",
                        help="Hugging Face model name (default: dima806/deepfake_vs_real_image_detection)")
    parser.add_argument("--no-face-crop", dest="use_face_crop", action="store_false",
                        help="Disable face cropping (use full image)")
    args = parser.parse_args()

    print("[info] face_recognition installed:", FACE_REC_AVAILABLE)
    detector = DeepfakeDetector(model_name=args.model)
    result = detector.predict(args.image_path, use_face_crop=args.use_face_crop)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\n--- Detection Result ---")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nDetailed Scores:")
        for label, score in result['all_scores'].items():
            print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()