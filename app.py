"""Train and use a coffee-bean defect percentage model.

The model is binary at bean level: images from the defect archive are labeled
Defect, and images from the four color classes in the healthy archive are
labeled Good. A photo containing multiple beans is segmented before scoring.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_VERSION = 1


def image_from_bytes(data: bytes) -> np.ndarray | None:
	array = np.frombuffer(data, dtype=np.uint8)
	return cv2.imdecode(array, cv2.IMREAD_COLOR)


def resolve_path(path: Path) -> Path:
	if path.is_absolute():
		return path
	candidates = [Path.cwd() / path, Path(__file__).resolve().parent / path]
	for candidate in candidates:
		if candidate.exists():
			return candidate.resolve()
	return (Path(__file__).resolve().parent / path).resolve()


def feature_vector(image: np.ndarray) -> np.ndarray:
	image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	histograms = [
		cv2.calcHist([hsv], [0], None, [16], [0, 180]),
		cv2.calcHist([hsv], [1], None, [16], [0, 256]),
		cv2.calcHist([hsv], [2], None, [16], [0, 256]),
	]
	histograms = [cv2.normalize(hist, hist).flatten() for hist in histograms]
	small_gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
	color_stats = np.concatenate(
		[image.mean(axis=(0, 1)), image.std(axis=(0, 1)), hsv.mean(axis=(0, 1))]
	)
	return np.concatenate([*histograms, small_gray.flatten() / 255.0, color_stats])


def labeled_images(defect_zip: Path, healthy_zip: Path):
	with zipfile.ZipFile(defect_zip) as archive:
		for name in archive.namelist():
			path = Path(name)
			if path.suffix.lower() in IMAGE_EXTENSIONS and not name.endswith("/"):
				image = image_from_bytes(archive.read(name))
				if image is not None:
					yield image, 1
	with zipfile.ZipFile(healthy_zip) as archive:
		for name in archive.namelist():
			path = Path(name)
			if path.suffix.lower() in IMAGE_EXTENSIONS and not name.endswith("/"):
				image = image_from_bytes(archive.read(name))
				if image is not None:
					yield image, 0


def train(defect_zip: Path, healthy_zip: Path, model_path: Path) -> None:
	defect_zip = resolve_path(defect_zip)
	healthy_zip = resolve_path(healthy_zip)
	model_path = resolve_path(model_path)
	features, labels = [], []
	for image, label in labeled_images(defect_zip, healthy_zip):
		features.append(feature_vector(image))
		labels.append(label)
	if not features or len(set(labels)) != 2:
		raise ValueError("Both ZIP files must contain readable images.")

	x_train, x_test, y_train, y_test = train_test_split(
		features, labels, test_size=0.2, random_state=42, stratify=labels
	)
	classifier = RandomForestClassifier(
		n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
	)
	classifier.fit(x_train, y_train)
	predictions = classifier.predict(x_test)
	print(classification_report(y_test, predictions, target_names=["Good", "Defect"]))
	joblib.dump(
		{"version": MODEL_VERSION, "classifier": classifier}, model_path
	)
	print(f"Saved model to {model_path}")
	print(f"Training images: {len(features)} ({sum(labels)} defect, {len(labels) - sum(labels)} good)")


def bean_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	threshold, _ = cv2.threshold(
		blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
	)
	height, width = gray.shape
	corner_size = max(5, min(height, width) // 20)
	corners = np.concatenate([
		blurred[:corner_size, :corner_size].ravel(),
		blurred[:corner_size, -corner_size:].ravel(),
		blurred[-corner_size:, :corner_size].ravel(),
		blurred[-corner_size:, -corner_size:].ravel(),
	])
	mode = cv2.THRESH_BINARY_INV if corners.mean() > threshold else cv2.THRESH_BINARY
	_, mask = cv2.threshold(blurred, threshold, 255, mode)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	image_area = height * width
	boxes = []
	for contour in contours:
		x, y, box_width, box_height = cv2.boundingRect(contour)
		if x == 0 or y == 0 or x + box_width >= width or y + box_height >= height:
			continue
		area = cv2.contourArea(contour)
		if image_area * 0.0003 <= area <= image_area * 0.20:
			boxes.append((x, y, box_width, box_height))
	return boxes


def crops_for_photo(image: np.ndarray) -> list[np.ndarray]:
	boxes = bean_boxes(image)
	if not boxes:
		return [image]
	height, width = image.shape[:2]
	crops = []
	for x, y, box_width, box_height in boxes:
		pad = max(4, int(min(box_width, box_height) * 0.08))
		crops.append(image[max(0, y - pad):min(height, y + box_height + pad),
						   max(0, x - pad):min(width, x + box_width + pad)])
	return crops


def predict(model_path: Path, photo_path: Path) -> None:
	model_path = resolve_path(model_path)
	photo_path = resolve_path(photo_path)
	if not photo_path.exists():
		raise FileNotFoundError(f"Could not read image: {photo_path}")
	image = cv2.imread(str(photo_path))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {photo_path}")
	saved = joblib.load(model_path)
	classifier = saved["classifier"]
	crops = crops_for_photo(image)
	probabilities = classifier.predict_proba([feature_vector(crop) for crop in crops])[:, 1]
	defect_count = int((probabilities >= 0.5).sum())
	percentage = 100.0 * defect_count / len(crops)
	print(json.dumps({
		"photo": str(photo_path),
		"beans_detected": len(crops),
		"defect_beans": defect_count,
		"good_beans": len(crops) - defect_count,
		"defect_percentage": round(percentage, 2),
		"average_defect_probability": round(float(probabilities.mean()), 4),
	}, indent=2))


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	subparsers = parser.add_subparsers(dest="command", required=True)
	train_parser = subparsers.add_parser("train", help="Train from two ZIP datasets")
	train_parser.add_argument("defect_zip", type=Path)
	train_parser.add_argument("healthy_zip", type=Path)
	train_parser.add_argument("--model", type=Path, default=Path("coffee_defect_model.joblib"))
	predict_parser = subparsers.add_parser("predict", help="Estimate defects in a photo")
	predict_parser.add_argument("photo", type=Path)
	predict_parser.add_argument("--model", type=Path, default=Path("coffee_defect_model.joblib"))
	args = parser.parse_args()
	try:
		if args.command == "train":
			train(args.defect_zip, args.healthy_zip, args.model)
		else:
			predict(args.model, args.photo)
	except (FileNotFoundError, ValueError) as exc:
		parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
	main()








