import os
import tempfile
import unittest
from pathlib import Path

import app
import cv2


class ResolvePathTests(unittest.TestCase):
    def test_relative_photo_path_resolves_from_app_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp_dir:
            os.chdir(tmp_dir)
            try:
                resolved = app.resolve_path(Path("my_beans.jpg"))
                self.assertTrue(resolved.is_file())
                self.assertEqual(resolved.name, "my_beans.jpg")
                self.assertEqual(resolved.parent, Path(app.__file__).resolve().parent)
            finally:
                os.chdir(original)


class BeanDetectionTests(unittest.TestCase):
    def test_ignores_contour_touching_image_border(self):
        image = cv2.imread(str(Path(app.__file__).resolve().parent / "my_beans.jpg"))

        boxes = app.bean_boxes(image)

        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(x > 0 and y > 0 for x, y, _, _ in boxes))
        self.assertTrue(all(x + width < image.shape[1] and y + height < image.shape[0]
                            for x, y, width, height in boxes))


if __name__ == "__main__":
    unittest.main()
