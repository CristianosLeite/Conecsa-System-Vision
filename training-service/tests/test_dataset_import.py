# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Unit tests for YOLO dataset ZIP import validation and normalization."""
import json
import os
import zipfile

import cv2
import numpy as np
import pytest
from service.dataset_import import (
    DatasetImportError,
    _classes_from_yaml,
    _label_for,
    _parse_label_file,
    _validate_classes,
    import_dataset_zip,
)


def _jpeg():
    frame = np.full((80, 120, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class TestValidateClasses:
    def test_valid(self):
        assert _validate_classes(["cap", "bottle"], "data.yaml") == ["cap", "bottle"]

    def test_empty_rejected(self):
        with pytest.raises(DatasetImportError):
            _validate_classes([], "data.yaml")

    def test_invalid_char_rejected(self):
        with pytest.raises(DatasetImportError):
            _validate_classes(["good", "bad/name"], "data.yaml")

    def test_too_long_rejected(self):
        with pytest.raises(DatasetImportError):
            _validate_classes(["x" * 65], "data.yaml")

    def test_duplicates_rejected(self):
        with pytest.raises(DatasetImportError):
            _validate_classes(["cap", "cap"], "data.yaml")

    def test_unknown_is_reserved_for_face_only(self):
        assert _validate_classes(["unknown", "cap"], "data.yaml") == ["unknown", "cap"]
        assert _validate_classes(["unknown"], "classes.txt", task="classify") == ["unknown"]
        with pytest.raises(DatasetImportError, match="reserved"):
            _validate_classes(["Alice", "Unknown #ff0000"], "the class folders", task="face")


class TestClassesFromYaml:
    def test_list_form(self, tmp_path):
        p = tmp_path / "data.yaml"
        p.write_text("names: [cap, bottle]\n")
        assert _classes_from_yaml(str(p)) == ["cap", "bottle"]

    def test_dict_form_ordered_by_int_key(self, tmp_path):
        p = tmp_path / "data.yaml"
        p.write_text("names:\n  1: bottle\n  0: cap\n")
        assert _classes_from_yaml(str(p)) == ["cap", "bottle"]

    def test_missing_names_rejected(self, tmp_path):
        p = tmp_path / "data.yaml"
        p.write_text("train: images\n")
        with pytest.raises(DatasetImportError):
            _classes_from_yaml(str(p))


class TestLabelFor:
    def test_swaps_images_for_labels(self):
        img = os.path.join("root", "train", "images", "a.jpg")
        expected = os.path.join("root", "train", "labels", "a.txt")
        assert _label_for(img) == expected

    def test_sibling_txt_when_no_images_dir(self):
        img = os.path.join("root", "a.jpg")
        assert _label_for(img) == os.path.join("root", "a.txt")


class TestParseLabelFile:
    def test_detection_rows(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n")
        boxes = _parse_label_file(str(p), n_classes=2)
        assert boxes == [
            (0, 0.5, 0.5, 0.2, 0.2),
            (1, 0.1, 0.1, 0.05, 0.05),
        ]

    def test_polygon_rows_are_refused(self, tmp_path):
        # A segmentation export is not turned into boxes behind the
        # operator's back: it imports only into a segmentation dataset.
        p = tmp_path / "a.txt"
        p.write_text("0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n")
        with pytest.raises(DatasetImportError, match="segmentation polygon"):
            _parse_label_file(str(p), n_classes=1)

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("\n0 0.5 0.5 0.2 0.2\n\n")
        assert len(_parse_label_file(str(p), n_classes=1)) == 1

    def test_non_numeric_rejected(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 a b c d\n")
        with pytest.raises(DatasetImportError):
            _parse_label_file(str(p), n_classes=1)

    def test_class_out_of_range_rejected(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("5 0.5 0.5 0.2 0.2\n")
        with pytest.raises(DatasetImportError):
            _parse_label_file(str(p), n_classes=2)

    def test_coords_out_of_range_rejected(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 1.5 0.5 0.2 0.2\n")
        with pytest.raises(DatasetImportError):
            _parse_label_file(str(p), n_classes=1)

    def test_wrong_value_count_rejected(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("0 0.5 0.5\n")
        with pytest.raises(DatasetImportError):
            _parse_label_file(str(p), n_classes=1)


class TestImportDatasetZip:
    def _make_zip(self, tmp_path, with_label=True):
        zip_path = tmp_path / "ds.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("data.yaml", "names: [cap, bottle]\n")
            z.writestr("images/img1.jpg", _jpeg())
            if with_label:
                z.writestr("labels/img1.txt", "0 0.5 0.5 0.2 0.2\n")
        return str(zip_path)

    def test_valid_import(self, tmp_path):
        zip_path = self._make_zip(tmp_path)
        dest = tmp_path / "out"
        classes, count = import_dataset_zip(zip_path, str(dest))
        assert classes == ["cap", "bottle"]
        assert count == 1
        assert (dest / "classes.json").exists()
        assert len(list((dest / "images").glob("*.jpg"))) == 1
        assert len(list((dest / "labels").glob("*.txt"))) == 1

    def test_unlabeled_image_still_imported(self, tmp_path):
        zip_path = self._make_zip(tmp_path, with_label=False)
        dest = tmp_path / "out"
        classes, count = import_dataset_zip(zip_path, str(dest))
        assert count == 1
        assert len(list((dest / "labels").glob("*.txt"))) == 0

    def test_bad_zip_rejected(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a zip")
        with pytest.raises(DatasetImportError):
            import_dataset_zip(str(bad), str(tmp_path / "out"))

    def test_missing_classes_rejected(self, tmp_path):
        zip_path = tmp_path / "noclasses.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("images/img1.jpg", _jpeg())
        with pytest.raises(DatasetImportError):
            import_dataset_zip(str(zip_path), str(tmp_path / "out"))

    def test_a_colour_suffixed_class_name_reimports(self, tmp_path):
        # The dataset export writes class names with their "#rrggbb" suffix.
        zip_path = tmp_path / "ds.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("data.yaml", "names: ['cap #ff0000']\n")
            z.writestr("images/img1.jpg", _jpeg())
        classes, _ = import_dataset_zip(str(zip_path), str(tmp_path / "out"))
        assert classes == ["cap #ff0000"]


def _zip(tmp_path, entries, name="cls.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for arcname, data in entries.items():
            z.writestr(arcname, data)
    return str(path)


def _labels(dest):
    return sorted(p.read_text() for p in (dest / "labels").glob("*.txt"))


class TestClassificationImport:
    """A classify dataset imports one folder of images per class."""

    def _import(self, tmp_path, entries, img_size=0):
        dest = tmp_path / "out"
        classes, count = import_dataset_zip(_zip(tmp_path, entries), str(dest),
                                            img_size=img_size, task="classify")
        return classes, count, dest

    def test_split_folders_are_merged_and_classes_sorted(self, tmp_path):
        classes, count, dest = self._import(tmp_path, {
            n: _jpeg() for n in ("train/dog/a.jpg", "train/cat/b.jpg", "val/cat/c.jpg",
                                 "test/dog/d.jpg")})
        assert (classes, count) == (["cat", "dog"], 4)
        # One label line per image: its class id.
        assert _labels(dest) == ["0\n", "0\n", "1\n", "1\n"]
        assert json.loads((dest / "classes.json").read_text()) == ["cat", "dog"]

    def test_top_level_class_folders_inside_a_wrapper(self, tmp_path):
        classes, count, _ = self._import(tmp_path, {"pets/cat/a.jpg": _jpeg(),
                                                    "pets/dog/b.jpg": _jpeg()})
        assert (classes, count) == (["cat", "dog"], 2)

    def test_a_single_class_archive(self, tmp_path):
        classes, count, _ = self._import(tmp_path, {"cat/a.jpg": _jpeg(), "cat/b.jpg": _jpeg()})
        assert (classes, count) == (["cat"], 2)

    def test_classes_txt_fixes_the_order_and_keeps_empty_classes(self, tmp_path):
        classes, _, dest = self._import(tmp_path, {
            "classes.txt": "dog\ncat\nbird\n", "train/cat/a.jpg": _jpeg(),
            "train/dog/b.jpg": _jpeg()})
        assert classes == ["dog", "cat", "bird"]
        assert _labels(dest) == ["0\n", "1\n"]

    def test_a_folder_missing_from_classes_txt_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="'dog' is not listed"):
            self._import(tmp_path, {"classes.txt": "cat\n", "train/cat/a.jpg": _jpeg(),
                                    "train/dog/b.jpg": _jpeg()})

    def test_a_colour_suffixed_class_folder_is_kept(self, tmp_path):
        classes, _, _ = self._import(tmp_path, {"train/cat #ff0000/a.jpg": _jpeg(),
                                                "train/dog/b.jpg": _jpeg()})
        assert classes == ["cat #ff0000", "dog"]

    def test_a_dot_class_folder_is_kept_and_a_hidden_tool_folder_ignored(self, tmp_path):
        classes, count, dest = self._import(tmp_path, {
            "train/.defective/a.jpg": _jpeg(), "train/good/b.jpg": _jpeg(),
            "train/.git/config": "[core]\n"})
        assert (classes, count) == ([".defective", "good"], 2)
        assert _labels(dest) == ["0\n", "1\n"]

    def test_an_invalid_class_folder_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="Invalid class name"):
            self._import(tmp_path, {"train/c*t/a.jpg": _jpeg(), "train/dog/b.jpg": _jpeg()})

    def test_a_detection_archive_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="object-detection dataset"):
            self._import(tmp_path, {"data.yaml": "names: [cap]\n", "images/a.jpg": _jpeg(),
                                    "labels/a.txt": "0 0.5 0.5 0.2 0.2\n"})

    def test_an_archive_without_class_folders_is_refused(self, tmp_path):
        with pytest.raises(DatasetImportError, match="class folders"):
            self._import(tmp_path, {"a.jpg": _jpeg()})

    def test_images_are_stored_in_the_dataset_geometry(self, tmp_path):
        _, _, dest = self._import(tmp_path, {"cat/a.jpg": _jpeg(), "dog/b.jpg": _jpeg()},
                                  img_size=640)
        shapes = set()
        for path in (dest / "images").glob("*.jpg"):
            image = cv2.imread(str(path))
            assert image is not None
            shapes.add(image.shape)
        assert shapes == {(640, 640, 3)}

    def test_a_classification_layout_is_refused_for_segmentation(self, tmp_path):
        archive = _zip(tmp_path, {"train/cap/a.jpg": _jpeg(), "train/bottle/b.jpg": _jpeg()})
        with pytest.raises(DatasetImportError, match="runs segmentation"):
            import_dataset_zip(archive, str(tmp_path / "out"), task="segment")


class TestNativeGeometryImport:
    """``img_size=0`` keeps the source resolution and the labels verbatim."""

    def _make_zip(self, tmp_path, label="0 0.500000 0.250000 0.200000 0.100000\n"):
        zip_path = tmp_path / "ds.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("data.yaml", "names: [cap]\n")
            z.writestr("images/img1.jpg", _jpeg())  # 120×80 source
            z.writestr("labels/img1.txt", label)
        return str(zip_path)

    def test_letterbox_import_transforms_labels(self, tmp_path):
        dest = tmp_path / "out"
        import_dataset_zip(self._make_zip(tmp_path), str(dest), img_size=640)
        (image,) = (dest / "images").glob("*.jpg")
        stored = cv2.imread(str(image))
        assert stored is not None and stored.shape == (640, 640, 3)
        (label,) = (dest / "labels").glob("*.txt")
        # 120×80 content is centered vertically in the square, so cy moves.
        parts = label.read_text().split()
        assert float(parts[2]) != pytest.approx(0.25)

    def test_native_import_keeps_dims_and_labels(self, tmp_path):
        dest = tmp_path / "out"
        import_dataset_zip(self._make_zip(tmp_path), str(dest), img_size=0)
        (image,) = (dest / "images").glob("*.jpg")
        stored = cv2.imread(str(image))
        assert stored is not None and stored.shape == (80, 120, 3)
        (label,) = (dest / "labels").glob("*.txt")
        assert label.read_text() == "0 0.500000 0.250000 0.200000 0.100000\n"

    def test_native_import_refuses_polygons_too(self, tmp_path):
        dest = tmp_path / "out"
        poly = "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n"
        with pytest.raises(DatasetImportError, match="segmentation"):
            import_dataset_zip(self._make_zip(tmp_path, label=poly), str(dest),
                               img_size=0)

    def test_a_classification_layout_is_refused(self, tmp_path):
        # ultralytics classification export: <split>/<class>/*.jpg, no labels.
        path = tmp_path / "cls.zip"
        with zipfile.ZipFile(path, "w") as z:
            for name in ("train/cap/a.jpg", "train/bottle/b.jpg", "val/cap/c.jpg"):
                z.writestr(name, _jpeg())
        with pytest.raises(DatasetImportError, match="classification dataset"):
            import_dataset_zip(str(path), str(tmp_path / "out"))

    def test_a_detection_archive_is_refused_on_a_classification_device(self, tmp_path):
        with pytest.raises(DatasetImportError, match="object-detection dataset"):
            import_dataset_zip(self._make_zip(tmp_path), str(tmp_path / "out"),
                               task="classify")

    def test_native_import_still_rejects_oversized_images(self, tmp_path,
                                                          monkeypatch):
        from service import dataset_import
        monkeypatch.setattr(dataset_import, "_MAX_IMAGE_PIXELS", 1000)

        def never(path):
            raise AssertionError("cv2.imread must not run on a rejected image")

        monkeypatch.setattr(dataset_import.cv2, "imread", never)
        with pytest.raises(DatasetImportError, match="too large"):
            import_dataset_zip(self._make_zip(tmp_path), str(tmp_path / "out"),
                               img_size=0)

    def test_native_import_still_rejects_undecodable_images(self, tmp_path):
        zip_path = tmp_path / "ds.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("data.yaml", "names: [cap]\n")
            z.writestr("images/img1.jpg", b"\xff\xd8not really a jpeg")
        with pytest.raises(DatasetImportError, match="decode"):
            import_dataset_zip(str(zip_path), str(tmp_path / "out"), img_size=0)


class TestExtractionLimits:
    """Real (written-byte) limits, not ZIP-header claims."""

    def _zip_with(self, tmp_path, entries):
        zip_path = tmp_path / "ds.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("data.yaml", "names: [cap]\n")
            for name, data in entries:
                z.writestr(name, data)
        return str(zip_path)

    def test_the_budget_counts_written_bytes_not_header_claims(self, tmp_path):
        # The old cap summed the ZIP's declared file_size — attacker-
        # controlled, so a header claiming 0 bytes bypassed it. The budget is
        # now a counter on the bytes actually written, which no header field
        # can influence.
        zip_path = self._zip_with(tmp_path,
                                  [("images/big.bin", b"x" * (4 << 20))])
        with pytest.raises(DatasetImportError, match="exceeds the 1 MB limit"):
            import_dataset_zip(zip_path, str(tmp_path / "out"), max_total_mb=1)

    def test_entry_count_is_capped(self, tmp_path, monkeypatch):
        from service import dataset_import
        monkeypatch.setattr(dataset_import, "_MAX_ZIP_ENTRIES", 3)
        zip_path = self._zip_with(tmp_path, [
            (f"images/i{i}.jpg", _jpeg()) for i in range(4)
        ])
        with pytest.raises(DatasetImportError, match="entries"):
            import_dataset_zip(zip_path, str(tmp_path / "out"))

    def test_per_entry_size_is_capped(self, tmp_path, monkeypatch):
        from service import dataset_import
        monkeypatch.setattr(dataset_import, "_MAX_ZIP_ENTRY_BYTES", 1 << 20)
        zip_path = self._zip_with(tmp_path,
                                  [("images/big.bin", b"x" * (2 << 20))])
        with pytest.raises(DatasetImportError, match="exceeds"):
            import_dataset_zip(zip_path, str(tmp_path / "out"),
                               max_total_mb=512)

    def test_a_decompression_bomb_ratio_is_rejected(self, tmp_path,
                                                    monkeypatch):
        from service import dataset_import
        monkeypatch.setattr(dataset_import, "_MAX_ZIP_RATIO", 50)
        # Highly repetitive content compresses ~1000:1.
        zip_path = self._zip_with(tmp_path,
                                  [("images/bomb.bin", b"\0" * (8 << 20))])
        with pytest.raises(DatasetImportError, match="compression ratio"):
            import_dataset_zip(zip_path, str(tmp_path / "out"),
                               max_total_mb=512)

    def test_symlink_entries_are_rejected(self, tmp_path):
        zip_path = tmp_path / "link.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("data.yaml", "names: [cap]\n")
            info = zipfile.ZipInfo("images/evil")
            info.external_attr = 0o120777 << 16  # symlink mode
            z.writestr(info, "/etc/passwd")
        with pytest.raises(DatasetImportError, match="symlink"):
            import_dataset_zip(str(zip_path), str(tmp_path / "out"))

    def test_an_oversized_image_is_rejected_before_decode(self, tmp_path,
                                                          monkeypatch):
        from service import dataset_import
        monkeypatch.setattr(dataset_import, "_MAX_IMAGE_PIXELS", 1000)

        def never(path):
            raise AssertionError("cv2.imread must not run on a rejected image")

        # A real (small) PNG whose header says 120x80 = 9600 px > 1000.
        frame = np.full((80, 120, 3), 128, dtype=np.uint8)
        ok, png = cv2.imencode(".png", frame)
        assert ok
        zip_path = self._zip_with(tmp_path, [("images/huge.png", png.tobytes())])
        monkeypatch.setattr(dataset_import.cv2, "imread", never)
        with pytest.raises(DatasetImportError, match="too large"):
            import_dataset_zip(zip_path, str(tmp_path / "out"))

    def test_a_normal_dataset_still_imports(self, tmp_path):
        zip_path = self._zip_with(tmp_path, [
            ("images/img1.jpg", _jpeg()),
            ("labels/img1.txt", "0 0.5 0.5 0.2 0.2\n"),
        ])
        classes, count = import_dataset_zip(zip_path, str(tmp_path / "out"))
        assert classes == ["cap"]
        assert count == 1


class TestImageDimensions:
    def test_png_jpeg_bmp_headers(self, tmp_path):
        from service.dataset_import import _image_dimensions
        frame = np.full((80, 120, 3), 128, dtype=np.uint8)
        for ext in (".png", ".jpg", ".bmp"):
            ok, buf = cv2.imencode(ext, frame)
            assert ok
            p = tmp_path / f"img{ext}"
            p.write_bytes(buf.tobytes())
            assert _image_dimensions(str(p)) == (120, 80), ext

    def test_unknown_format_returns_none(self, tmp_path):
        from service.dataset_import import _image_dimensions
        p = tmp_path / "img.webp"
        p.write_bytes(b"RIFF....WEBP")
        assert _image_dimensions(str(p)) is None

    def test_from_bytes_matches_the_path_variant(self):
        from service.dataset_import import image_dimensions_from_bytes
        frame = np.full((80, 120, 3), 128, dtype=np.uint8)
        for ext in (".png", ".jpg", ".bmp"):
            ok, buf = cv2.imencode(ext, frame)
            assert ok
            assert image_dimensions_from_bytes(buf.tobytes()) == (120, 80), ext
        assert image_dimensions_from_bytes(b"RIFF....WEBP") is None
        assert image_dimensions_from_bytes(b"") is None
