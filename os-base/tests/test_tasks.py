# SPDX-FileCopyrightText: 2026 Conecsa
#
# SPDX-License-Identifier: Apache-2.0

"""The shared task ids and the reserved face name."""
import pytest
from conecsa_common.tasks import (
    UNKNOWN_FACE,
    is_reserved_face_name,
    is_safe_class_name,
    is_task,
    person_key,
    task_or_default,
)


def test_task_ids():
    assert is_task("face") and not is_task("pose") and not is_task(None)
    assert task_or_default("") == "detect" and task_or_default(" face ") == "face"


@pytest.mark.parametrize("name", ["unknown", "Unknown", "UNKNOWN", "  unknown ",
                                  "unknown #ff0000", "Unknown#ABCDEF"])
def test_the_unknown_face_name_is_reserved_in_any_spelling(name):
    assert is_reserved_face_name(name)


@pytest.mark.parametrize("name", ["unknown person", "known", "unknown #ff00", "", None, 3])
def test_other_names_are_not(name):
    assert not is_reserved_face_name(name)


def test_the_sentinel_is_the_lowercase_word():
    assert UNKNOWN_FACE == "unknown"


def test_people_compare_without_case_or_colour():
    assert person_key("Alice") == person_key(" ALICE #ff0000 ") == "alice"
    assert person_key("Alice #ff00") == "alice #ff00"  # not a colour suffix
    assert person_key(None) == ""


@pytest.mark.parametrize("name,ok", [
    ("Ana Souza", True), ("cap #ff0000", True), ("a.b_c-d", True), ("x" * 64, True),
    ("", False), ("  ", False), ("x" * 65, False), ("Ana\nBruno", False), ("bad/name", False),
    ("semi;colon", False), (None, False),
])
def test_safe_class_names(name, ok):
    assert is_safe_class_name(name) is ok
