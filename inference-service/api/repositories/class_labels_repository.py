"""
Class labels repository - Handles class labels file persistence.
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class ClassLabelsRepository:
    """Repository for class labels file operations."""

    def __init__(self, classes_file_path: str):
        """
        Initialize the class labels' repository.

        Args:
            classes_file_path: Path to classes.txt file
        """
        self.classes_file_path = classes_file_path

    def load_labels(self) -> List[str]:
        """
        Load class labels from file.

        Returns:
            List of class label strings
        """
        try:
            if not os.path.exists(self.classes_file_path):
                logger.warning(f"Classes file not found: {self.classes_file_path}")
                return []

            with open(self.classes_file_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]

            logger.info(f"Loaded {len(labels)} class labels")
            return labels

        except Exception as ex:
            logger.error(f"Error loading class labels: {ex}")
            return []

    def save_labels(self, labels: List[str]) -> bool:
        """
        Save class labels to file.

        Args:
            labels: List of class label strings

        Returns:
            True if saved successfully
        """
        try:
            from conecsa_common import atomic_write_bytes

            # Atomic + fsync: the labels file maps class indices to names for
            # the live detector; a power cut must never truncate it.
            atomic_write_bytes(self.classes_file_path,
                               '\n'.join(labels).encode("utf-8"), mode=0o644)

            logger.info(f"Saved {len(labels)} class labels to {self.classes_file_path}")
            return True

        except Exception as ex:
            logger.error(f"Error saving class labels: {ex}")
            return False
