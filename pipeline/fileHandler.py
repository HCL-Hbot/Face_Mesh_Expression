import datetime
import glob
import os
from typing import List, Optional

"""
Utility module for handling file operations in the FaceMesh pipeline.

This module provides functions for:
- Loading image files from the dataset directory
- Creating timestamped output directories
- Managing CSV files for facemesh landmarks

The module supports PNG, JPG and JPEG image formats and implements recursive
directory searching for datasets.

Example usage:
    # Load all images from default dataset directory
    image_files = load_files()
    
    # Create output directory with timestamp
    output_dir = create_output_dir()
    
    # Create CSV file path for landmarks
    csv_path = create_facemesh_csv(output_dir)
"""


def load_files(data_path: Optional[str] = None) -> List[str]:
    """
    Load all image files from the specified directory recursively.

    Args:
        data_path: Optional path to the data directory. If None, uses the default
                  'dataset' directory in the project root.

    Returns:
        List of full paths to all found image files.

    Example:
        >>> files = load_files()  # Use default path
        >>> files = load_files("/custom/data/path")  # Use custom path
    """
    # Get the data path - if none provided, use default dataset directory
    if data_path is None:
        __location__ = os.path.realpath(
            os.path.join(os.path.dirname(__file__), os.pardir)
        )
        data_path = os.path.join(__location__, "dataset")

    # Define patterns for supported image formats (png, jpg, jpeg)
    print("[INFO] Loading images...")
    patterns = [
        os.path.join(data_path, "**", f"*.{ext}") for ext in ["png", "jpg", "jpeg"]
    ]

    # First pass: Count total number of files for user feedback
    count = 0
    for pattern in patterns:
        for filename in glob.iglob(pattern, recursive=True):
            count += 1
    print("[INFO] Loaded", count, "images")

    # Second pass: Build list of valid image files
    file_list = []
    for pattern in patterns:
        # Extend list with files that match pattern and are actual files (not directories)
        file_list.extend(
            [f for f in glob.iglob(pattern, recursive=True) if (os.path.isfile(f))]
        )

    return file_list


def create_output_dir() -> str:
    """
    Create a new output directory with timestamp.

    Creates a directory in the project's output folder named with the current
    timestamp in format YYYY-MM-DD_HH-MM-SS.

    Returns:
        Full path to the created output directory.

    Example:
        >>> output_dir = create_output_dir()
        >>> print(output_dir)
        '.../output/2025-03-10_21-30-00'
    """
    # Generate timestamp for unique directory name
    now = datetime.datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Construct output path in project's output directory
    __location__ = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_path = os.path.join(__location__, "output", date_time)

    # Create the directory
    os.makedirs(output_path)

    return output_path


def create_facemesh_csv(output_path: str) -> str:
    """
    Generate path for the facemesh landmarks CSV file.

    Args:
        output_path: Directory path where the CSV file should be created.

    Returns:
        Full path to the facemesh landmarks CSV file.

    Example:
        >>> output_dir = create_output_dir()
        >>> csv_path = create_facemesh_csv(output_dir)
        >>> print(csv_path)
        '.../output/2025-03-10_21-30-00/facemesh_landmarks.csv'
    """
    # Construct and return the full CSV file path
    facemesh_csv = os.path.join(output_path, "facemesh_landmarks.csv")
    return facemesh_csv


# Example usage and testing
if __name__ == "__main__":
    # Test loading files from default location
    file_list = load_files()

    # Print each found file path
    for filename in file_list:
        print(filename)
