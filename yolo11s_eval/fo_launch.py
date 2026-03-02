"""Quick-launch the FiftyOne interactive app.

Loads the already-evaluated dataset and starts the web server.
Run ``voxel51_eval.py`` first to populate the dataset.

Usage:
    python -m yolo11s_eval.fo_launch
    # Then open http://localhost:5151 in your browser
"""

import os
import sys
import time

# Fix builtin-plugin discovery on external drives (FiftyOne #5484)
_site_pkgs = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "venv", "lib", "python3.14", "site-packages",
)
_plugins = os.path.join(_site_pkgs, "plugins")
if os.path.isdir(_plugins):
    os.environ["FIFTYONE_PLUGINS_DIR"] = _plugins

import fiftyone as fo  # noqa: E402

from .config import FIFTYONE_DATASET_NAME, FIFTYONE_PORT  # noqa: E402


def main() -> None:
    """Load the persistent dataset and launch the FiftyOne app."""
    if not fo.dataset_exists(FIFTYONE_DATASET_NAME):
        print(f"Dataset '{FIFTYONE_DATASET_NAME}' not found.")
        print("Run  python -m yolo11s_eval.voxel51_eval  first.")
        sys.exit(1)

    dataset = fo.load_dataset(FIFTYONE_DATASET_NAME)
    dataset.persistent = True
    print(f"Loaded: {dataset.name}  ({len(dataset)} samples)")

    print(f"\nStarting FiftyOne at http://localhost:{FIFTYONE_PORT} ...")
    session = fo.launch_app(dataset, port=FIFTYONE_PORT, auto=False)
    print(f"FiftyOne is running at http://localhost:{FIFTYONE_PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down FiftyOne ...")
        session.close()


if __name__ == "__main__":
    main()
