import os
import sys

import numpy as np
from data_utils import (
    build_loader,
    get_model,
)
from fortuna.conformal import AdaptivePredictionConformalClassifier
from fortuna.output_calib_model import OutputCalibClassifier
from train import get_outputs_and_targets

if __name__ == "__main__":
    name = sys.argv[1]
    assert os.path.exists(
        f"mpproject/models/files/weights/{name}_calibrated"
    ), f"{name} is not calibrated yet."
    model = get_model(name)
    calib_model = OutputCalibClassifier()
    calib_model.load_state(
        f"mpproject/models/files/weights/{name}_calibrated/checkpoint_500"
    )
    val_loader = build_loader("val")
    test_loader = build_loader("test")
    val_outputs, val_targets = get_outputs_and_targets(model, val_loader)
    test_outputs, test_targets = get_outputs_and_targets(model, test_loader)
    test_modes_calibrated = calib_model.predictive.mode(outputs=test_outputs)
    test_means_calibrated = calib_model.predictive.mean(outputs=test_outputs)
    val_means_calibrated = calib_model.predictive.mean(outputs=val_outputs)

    conformal_sets = AdaptivePredictionConformalClassifier().conformal_set(
        val_probs=val_means_calibrated,
        test_probs=test_means_calibrated,
        val_targets=val_targets,
        error=0.1,
    )
    sizes = [len(s) for s in np.array(conformal_sets, dtype="object")]
    set_sizes, sizes_counts = np.unique(sizes, return_counts=True)
    avg_size = np.mean(sizes)
    max_size = np.max(sizes)
    min_size = np.min(sizes)
    mode_size = set_sizes[np.argmax(sizes_counts)]

    avg_size_wellclassified = np.mean(
        [
            len(s)
            for s in np.array(conformal_sets, dtype="object")[
                test_modes_calibrated == test_targets
            ]
        ]
    )
    avg_size_misclassified = np.mean(
        [
            len(s)
            for s in np.array(conformal_sets, dtype="object")[
                test_modes_calibrated != test_targets
            ]
        ]
    )

    print(
        f"Average conformal set size: {avg_size}\nMax conformal set size: {max_size}\n"
        f"Min conformal set size: {min_size}\nMode conformal set size: {mode_size}"
    )

    print(
        f"Average conformal set size over \
        well classified input: {avg_size_wellclassified}"
    )
    print(
        f"Average conformal set size over misclassified input: {avg_size_misclassified}"
    )
