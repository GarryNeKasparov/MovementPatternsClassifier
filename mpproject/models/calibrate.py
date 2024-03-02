import sys

from data_utils import (
    build_loader,
    get_model,
)
from fortuna.output_calib_model import OutputCalibClassifier
from fortuna.output_calib_model.config.base import Config
from fortuna.output_calib_model.config.optimizer import Optimizer
from train import get_outputs_and_targets

if __name__ == "__main__":
    name = sys.argv[1]
    model = get_model(name)
    calib_loader = build_loader("calib")
    calib_outputs, calib_targets = get_outputs_and_targets(model, calib_loader)
    calib_model = OutputCalibClassifier()
    status = calib_model.calibrate(
        calib_outputs=calib_outputs,
        calib_targets=calib_targets,
        config=Config(optimizer=Optimizer(n_epochs=500)),
    )
    calib_model.save_state(f"mpproject/models/files/weights/{name}_calibrated")
