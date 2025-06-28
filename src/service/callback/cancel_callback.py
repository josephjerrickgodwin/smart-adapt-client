from threading import Event
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

class CancelCallback(TrainerCallback):
    """Internal callback class to cooperatively stop training when the user
    requests cancellation via ``stop_fine_tuning``."""

    def __init__(self, _event: Event):
        self._event = _event

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if self._event.is_set():
            control.should_training_stop = True
            control.should_epoch_stop = True
            control.should_save = True
        return control