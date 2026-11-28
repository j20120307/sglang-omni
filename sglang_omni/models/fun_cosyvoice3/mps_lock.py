# SPDX-License-Identifier: Apache-2.0
"""The one lock every Fun-CosyVoice3 thread takes before it encodes Torch MPS work."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType

import torch

# note (j20120307): Torch MPS encodes kernels into the device's command buffer without a lock of its own, so two threads encoding at once trip the Metal assertion "A command encoder is already encoding to this command buffer" (pytorch/pytorch#197805). The AR scheduler, the vocoder scheduler and the preprocessing pool share one process and one device, so their MPS steps take turns here. Re-entrant: a vocoder step that finishes a stream runs a fallback decode inside its own turn.
MPS_STEP_LOCK = threading.RLock()


class MpsStepTurn(AbstractContextManager[None]):
    """One step's turn on MPS_STEP_LOCK; reusable, so a stage keeps one and re-enters it."""

    def __enter__(self) -> None:
        MPS_STEP_LOCK.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        MPS_STEP_LOCK.release()


def mps_step_context(device: torch.device) -> AbstractContextManager[None]:
    """A turn on the shared MPS lock; a no-op on CUDA and CPU, whose paths stay as they were."""
    if device.type == "mps":
        return MpsStepTurn()
    else:
        return nullcontext()
