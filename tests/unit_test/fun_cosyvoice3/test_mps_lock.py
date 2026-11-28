# SPDX-License-Identifier: Apache-2.0
"""The Fun-CosyVoice3 threads that encode Torch MPS work take one shared lock."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Literal, Protocol

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fun_cosyvoice3 import request_builders, stages
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.mps_lock import (
    MPS_STEP_LOCK,
    MpsStepTurn,
    mps_step_context,
)
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    clear_cosyvoice3_preprocessing_context,
    preprocess_cosyvoice3_payload,
    set_cosyvoice3_preprocessing_context,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from tests.unit_test.fun_cosyvoice3.test_request_builders import (
    FakeModel,
    FakeSpeakerEncoder,
    FakeSpeechTokenizer,
    FakeTokenizer,
    make_payload,
)
from tests.unit_test.fun_cosyvoice3.test_vocoder import FakeFlow, FakeHiFT

EntryPoint = Literal["ar_execute", "vocoder_step", "preprocessing"]


class DeviceStep(Protocol):
    def __call__(self) -> None: ...


@dataclass
class HeldStep:
    """A fake device step that stays inside until released, so the test can probe the lock while it runs."""

    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)

    def run(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=5.0)


class PlacedFakeFlow(FakeFlow):
    """A Flow whose parameters report the device under test without a tensor there."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.placed_device = device

    def parameters(self, recurse: bool = True) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(device=self.placed_device, dtype=torch.float32)


@pytest.fixture
def preprocessing_context() -> Iterator[None]:
    clear_cosyvoice3_preprocessing_context()
    yield
    clear_cosyvoice3_preprocessing_context()


def test_mps_step_context_locks_only_torch_mps() -> None:
    assert isinstance(mps_step_context(torch.device("mps")), MpsStepTurn)
    assert isinstance(mps_step_context(torch.device("cuda")), nullcontext)
    assert isinstance(mps_step_context(torch.device("cpu")), nullcontext)


@pytest.mark.usefixtures("preprocessing_context")
@pytest.mark.parametrize("device_type", ["mps", "cpu"])
@pytest.mark.parametrize("entry_point", ["ar_execute", "vocoder_step", "preprocessing"])
def test_device_steps_hold_the_shared_lock_only_on_mps(
    entry_point: EntryPoint,
    device_type: Literal["mps", "cpu"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming on Torch MPS aborted the process with a Metal command-encoder assertion once the AR step, a vocoder step and request preprocessing ran at the same time; each of them now holds the one shared lock while it runs its device work, so a step on another thread waits. On CPU no lock is taken."""
    device = torch.device(device_type)
    held = HeldStep()
    step: DeviceStep
    if entry_point == "ar_execute":
        runner = object.__new__(FunCosyVoice3ModelRunner)
        runner.device = device
        monkeypatch.setattr(
            ModelRunner, "execute", lambda self, scheduler_output: held.run()
        )
        step = lambda: runner.execute(SimpleNamespace())  # noqa: E731
    elif entry_point == "vocoder_step":
        vocoder = stages.CosyVoice3Vocoder(PlacedFakeFlow(device), FakeHiFT())
        scheduler = FunCosyVoice3StreamingVocoderScheduler(vocoder)
        scheduler.select_step_participants = lambda: [("req-stream", object())]
        scheduler.launch_step = lambda participants: held.run()
        step = lambda: scheduler.pump_one_step()  # noqa: E731
    else:
        set_cosyvoice3_preprocessing_context(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            speech_tokenizer=FakeSpeechTokenizer(),
            speaker_encoder=FakeSpeakerEncoder(),
            device=device,
        )

        def prepare_held(**kwargs: object) -> SimpleNamespace:
            held.run()
            return SimpleNamespace(
                state=kwargs["state"],
                flow_embedding=None,
                flow_prompt_speech_token=None,
                flow_prompt_speech_feat=None,
            )

        monkeypatch.setattr(
            request_builders, "prepare_cosyvoice3_request", prepare_held
        )
        payload = make_payload({"text": "hello"})
        step = lambda: preprocess_cosyvoice3_payload(payload)  # noqa: E731

    step_errors: list[Exception] = []

    def run_step() -> None:
        try:
            step()
        except Exception as error:
            step_errors.append(error)
            held.entered.set()

    worker = threading.Thread(target=run_step)
    worker.start()
    assert held.entered.wait(timeout=5.0)
    is_lock_free = MPS_STEP_LOCK.acquire(blocking=False)
    if is_lock_free:
        MPS_STEP_LOCK.release()
    else:
        pass
    held.release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert step_errors == []
    assert is_lock_free == (device_type == "cpu")
