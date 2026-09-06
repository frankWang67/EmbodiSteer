"""SpaceMouse dependency/startup failures, without starting an input process."""

from unittest.mock import Mock, call

import pytest

from umi.real_world import spacemouse_shared_memory as module


@pytest.mark.parametrize("error", [ModuleNotFoundError("spnav"), OSError("libspnav.so")])
def test_missing_spacemouse_dependency_fails_before_shared_memory(monkeypatch, error):
    monkeypatch.setattr(module, "SPNAV_IMPORT_ERROR", error)
    allocate = Mock(side_effect=AssertionError("No shared memory should be allocated"))
    monkeypatch.setattr(module.SharedMemoryRingBuffer, "create_from_examples", allocate)
    with pytest.raises(RuntimeError, match="spacemouse-requirements.txt") as exc:
        module.Spacemouse(shm_manager=None)
    assert exc.value.__cause__ is error
    allocate.assert_not_called()


@pytest.mark.parametrize("still_alive", [False, True])
def test_daemon_startup_failure_times_out_and_cleans_up(monkeypatch, still_alive):
    # Bypass construction entirely: no shared memory, daemon or child process.
    controller = module.Spacemouse.__new__(module.Spacemouse)
    controller.launch_timeout = 0.1
    controller.ready_event = Mock()
    controller.ready_event.wait.return_value = False
    controller.stop_event = Mock()
    controller.join = Mock()
    controller.is_alive = Mock(return_value=still_alive)
    controller.terminate = Mock()
    start_process = Mock()
    monkeypatch.setattr(module.mp.Process, "start", start_process)

    with pytest.raises(RuntimeError, match="spacenavd"):
        controller.start()

    start_process.assert_called_once()
    controller.ready_event.wait.assert_called_once_with(0.1)
    controller.stop_event.set.assert_called_once()
    assert controller.join.call_args_list == [call(timeout=0.1)] * (2 if still_alive else 1)
    assert controller.terminate.call_count == int(still_alive)


def test_ready_input_process_is_not_stopped(monkeypatch):
    controller = module.Spacemouse.__new__(module.Spacemouse)
    controller.launch_timeout = 0.1
    controller.ready_event = Mock()
    controller.ready_event.wait.return_value = True
    controller.stop_event = Mock()
    controller.join = Mock()
    monkeypatch.setattr(module.mp.Process, "start", Mock())

    controller.start()

    controller.ready_event.wait.assert_called_once_with(0.1)
    controller.stop_event.set.assert_not_called()
    controller.join.assert_not_called()
