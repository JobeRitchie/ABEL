"""The slow-video warning fires for network locations only, never local or USB drives."""
import sys

import pytest

from abel.services import context_feature_service as cfs


def test_unc_path_is_network():
    root = cfs._network_video_root(r"\\server\share\videos\a.avi")
    assert root == r"\\server\share"


def test_relative_path_is_not_network():
    assert cfs._network_video_root("videos/a.avi") is None


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows only")
def test_system_drive_is_not_network():
    assert cfs._network_video_root(r"C:\videos\a.avi") is None


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows only")
def test_drive_type_decides(monkeypatch):
    import ctypes

    class _Kernel:
        def __init__(self, drive_type):
            self._drive_type = drive_type

        def GetDriveTypeW(self, _root):
            return self._drive_type

    class _Windll:
        def __init__(self, drive_type):
            self.kernel32 = _Kernel(drive_type)

    # 4 = DRIVE_REMOTE (mapped network drive), 3 = DRIVE_FIXED (internal or USB SSD).
    monkeypatch.setattr(ctypes, "windll", _Windll(4))
    assert cfs._network_video_root(r"j:\lab\a.avi") == "J:"
    monkeypatch.setattr(ctypes, "windll", _Windll(3))
    assert cfs._network_video_root(r"G:\lab\a.avi") is None
