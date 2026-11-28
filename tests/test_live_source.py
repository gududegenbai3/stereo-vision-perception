"""Check complete-frame handling and reconnect without depending on socket permissions."""

from collections.abc import Iterator
from io import BytesIO
import json
from pathlib import Path
import socket
import tempfile
from threading import Event
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from src.live_source import Frame, LatestFrame, read_raw_frames, receive_camera


def simulated_connection(chunks: list[bytes]) -> MagicMock:
    """Model the socket recv_into contract, including EOF on the last read."""
    connection: MagicMock = MagicMock()
    remaining: Iterator[bytes] = iter(chunks)

    def recv_into(buffer: memoryview) -> int:
        chunk: bytes = next(remaining, b"")
        buffer[:len(chunk)] = chunk
        return len(chunk)

    connection.recv_into.side_effect = recv_into
    return connection


class LiveSourceTests(unittest.TestCase):
    def test_pipe_discards_partial_image_after_complete_frame(self) -> None:
        latest: LatestFrame = LatestFrame()
        # One 2x4 BGR frame, followed by a partial frame.
        pipe: BytesIO = BytesIO(bytes([5]) * 24 + bytes([7]) * 12)
        read_raw_frames(pipe, (4, 2), latest, Event())
        frame: Frame | None = latest.snapshot()[0]
        assert frame is not None
        self.assertEqual(frame.sequence, 1)
        self.assertEqual(frame.image.shape, (2, 4, 3))
        self.assertTrue(np.all(frame.image == 5))


    def test_reconnect_starts_new_epoch_and_does_not_publish_partial_frame(self) -> None:
        stop: Event = Event()
        latest: LatestFrame = LatestFrame()
        seen: list[Frame] = []
        first: MagicMock = simulated_connection([bytes([6]) * 8, bytes([6]) * 16, bytes([1]) * 12])
        second: MagicMock = simulated_connection([bytes([9]) * 24])
        sockets: Iterator[MagicMock] = iter((first, second))
        listener: MagicMock = MagicMock()
        listener.__enter__.return_value = listener
        listener.getsockname.return_value = ("127.0.0.1", 12345)

        def accept() -> tuple[MagicMock, tuple[str, int]]:
            self.assertIsNone(latest.snapshot()[0])
            connection: MagicMock | None = next(sockets, None)
            if connection is None:
                stop.set()
                raise socket.timeout()
            return connection, ("127.0.0.1", 0)

        def record(image: NDArray[np.uint8]) -> None:
            LatestFrame.publish(latest, image)
            frame: Frame | None = latest.snapshot()[0]
            assert frame is not None
            seen.append(frame)

        listener.accept.side_effect = accept
        with tempfile.TemporaryDirectory() as directory:
            root: Path = Path(directory)
            with patch("src.live_source.socket.socket", return_value=listener), patch.object(latest, "publish", side_effect=record):
                receive_camera(root, (4, 2), latest, stop)
            self.assertEqual(json.loads((root / "receiver.json").read_text())["port"], 12345)
            self.assertEqual(json.loads((root / "camera-connected.json").read_text()), {
                "backend": "external-ffmpeg-tcp", "frame": 2, "size_wh": [4, 2],
            })
        self.assertEqual([frame.sequence for frame in seen], [1, 2])
        self.assertEqual([frame.epoch for frame in seen], [0, 1])
        self.assertTrue(np.all(seen[0].image == 6))
        self.assertTrue(np.all(seen[1].image == 9))


if __name__ == "__main__":
    unittest.main()
