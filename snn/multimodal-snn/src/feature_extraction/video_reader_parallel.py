import cv2
import threading
import queue
from typing import Optional, Tuple, List

class ThreadedVideoReader:
    def __init__(self, path: str, queue_size: int = 96, drop_oldest: bool = True):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.q = queue.Queue(maxsize=queue_size)
        self.drop_oldest = drop_oldest
        self.stopped = False
        self.reader_thread = threading.Thread(target=self._reader, daemon=True)
        self.idx = 0

    def start(self):
        self.reader_thread.start()
        return self

    def _reader(self):
        try:
            while not self.stopped:
                ret, frame = self.cap.read()
                if not ret:
                    break
                try:
                    self.q.put((self.idx, frame), block=False)
                except queue.Full:
                    if self.drop_oldest:
                        try:
                            _ = self.q.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            self.q.put((self.idx, frame), block=False)
                        except queue.Full:
                            pass
                    else:
                        self.q.put((self.idx, frame), block=True)
                self.idx += 1
        finally:
            self.stopped = True
            self.cap.release()
            try:
                self.q.put_nowait((None, None))
            except queue.Full:
                pass

    def read_next(self, timeout: Optional[float] = 0.5) -> Optional[Tuple[int, any]]:
        if self.stopped and self.q.empty():
            return None
        try:
            item = self.q.get(timeout=timeout)
            idx, frame = item
            if idx is None:
                return None
            return item
        except queue.Empty:
            if self.stopped:
                return None
            return None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

def iter_frame_windows(reader: ThreadedVideoReader, win_sec: float = 1.0, hop_sec: float = 0.5):
    fps = reader.fps
    win_frames = max(1, int(round(win_sec * fps)))
    hop_frames = max(1, int(round(hop_sec * fps)))
    buffer: List[Tuple[int, any]] = []

    # Prefill
    while len(buffer) < win_frames:
        item = reader.read_next()
        if item is None:
            break
        buffer.append(item)

    # Emit windows and advance by hop
    while len(buffer) >= win_frames:
        frames = [f for _, f in buffer[:win_frames]]
        yield frames
        advance = min(hop_frames, len(buffer))
        buffer = buffer[advance:]
        while len(buffer) < win_frames:
            item = reader.read_next()
            if item is None:
                break
            buffer.append(item)
