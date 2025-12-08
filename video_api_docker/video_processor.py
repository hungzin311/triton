import os
import base64
import unicodedata
import gc
import multiprocessing as mp
from difflib import SequenceMatcher

import numpy as np
import cv2
from decord import VideoReader, cpu
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from httpx import Client

def normalized(text):
    if text is None:
        return ""

    s = text.strip()
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = " ".join(s.split())
    s = s.lower()
    s = s.replace(' - ', '-')

    s = unicodedata.normalize('NFD', s)
    s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')

    return s


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_similar_text(a, b, sim_threshold=0.85):
    na, nb = normalized(a), normalized(b)
    if na == "" and nb == "":
        return True
    sim = similarity(na, nb)
    return sim >= sim_threshold


def numpy_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    img_bytes = buffer.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64


def create_llm(temperature=0.0):
    """Tạo LLM client mới (dùng trong mỗi process)"""
    http_client = Client(verify=False)
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        openai_api_base=os.getenv("OPENAI_API_BASE", "http://localhost:8233/v1"),
        openai_api_key="your_api_key",
        http_client=http_client,
        temperature=temperature,
        max_tokens = 200,
    )
    return llm, http_client


class VideoOCRProcessor:
    def __init__(self, video_path: str, crop=(100, 530, 1200, 680)):
        self.video_path = video_path
        self.vr = VideoReader(video_path, ctx=cpu(0))
        self.total_frames = len(self.vr)
        self.video_fps = float(self.vr.get_avg_fps())
        self.ocr_calls = 0
        self.ocr_cache = {}
        self._llm, self._http_client = create_llm()
    
    def cleanup(self):
        self.ocr_cache.clear()
        
        # Đóng httpx client
        if hasattr(self, '_http_client') and self._http_client:
            try:
                self._http_client.close()
            except:
                pass
        
        # Xóa LLM
        if hasattr(self, '_llm'):
            del self._llm
        
        if hasattr(self, 'vr'):
            try:
                self.vr.seek(0)  # Reset position
            except:
                pass
            del self.vr
            gc.collect()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def read_frame(self, time_point: float):
        frame_index = int(time_point * self.video_fps) + 1
        frame_index = min(frame_index, self.total_frames - 1)

        if frame_index < 0 or frame_index >= self.total_frames:
            return None

        frame = self.vr[frame_index].asnumpy()
        h,w, _ = frame.shape
        x1, y1, x2, y2 = (100, int(h*2/3), w, h-50)
        return frame[y1:y2, x1:x2]

    def ocr(self, frame_time: float):
        if frame_time in self.ocr_cache:
            return self.ocr_cache[frame_time]

        frame = self.read_frame(frame_time)
        if frame is None:
            return ""

        self.ocr_calls += 1
        img_b64 = numpy_to_base64(frame)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Hãy trích xuất văn bản (OCR) từ hình ảnh. Chỉ trả về đúng nội dung văn bản. Nếu không có chữ hoặc không thể nhận dạng được, trả về 'KHÔNG CÓ CHỮ'.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        )

        response = self._llm.invoke([message])
        text = response.content
        self.ocr_cache[frame_time] = text
        return text

    def binary_segmentation(self, left_time, right_time, left_text=None, right_text=None, threshold=0.5, sim_threshold=0.9):
        """Phân đoạn nhị phân để tìm điểm chuyển text"""
        if left_text is None:
            left_text = self.ocr(left_time)
        if right_text is None:
            right_text = self.ocr(right_time)

        if is_similar_text(left_text, right_text, sim_threshold):
            return []

        if right_time - left_time <= threshold:
            return [{"end": int(right_time), "text": left_text}]

        mid_time = (left_time + right_time) / 2
        mid_text = self.ocr(mid_time)

        return (
            self.binary_segmentation(left_time, mid_time, left_text, mid_text, threshold, sim_threshold)
            + self.binary_segmentation(mid_time, right_time, mid_text, right_text, threshold, sim_threshold)
        )

    def scan_video(self, scan_step=4):
        """Scan video để tìm tất cả timestamps thay đổi text"""
        video_duration = int(self.total_frames / self.video_fps)
        timestamps = []

        prev_time = 1
        prev_text = self.ocr(prev_time)

        for t in range(scan_step, video_duration + scan_step, scan_step):
            curr_time = min(t, video_duration - 1)
            curr_text = self.ocr(curr_time)

            if curr_text != prev_text:
                change_times = self.binary_segmentation(prev_time, curr_time, prev_text, curr_text)
                timestamps.extend(change_times)

            prev_time = curr_time
            prev_text = curr_text

        return timestamps

    @staticmethod
    def build_segments(timestamps):
        """Xây dựng segments với start và end time"""
        timestamps_with_start = []
        prev_end = 0.0

        for seg in timestamps:
            start = prev_end
            end = seg['end']
            text = seg['text']
            timestamps_with_start.append({
                "start": start,
                "end": end,
                "text": text
            })
            prev_end = end

        return timestamps_with_start


def serialize_segments(segments):
    output = []

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        text = seg.get("text", "")

        output.append({
            "id": i,
            "start": start,
            "end": end,
            "text": text,
        })

    return output

def _process_video_worker(video_path: str, scan_step: int, crop, result_queue):
    """Worker chạy trong subprocess riêng để tránh memory leak"""
    try:
        with VideoOCRProcessor(video_path, crop=crop) as processor:
            timestamps = processor.scan_video(scan_step=scan_step)
            segments = processor.build_segments(timestamps)
        result_queue.put(("success", serialize_segments(segments)))
    except Exception as e:
        result_queue.put(("error", str(e)))


def process_video_to_segments(video_path: str, scan_step: int = 4, crop=None, timeout: int = 600):

    if crop is None:
        crop = (100, 530, 1200, 680)
    
    # Khi subprocess kết thúc, kernel giải phóng 100% memory (bao gồm Decord/FFmpeg)
    result_queue = mp.Queue()
    p = mp.Process(target=_process_video_worker, args=(video_path, scan_step, crop, result_queue))
    p.start()
    p.join(timeout=timeout)  # Đợi subprocess với timeout
    
    # Kiểm tra nếu process vẫn đang chạy (quá timeout)
    if p.is_alive():
        # Kill process
        p.terminate()
        p.join(timeout=5)  
        
        # Nếu vẫn không chết, force kill
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        
        raise TimeoutError(f"Video processing timed out after {timeout} seconds ({timeout/60:.1f} minutes)")
    
    try:
        status, data = result_queue.get_nowait()
    except:
        raise RuntimeError("Process ended without returning result")
    
    if status == "error":
        raise RuntimeError(data)
    
    return data
