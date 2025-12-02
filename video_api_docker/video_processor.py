import os
import json
import unicodedata
import gc
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, as_completed
import base64
import psutil

import numpy as np
import cv2
from decord import VideoReader, cpu
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from httpx import Client

proc = psutil.Process(os.getpid())

@dataclass
class PlaylistItem:
    index: int
    title: str
    video_path: str

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
    )
    return llm, http_client


class VideoOCRProcessor:
    def __init__(self, video_path: str, crop=(100, 530, 1200, 680)):
        self.video_path = video_path
        self.crop = crop
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
        x1, y1, x2, y2 = self.crop
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
                    "text": "Hãy đọc OCR trong ảnh này, trả về text đúng nhất. Chỉ trả về text, không giải thích thêm. Nếu trong ảnh không có text hoặc không đọc được ocr thì trả về \"\". Nếu trong ảnh có phần bảng tên của người phỏng vấn thì cũng không trả về text ở trong đó.",
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


def save_segments_json(segments, json_output="segments.json"):
    output = serialize_segments(segments)

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def _process_single_video(args):
    """
    Wrapper function cho ProcessPoolExecutor.
    Nhận tuple (index, title, video_path, base_output_dir, scan_step)
    """
    index, title, video_path, base_output_dir, scan_step = args
    
    video_path = Path(video_path)
    output_dir = Path(base_output_dir) / f"{index:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dùng context manager để tự động cleanup sau khi xử lý xong
    with VideoOCRProcessor(str(video_path)) as processor:
        timestamps = processor.scan_video(scan_step=scan_step)
        segments = processor.build_segments(timestamps)

        segments_json = output_dir / f"{index:03d}_segments.json"
        save_segments_json(
            segments=segments,
            json_output=str(segments_json)
        )

        return {
            "index": index,
            "title": title,
            "video_path": str(video_path),
            "segments_json": str(segments_json),
            "segment_count": len(segments),
            "ocr_calls": processor.ocr_calls,
        }


def process_video_item(item: PlaylistItem, base_output_dir: str, scan_step: int = 4):
    return _process_single_video((
        item.index,
        item.title,
        item.video_path,
        base_output_dir,
        scan_step
    ))


def _process_video_worker(video_path: str, scan_step: int, crop, result_queue):
    """Worker chạy trong subprocess riêng để tránh memory leak"""
    try:
        with VideoOCRProcessor(video_path, crop=crop) as processor:
            timestamps = processor.scan_video(scan_step=scan_step)
            segments = processor.build_segments(timestamps)
        result_queue.put(("success", serialize_segments(segments)))
    except Exception as e:
        result_queue.put(("error", str(e)))


def process_video_to_segments(video_path: str, scan_step: int = 4, crop=None):
    """
    Xử lý một video và trả về danh sách segments (không lưu file).
    Chạy trong subprocess riêng để kernel giải phóng hoàn toàn memory sau mỗi request.
    """
    if crop is None:
        crop = (100, 530, 1200, 680)
    
    # Spawn subprocess để xử lý video
    # Khi subprocess kết thúc, kernel giải phóng 100% memory (bao gồm Decord/FFmpeg)
    result_queue = mp.Queue()
    p = mp.Process(target=_process_video_worker, args=(video_path, scan_step, crop, result_queue))
    p.start()
    p.join()  # Đợi subprocess kết thúc
    
    status, data = result_queue.get()
    if status == "error":
        raise RuntimeError(data)
    
    return data


def _iter_batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def process_playlist_items(
    items,
    base_output_dir: str,
    batch_size: int = 4,
    scan_step: int = 4,
    start_index: int | None = None,
    end_index: int | None = None,
):
    os.makedirs(base_output_dir, exist_ok=True)

    filtered_items = [
        item for item in items
        if (start_index is None or item.index >= start_index)
        and (end_index is None or item.index <= end_index)
    ]

    results = []
    total_batches = (len(filtered_items) + batch_size - 1) // batch_size
    
    for batch_idx, chunk in enumerate(_iter_batches(filtered_items, batch_size), start=1):
        batch_results = []

        task_args = [
            (item.index, item.title, item.video_path, base_output_dir, scan_step)
            for item in chunk
        ]
        
        # Mỗi process khi kết thúc sẽ được kernel giải phóng TOÀN BỘ memory
        with ProcessPoolExecutor(max_workers=len(chunk)) as executor:
            future_map = {
                executor.submit(_process_single_video, args): args
                for args in task_args
            }
            for future in as_completed(future_map):
                args = future_map[future]
                index, title = args[0], args[1]
                try:
                    result = future.result()
                    print(f"✓ Processed #{index:03d}: {title}")
                    results.append(result)
                    batch_results.append(result)
                except Exception as exc:
                    print(f"✗ Failed #{index:03d}: {title} -> {exc}")
        
        # Print kết quả sau mỗi batch
        if batch_results:
            print("\n" + "─" * 60)
            print(f"📦 BATCH {batch_idx}/{total_batches} HOÀN THÀNH ({len(batch_results)}/{len(chunk)} thành công)")
            sample = batch_results[0]
            print(f"   Sample: #{sample['index']:03d} - {sample['title']}")
            print(f"           Segments: {sample['segment_count']} | OCR calls: {sample['ocr_calls']}")
            print(f"           JSON: {sample['segments_json']}")
            print(f"   Tổng đã xử lý: {len(results)}/{len(filtered_items)} video")
            print("─" * 60 + "\n")
        
        gc.collect()
    return sorted(results, key=lambda item: item['index'])


def scan_video_directory(video_dir: str):
    """
    Scan thư mục video và tạo danh sách PlaylistItem
    """
    video_dir_path = Path(video_dir)
    if not video_dir_path.exists():
        raise ValueError(f"Thư mục không tồn tại: {video_dir}")
    
    video_extensions = ['.mp4']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(video_dir_path.glob(f"*{ext}"))
    
    video_files.sort()
    
    playlist_items = []
    for idx, video_path in enumerate(video_files, start=1):
        title = video_path.stem
        playlist_items.append(PlaylistItem(
            index=idx,
            title=title,
            video_path=str(video_path)
        ))
    
    return playlist_items


if __name__ == "__main__":
    VIDEO_INPUT_DIR = os.getenv(
        "VIDEO_INPUT_DIR",
        "/home/app/cuonglp1/speech_topic/data/raw",
    )
    
    PROCESSING_OUTPUT_DIR = os.getenv(
        "PROCESSING_OUTPUT_DIR",
        "/home/app/cuonglp1/speech_topic/data/processed_final",
    )
    
    # Các tham số xử lý
    BATCH_SIZE = 1          # Số video xử lý song song
    SCAN_STEP = 4           # Bước nhảy khi scan video (giây)
    START_INDEX = None      # Index video đầu tiên cần xử lý (None = từ đầu)
    END_INDEX = None        # Index video cuối cùng cần xử lý (None = đến hết)

    # Scan thư mục video
    print(f"\n📂 Đang scan thư mục: {VIDEO_INPUT_DIR}")
    playlist_items = scan_video_directory(VIDEO_INPUT_DIR)
    print(f"✓ Tìm thấy {len(playlist_items)} video")
    
    if not playlist_items:
        print("⚠️  Không tìm thấy video nào trong thư mục!")
        exit(1)
    
    # Hiển thị danh sách video
    print("\n Danh sách video:")
    for item in playlist_items[:5]:  # Hiển thị 5 video đầu tiên
        print(f"  #{item.index:03d}: {item.title}")
    if len(playlist_items) > 5:
        print(f"  ... và {len(playlist_items) - 5} video khác")
    
    # Xử lý video
    print(f"\n  Bắt đầu xử lý video...")
    print(f"  - Output directory: {PROCESSING_OUTPUT_DIR}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Scan step: {SCAN_STEP}s")
    print(f"  - Index range: {START_INDEX or 'start'} → {END_INDEX or 'end'}")
    
    processing_results = process_playlist_items(
        playlist_items,
        base_output_dir=PROCESSING_OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        scan_step=SCAN_STEP,
        start_index=START_INDEX,
        end_index=END_INDEX,
    )

