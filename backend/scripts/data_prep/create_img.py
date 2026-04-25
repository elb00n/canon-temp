from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
SOURCE_VIDEO_DIR = BASE_DIR / "sample_video"
OUTPUT_ROOT_DIR = BASE_DIR / "sample_images_from_videos"


def extract_frames(video_path, output_folder, start_sec=0, end_sec=None, interval_sec=4.0):
    """영상에서 일정 간격으로 프레임을 추출한다."""
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError(f"FPS 정보를 읽을 수 없습니다: {video_path}")

    duration_sec = total_frames / fps
    if end_sec is None or end_sec > duration_sec:
        end_sec = duration_sec

    start_frame = max(0, int(start_sec * fps))
    end_frame = min(total_frames - 1, int(end_sec * fps))
    frame_step = max(1, int(fps * interval_sec))

    print(f"영상: {video_path.name}")
    print(f"영상 총 길이: {duration_sec:.2f}초")
    print(f"추출 범위: {start_sec}초 ~ {end_sec}초")
    print(f"추출 간격: {interval_sec}초 마다 1장")
    print("---------------------------------------")

    saved_count = 0
    current_frame = start_frame

    while current_frame <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame / fps
        filename = f"{video_path.stem}_frame_{saved_count:04d}_time_{timestamp:.1f}s.jpg"
        save_path = output_folder / filename
        cv2.imwrite(str(save_path), frame)
        saved_count += 1
        print(f"저장 중... [{saved_count}] {filename}")

        current_frame += frame_step

    cap.release()
    print("---------------------------------------")
    print(f"작업 완료! 총 {saved_count}장의 이미지가 저장되었습니다.")


def extract_all_sample_videos(source_dir=SOURCE_VIDEO_DIR, output_root=OUTPUT_ROOT_DIR, interval_sec=4.0):
    """sample_video 폴더 안의 mp4 파일들을 모두 프레임 추출한다."""
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(source_dir.glob("*.mp4"))
    if not video_files:
        print(f"처리할 mp4 파일이 없습니다: {source_dir}")
        return

    for video_file in video_files:
        output_folder = output_root / video_file.stem
        extract_frames(video_file, output_folder, interval_sec=interval_sec)


if __name__ == "__main__":
    extract_all_sample_videos()