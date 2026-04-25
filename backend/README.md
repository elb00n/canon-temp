## Canon Backend

산업 현장용 화면 인식 백엔드입니다. 현재 구조는 다음 흐름을 기준으로 정리되어 있습니다.

- YOLO로 화면 후보를 찾음
- crop 후 warping으로 입력을 정규화함
- target별 ResNet18 yes/no 분류를 수행함
- 영상에서는 target_1 -> target_2 -> target_3 -> target_4 순서로 순차 판정함

모델 가중치는 모두 [assets/weight](assets/weight) 아래에 둡니다.

## 프로젝트 구조

```text
app/
	core/        공용 설정, 경로, 순차 검출 보조 코드
	models/      ResNet18, YOLO warp, OpenVINO 변환 관련 모델 코드
	service/     target test, sequence video 같은 서비스 계층
images/        데이터 준비용 입력/출력 폴더
scripts/       실행용 CLI 스크립트
outputs/       실행 결과 저장 폴더
assets/weight/ 영구 보관용 가중치 폴더
```

## 환경 준비

Python 3.12 기준으로 동작합니다. Windows에서는 아래 순서로 준비하면 됩니다.

```powershell
uv sync
```

필요하면 가상환경을 직접 활성화해서 써도 됩니다.

```powershell
& .\.venv\Scripts\Activate.ps1
```

## 의존성

런타임 핵심 패키지:

- `numpy`
- `opencv-python`
- `pillow`
- `torch`
- `torchvision`
- `ultralytics`
- `openvino`

개발용 패키지:

- `pytest`
- `ruff`

버전은 [pyproject.toml](pyproject.toml)과 `uv.lock`을 기준으로 맞춥니다.

## 데이터와 가중치 위치

### 입력 데이터

- [images/target_image](images/target_image): target 원본 이미지
- [images/target_images](images/target_images): 리사이즈/패딩된 target 이미지 캐시
- [images/sample_images](images/sample_images): 샘플 이미지
- [images/sample_images_from_videos](images/sample_images_from_videos): 영상에서 추출한 샘플 프레임
- [images/sample_video](images/sample_video): 테스트용 영상 원본

### 가중치

- [assets/weight/yolo/yolo11n.pt](assets/weight/yolo/yolo11n.pt): 기본 YOLO weight
- [assets/weight/target_1/best.pt](assets/weight/target_1/best.pt): target_1 PyTorch weight
- [assets/weight/target_2/best.pt](assets/weight/target_2/best.pt): target_2 PyTorch weight
- [assets/weight/target_3/best.pt](assets/weight/target_3/best.pt): target_3 PyTorch weight
- [assets/weight/target_4/best.pt](assets/weight/target_4/best.pt): target_4 PyTorch weight
- OpenVINO IR은 각 target 폴더 아래 `target_x_weight_openvino/model.xml` 형태로 저장됩니다.

## 대표 실행 명령

### 1. target 분류 테스트

YOLO detection 후 warping하고, yes/no 분류 결과를 저장합니다.

```powershell
uv run python scripts/target_test.py --target-name target_1
```

자주 쓰는 옵션:

- `--source`: 입력 폴더 여러 개 지정
- `--threshold`: target별 yes/no 임계값 수동 오버라이드
- `--conf`: YOLO confidence threshold
- `--imgsz`: YOLO 입력 크기
- `--save-crops`: crop 저장
- `--save-contour-warped`: contour 기반 warped 이미지 저장

### 2. 순차 영상 검출

영상에서 target을 순서대로 판정합니다.

```powershell
uv run python scripts/run_sequence_video.py --source images/sample_video
```

자주 쓰는 옵션:

- `--target-order`: 순서 지정
- `--frame-step`: 프레임 간격 제어
- `--sample-seconds`: 영상 FPS 기준 샘플링 간격
- `--min-consecutive`: 연속 yes 판정 수
- `--max-missed`: missed 허용 횟수
- `--save-confirmed-frames`: 확정 프레임만 저장

### 3. OpenVINO 변환

target ResNet18 가중치를 OpenVINO IR로 변환합니다.

```powershell
uv run python scripts/openvino.py
```

기본적으로 [assets/weight](assets/weight) 아래의 `target_*` 폴더를 찾아 변환합니다.

### 4. 학습

각 target의 yes/no 폴더로 ResNet18 binary classifier를 학습합니다.

```powershell
uv run python scripts/train_target.py --target-name target_1
```

입력 폴더 구조 예시:

```text
images/cnn_train/target_1/
images/cnn_train/target_1_no/
```

### 5. 데이터 준비

프레임 추출, 자동 라벨링, 증강은 `scripts/data_prep` 계열을 사용합니다.

```powershell
uv run python scripts/data_prep/create_img.py
uv run python scripts/data_prep/labeling.py
uv run python scripts/data_prep/agumentation.py
```

### 6. 샘플 warping 테스트

```powershell
uv run python scripts/test_warping_samples.py
```

### 7. 영상 노이즈 테스트

```powershell
uv run python scripts/video_noise.py
```

### 8. target 이미지 리사이즈/패딩

```powershell
uv run python scripts/resizing.py
```

## 결과 폴더 규칙

실행 결과는 `outputs/` 아래에 timestamp 기반으로 저장합니다.

- `outputs/target_test_runs/`: target test 결과
- `outputs/sequence_video_runs/`: 순차 영상 검출 결과
- `outputs/warping_test_runs/`: warping 검증 결과
- `outputs/video_noise_runs/`: 노이즈 영상 결과
- `outputs/labeling_runs/`: 자동 라벨링 결과
- `outputs/template_matcher_v1/`: 오래된 template matching 결과

각 실행은 보통 아래처럼 정리됩니다.

- 요약 JSON 1개
- 필요 시 CSV 1개
- 프리뷰 이미지 또는 저장 프레임 폴더

## 코드 기준

- 공용 경로는 [app/core/paths.py](app/core/paths.py)를 사용합니다.
- target 추론은 [app/service/target_service.py](app/service/target_service.py)를 사용합니다.
- 순차 영상 추론은 [app/service/sequence_service.py](app/service/sequence_service.py)를 사용합니다.
- target test는 [app/service/target_test_service.py](app/service/target_test_service.py)를 사용합니다.

## 메모

- `scripts/`는 CLI 진입점만 남기는 방향으로 정리했습니다.
- `assets/weight`를 canonical weight root로 사용합니다.
- OpenVINO가 있으면 우선 사용하고, 없으면 PyTorch로 fallback합니다.
- 기존 실험용 파일은 남아 있을 수 있지만, 현재 주력 흐름은 service 계층입니다.
