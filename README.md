# Canon — 산업용 화면 인식 시스템

복합기·스마트 디바이스의 LCD 화면을 카메라로 비추면, **AI가 화면 종류를 자동으로 식별**하고
**Target1 → Target2 → Target3 → Target4** 순서대로 검사가 잘 진행됐는지 실시간으로
판정하는 시스템입니다. 운영자는 PC에서 결과를 모니터링하고, 검사 카메라는 휴대폰
하나로 충분합니다.

> **무엇을 위한 시스템인가요?**
> - 공정 마지막 단계에서 “기기 화면이 정해진 순서대로 표시되었는가”를 사람이 일일이
>   확인하던 작업을 자동화합니다.
> - 모든 검사 결과는 SQLite에 영구 저장되어, 사후에 “어떤 frame이 왜 모호했는지”까지
>   추적할 수 있습니다.

---

## 무엇을 할 수 있나

세 가지 진입점으로 검사를 수행할 수 있습니다.

| 모드 | 설명 |
|---|---|
| 📱 **휴대폰 카메라** | 휴대폰을 검사 카메라로 사용. 실시간으로 화면을 비추면 PC 모니터에 결과 표시 |
| 🖼 **이미지 파일** | 정지 이미지 여러 장을 업로드 → 좌우 캐러셀로 한 장씩 결과 확인 |
| 🎬 **영상 파일** | 녹화된 영상을 업로드 → 영상이 재생되면서 옆 사이드바에 frame별 검사 결과 progressive 표시 |

추가로 PC 화면에서 **카메라 송출 강제 종료**, **검사 로그 (SQLite) 조회**, **단일 frame
재검사**, **관리자 override** 가 가능합니다.

---

## 핵심 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 백엔드 | Python 3.12, FastAPI, uvicorn, **PyTorch (CUDA)**, ultralytics(YOLO), OpenCV |
| 모델 | YOLO11n(화면 검출) + ResNet18 ×4 (Target 1~4 binary classifier) |
| 프론트엔드 | Vite, React 19, TypeScript, Tailwind CSS, Zustand |
| 통신 | WebSocket(실시간 frame), REST API(파일 업로드/로그) |
| 저장 | SQLite (frame별 추론 결과 영구 보관) |

---

## 설치 가이드

> 처음 한 번만 실행하면 됩니다. 이미 설치돼 있으면 [실행 방법](#실행-방법)으로 건너뛰세요.

### 0. 시스템 요구사항

- Linux (또는 macOS) — 본 가이드는 Linux 기준
- NVIDIA GPU + CUDA 드라이버 (없어도 동작은 하지만 매우 느림)
- 대략 10GB 디스크 여유 (모델 + Python 의존성)

다음 명령으로 GPU가 보이는지 확인합니다.

```bash
nvidia-smi
```

### 1. 백엔드 설치 — Python + uv

```bash
# (1) uv 설치 (한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 셸을 새로 열거나 PATH 갱신
export PATH="$HOME/.local/bin:$PATH"
uv --version   # 버전이 찍히면 OK

# (2) 백엔드 의존성 설치
cd backend
uv sync
```

> `uv sync`는 첫 실행 시 PyTorch·OpenCV·ultralytics 등 GB 단위 패키지를 받아오므로
> 5~10분 걸릴 수 있습니다.

### 2. 프론트엔드 설치 — Node 22 + npm

Node.js 20 이상이 필요합니다. nvm으로 사용자 영역에 설치하는 방법입니다.

```bash
# (1) nvm 설치 (한 번만)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# 셸 재시작 또는 즉시 활성화
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"

# (2) Node 22 LTS 설치 후 활성화
nvm install 22 --lts
nvm use 22

# (3) 프론트엔드 의존성 설치
cd frontend
npm install
```

### 3. HTTPS 인증서 (모바일에서 카메라 사용을 위해 필수)

휴대폰 브라우저는 **HTTPS가 아니면 카메라 권한을 주지 않습니다.** 본 저장소는
`frontend/certs/`에 self-signed 인증서를 포함하고 있어 별도 설정 없이 동작합니다.

직접 새로 만들고 싶다면:

```bash
cd frontend/certs
openssl req -x509 -newkey rsa:2048 -keyout local-key.pem -out local-cert.pem -days 365 -nodes -subj "/CN=localhost"
```

---

## 실행 방법

백엔드와 프론트엔드를 **각각 다른 터미널**에서 실행합니다.

### 터미널 1 — 백엔드

```bash
cd backend
uv run python -m app.main
```

다음과 같은 메시지가 보이면 성공입니다.

```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

> 부팅 시 GPU에 모델을 미리 올리는 워밍업이 1~3초 정도 걸립니다.

### 터미널 2 — 프론트엔드

```bash
cd frontend
npm run dev
```

다음과 같은 메시지가 보이면 성공입니다.

```
  VITE v8.x  ready in N ms
  ➜  Local:   https://localhost:5173/
  ➜  Network: https://<서버IP>:5173/
```

---

## 접속 방법

### PC 모니터링 화면

같은 네트워크의 PC 브라우저에서:

```
https://localhost:5173
```

또는 다른 PC에서 서버 IP로:

```
https://<서버IP>:5173
```

> **첫 접속 시** “안전하지 않음” 경고가 뜹니다. self-signed 인증서이므로 정상이며,
> **고급 → 안전하지 않음 계속**으로 진행하세요.

### 휴대폰 카메라 사용

1. 휴대폰을 **PC와 같은 Wi-Fi**에 연결
2. 휴대폰 브라우저(Chrome / Safari)에서 `https://<서버IP>:5173` 접속
3. 인증서 경고 → **안전하지 않음 계속** (PC와 동일)
4. 화면 중앙의 큰 빨간 버튼 **「이 휴대폰으로 검사 시작」** 탭
5. 카메라 권한 허용
6. 화면에 표시되는 **「송출 시작」** 누르면 PC 화면에 실시간 영상이 나타남

---

## 사용 가이드 (3가지 시나리오)

### 시나리오 A — 휴대폰으로 실시간 검사

1. PC 화면을 띄워둠
2. 휴대폰으로 위 [접속 방법](#휴대폰-카메라-사용) 따라 송출 시작
3. 카메라로 검사할 기기 화면을 비춤 → PC 화면에 영상이 나타나며
   **하단의 T1·T2·T3·T4 칸이 화면이 인식될 때마다 채워짐**
4. PC 화면 중 카메라 패널 우측의 **「⏻ 끊기」** 버튼으로 송출 강제 종료 가능
5. 휴대폰을 다른 사람이 들고 있어도 PC에서 끊을 수 있음

### 시나리오 B — 이미지 파일로 분석

1. PC 화면 헤더의 **「파일로 테스트」** 버튼 클릭 (또는 가운데 가이드 카드의 「이미지 파일」)
2. 모달 안에서 **「로컬 검사」** 클릭 → 이미지 파일 선택 (여러 장 가능)
3. 캐러셀 모달이 열리고 좌우 화살표(또는 ←/→ 키)로 이미지 전환
4. 사이드바에서 각 이미지의 예측 라벨, 분류기 점수, 판정 사유를 확인

### 시나리오 C — 영상 파일로 분석

1. 위와 같이 **「파일로 테스트」** → 영상 파일 선택
2. 영상이 즉시 재생되며 옆 사이드바에 frame 단위 결과가 progressive로 추가됨
3. ×1 / ×2 / ×4 / ×8 재생 속도로 분석 시간 단축
4. 영상 currentTime에 맞춰 사이드바 step bar가 자동 동기화

---

## 자주 묻는 문제 (트러블슈팅)

| 증상 | 원인 / 해결 |
|---|---|
| 휴대폰에서 카메라 권한이 안 떠요 | HTTP가 아닌 **HTTPS** 주소로 접속했는지 확인. `https://...` 로 |
| PC 화면이 OFFLINE으로 굳어요 | 백엔드를 재시작했을 때 일시적 발생. **자동 1.5초 후 재연결** 시도. 그래도 안 되면 페이지 새로고침 |
| 영상 frame이 잘 인식 안 돼요 | 영상이 또렷할수록 모델 false positive가 강해질 수 있음. backend 로그에서 `decision=ambiguous` 줄을 보고 점수 분포 확인 후 `backend/app/core/config.py`의 `DECISION_MARGIN` 또는 `TARGET_x_THRESHOLD` 조정 |
| 같은 휴대폰 두 대가 동시에 송출하면 한쪽이 안 보임 | 카메라 ID가 같으면 합쳐집니다. 첫 접속 시 **자동으로 디바이스별 고유 ID(localStorage)** 부여. 그래도 충돌하면 송출 화면에서 카메라 ID를 직접 다르게 입력 |
| 백엔드를 끄려는데 “Waiting for connections to close”에서 멈춤 | 활성 WebSocket 때문. `Ctrl+C`를 다시 누르거나 `pgrep -af app.main` 후 `kill` |
| GPU가 안 쓰여요 (느림) | `nvidia-smi`로 드라이버 확인. `backend/app/core/config.py`의 `device: str = "cuda"` 설정인지 확인. CPU만 있는 환경이면 `cpu`로 두면 동작은 됨 |

---

## 디렉토리 구조

```
canon-temp/
├── backend/                         # Python 백엔드
│   ├── pyproject.toml               # 의존성 정의 (uv 관리)
│   ├── uv.lock                      # 의존성 lock
│   ├── app/
│   │   ├── main.py                  # FastAPI 진입점
│   │   ├── api/                     # REST + WebSocket 라우터
│   │   ├── service/                 # 추론 파이프라인 / state machine / SQLite
│   │   ├── models/                  # ResNet18 / YOLO 모델 코드
│   │   └── core/config.py           # 임계값·디바이스 등 설정
│   ├── assets/weight/               # 모델 가중치
│   │   ├── yolo/                    # 화면 검출용 YOLO weight
│   │   └── target_1..4/             # Target 분류기 weight
│   └── data/db/operational_runs.sqlite3   # 검사 로그 (자동 생성)
│
├── frontend/                        # React + Vite 프론트엔드
│   ├── package.json
│   ├── vite.config.ts               # /api, /ws 프록시 + HTTPS 설정
│   ├── certs/                       # 개발용 self-signed 인증서
│   └── src/
│       ├── App.tsx                  # 모든 화면 (대시보드/모달/모바일 송출)
│       ├── store.ts                 # 전역 상태 (zustand)
│       └── locales.ts               # 언어 리소스 (한/영)
│
└── README.md                        # 본 문서
```

---

## 설정 핵심 값 (필요 시 조정)

`backend/app/core/config.py` 안의 값들이 검사 동작을 좌우합니다.

```python
TARGET1_THRESHOLD = 0.90    # 각 Target 분류기의 통과 임계값
TARGET2_THRESHOLD = 0.75
TARGET3_THRESHOLD = 0.90
TARGET4_THRESHOLD = 0.90
DECISION_MARGIN   = 0.005   # top1 vs top2 점수 차 최소값 (작을수록 너그러움)
DELTA_ACCEPT      = 0.05    # 단일 통과 시 임계값을 얼마나 넉넉히 넘겨야 하는지
REINSPECT_WINDOW  = 3       # 재검사할 frame 개수
device            = "cuda"  # GPU 사용. CPU만 있으면 "cpu"
```

수정 후 백엔드는 자동 reload (`uvicorn --reload`)되므로 별도 재기동 불필요.

---

## 검사 로그 직접 보기 (운영/감사용)

모든 추론 결과는 `backend/data/db/operational_runs.sqlite3` 에 영구 저장됩니다.

### 빠른 통계

```bash
sqlite3 -header -column backend/data/db/operational_runs.sqlite3 "
select json_extract(response_json,'\$.predicted_label') as label,
       json_extract(response_json,'\$.decision_type') as decision,
       count(*) as count
from frame_results
group by label, decision
order by count desc;
"
```

### 의심(오탐 가능) frame만 추리기

```bash
sqlite3 -header -column backend/data/db/operational_runs.sqlite3 "
select id, substr(timestamp,12,8) as time,
       json_extract(response_json,'\$.predicted_label') as pred,
       json_extract(response_json,'\$.decision_type') as decision,
       round(json_extract(response_json,'\$.decision.top1_score'),3) as top1,
       round(json_extract(response_json,'\$.decision.top2_score'),3) as top2,
       round(json_extract(response_json,'\$.decision.margin'),4) as margin
from frame_results
where json_extract(response_json,'\$.ambiguous')=1
   or json_extract(response_json,'\$.reinspect_needed')=1
order by id desc limit 30;
"
```

> 더 보기 좋은 GUI를 원하면 [DB Browser for SQLite](https://sqlitebrowser.org)에서
> 위 DB 파일을 열어보세요.

---

## 한눈에 보는 동작 흐름

```
[휴대폰 카메라]
      │  매 200ms JPEG
      ▼
[/ws/source]──────────────┐
                          ▼
              ┌──────────────────────────────┐
              │  YOLO 화면 검출              │
              │  → 크롭/와핑/정규화          │
              │  → ResNet18 ×4 (yes/no)      │
              │  → DecisionEngine            │
              │  → SequenceStateMachine      │
              └──────────┬───────────────────┘
                         │
            ┌────────────┼────────────┐
            ▼                         ▼
  [SQLite frame_results]      [PC 대시보드 (/ws)]
   (영구 저장)                 (실시간 영상 + 단계 표시)
```

---

문의나 버그 제보는 GitHub Issues로 부탁드립니다.
