# 🏭 캐논 코리아 스마트 팩토리 UI 프로젝트 가이드라인

## 1. 프로젝트 개요 (Project Overview)
- **목적**: 캐논 코리아 공장 현장 생산 관리 및 모니터링 소프트웨어
- **대상**: 공장 현장 작업자 (컴퓨터 조작이 서툴 수 있음)
- **핵심 가치**: **가시성(Visibility), 직관성(Intuitiveness), 안정성(Stability)**
- **특이사항**: 실시간 영상 스트리밍 및 이미지 데이터 처리 필요, 최종 Electron 패키징

## 2. 기술 스택 (Technical Stack)
- **Framework**: `Next.js (App Router)` - 고성능 렌더링 및 구조적 이점
- **Language**: `TypeScript` - 엄격한 타입 체크를 통한 현장 데이터 오류 방지
- **Styling**: `Tailwind CSS` - 빠르고 일관된 디자인 유틸리티 활용
- **State Management**:
  - Server State: `TanStack Query (React Query)` (실시간 데이터 동기화)
  - Client State: `Zustand` (가벼운 전역 상태 관리)
- **Visuals**:
  - Charts: `Apache ECharts` (대용량 데이터 시각화 성능 우수)
  - Icons: `Lucide React` (직관적인 아이콘 셋)
- **Distribution**: `Electron` (데스크탑 앱 패키징 및 로컬 리소스 접근)

## 3. 디자인 시스템 및 UI/UX 규칙 (Industrial Design Rules)
- **Industrial Look**: 모든 UI 요소는 **각진 형태(Squared UI, Border-radius: 0)**를 기본으로 하여 단단하고 전문적인 느낌을 줌
- **Color Palette**: 
  - **Main**: 캐논 브랜드 컬러 기반 (Grey 배경, White/Light Grey 카드, Red 포인트)
  - **Status Indicators**: 
    - 정상: `#22c55e` (Green)
    - 주의: `#f59e0b` (Amber)
    - 위험/정지: `#ef4444` (Red)
    - 미연결: `#64748b` (Slate)
- **Typography**: 가독성이 최우선인 고대비 폰트 (예: Inter, Roboto, Pretendard)
- **Layout**: 
  - **High Density**: 여백을 최소화하여 한 화면에 최대한 많은 정보를 정확하게 노출
  - **Large Targets**: 마우스 조작이 힘든 환경을 고려해 버튼 및 클릭 요소는 충분히 크게 설정
- **Media**: 실시간 영상 및 이미지는 지연 시간(Latency) 최소화에 집중

## 4. 코딩 컨벤션 (Coding Conventions)
- **Simplicity**: 복잡한 추상화보다 읽기 쉽고 명확한 코드 지향
- **Reusability**: `components/shared` 또는 `components/ui` 폴더에 재사용 가능한 원자 단위 컴포넌트 구성
- **Performance**: 실시간 이미지 데이터 처리 시 불필요한 리렌더링 방지 (`memo`, `useCallback` 적극 활용)
- **Error Handling**: 모든 오류 상황에 대해 작업자가 이해할 수 있는 명확한 한글 메시지 출력

## 5. 명령 지침 (Instruction for Agent)
- 코드를 생성할 때 항상 위 지침을 준수하십시오.
- UI 생성 시 '공장용 전문 소프트웨어' 느낌의 다크/그레이 톤 레이아웃을 우선 제안하십시오.
- 작업자 편의를 위해 복잡한 영어 용어보다는 직관적인 한글 UI 문구를 사용하십시오.

## 6. 백엔드로 부터 받을 데이터 구조
- **json**: predicted_label, confidence, effective_label, confirmed_state, allowed_transition, is_unknown