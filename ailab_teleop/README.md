# ailab_teleop: 카메라 원격조작으로 g1_fast_transport 시연 데이터 모으기

RealSense D555 한 대로 사람 동작을 읽어 챌린지 주방 환경(`g1_fast_transport`)의 G1 을 조작하고, 성공한 에피소드를
VLA 학습용 데이터로 저장합니다.

```
D555 → Fast SAM 3D Body (run_fastsam_publisher.py, conda: fast_sam_3d_body)
     → ZMQ :5556 (사람 자세, 골반 위치, 골반 기준 손목 자세, 그랩)
     → g1_camera 장치 (camera_teleop_device.py) → 23차원 Command (걷기, 양손목 목표, 그립)
     → 챌린지 환경의 기본 저수준 제어기 (보행 정책 + 팔 IK + 손 쥐기) → G1
     → Isaac Lab record_demos.py → HDF5 (성공한 에피소드만) → LeRobot (GR00T 학습 형식)
```

기록되는 행동은 챌린지의 `Command` 와 같은 23차원(API.md 8장)이라, 학습한 정책을 그대로 고수준 정책으로 쓸 수 있습니다.

## 준비

1. 챌린지 저장소 설치 (상위 README / `isaaclab_arena_g1/challenge/README.md` 2장): `uv sync --extra dev` → `.venv/`
2. Fast SAM 3D Body 와 conda 환경 `fast_sam_3d_body`:
   https://github.com/JihunHa-sh/2026_10_GIST_AILAB_Challenge README 의 설치 ④~⑥ (`~/third_party` 에 설치)
3. SMPL 모델을 `ailab_teleop/assets/smpl/SMPL_NEUTRAL.pkl` 에 둡니다 (라이선스 때문에 git 에 올리지 않음).

## 실행 (챌린지 저장소 루트에서, 터미널 2개)

| 터미널 | 환경 | 명령 |
|---|---|---|
| A | `conda activate fast_sam_3d_body` | `python ailab_teleop/run_fastsam_publisher.py` |
| B | (없음, `.venv` 사용) | `ailab_teleop/record_demos.sh 10` |

- 터미널 A 를 먼저 켜고 `Live: ... publish=50.0Hz` 가 나오는지 확인하세요 (카메라 앞에 혼자 서 있어야 송신됨).
- 터미널 B 의 인자: `record_demos.sh [모을 에피소드 수] [저장 파일]` (기본 `datasets/g1_fast_transport_camera.hdf5`).
- GPU 메모리가 넉넉해야 합니다 (Isaac Sim + Fast SAM). 다른 GPU 작업은 끄고 실행하세요.

## 조작 (키는 Isaac Sim 창에서)

1. 카메라를 정면으로 보고 **차렷 자세**(팔을 쭉 펴서 내림)로 서서 **`C`** → 보정 + 조작 시작 + 새 에피소드 시작.
   (세션 처음에 한 번이면 됨. 이후 에피소드에서도 보정이 유지되고, 다시 맞추고 싶을 때만 또 누름)
2. 움직이면 로봇이 따라옵니다.
   - **걷기**: 사람이 이동한 거리의 **2.5배**를 같은 방향(내 몸 기준 앞/옆)으로, 몸을 돌린 각도만큼 로봇이 따라갑니다.
     카메라 앞에서 1.3 m 걸으면 로봇은 과제 거리(3.3 m)를 갑니다.
   - **팔**: 손목 위치·방향을 따라갑니다 (차렷 자세 = 로봇 팔 내린 자세 기준).
   - **손**: 주먹 = 쥐기, 편 손 = 펴기.
3. **`P`** 일시정지/재개. 정지 중엔 로봇이 서 있고, 재개하면 그 자리가 새 기준이 됩니다.
   카메라 앞 공간이 좁으면 `P` → 화면 가운데로 돌아와서 → `P` 로 이어서 걸으세요 (마우스를 들었다 놓는 것처럼).
4. **성공하면 자동 저장**: 병이 목표 자리에 놓여 0.5초 멈추면 그 에피소드가 저장되고 바로 다음 에피소드가 시작됩니다.
   **`R`** 은 지금 시도를 **저장하지 않고** 버린 뒤 리셋합니다 (실패했거나 꼬였을 때).
   정한 개수(`record_demos.sh 10` 이면 10개)를 모으면 자동으로 끝납니다.

시뮬레이션이 실제 시간보다 느립니다 (RTX 4090 에서 화면 없이도 약 0.27배). 걷기는 "사람이 간 만큼 로봇이 가는"
위치 추종이라 거리는 맞지만 로봇이 늦게 따라오므로, **천천히 움직이고 로봇이 따라올 때까지 기다리면서** 조작하세요.
Isaac Sim 터미널에 2초마다 `[camera_teleop] ... cmd ... robot est ...` 가 출력됩니다.

## LeRobot (GR00T) 형식으로 변환

```bash
ailab_teleop/convert_to_lerobot.sh datasets/g1_fast_transport_camera.hdf5
# → datasets/g1_fast_transport_camera/lerobot/ (data/*.parquet, videos/*/ego_view mp4, meta/)
```
설정: `ailab_teleop/lerobot_g1_fast_transport.yaml` (언어 지시문, 영상 크기 등). Arena 의 GR00T 변환기를 그대로 씁니다.

HDF5 내용 (에피소드마다 `data/demo_N/`): `camera_obs/robot_head_cam_rgb` (T, 480, 640, 3), `actions` (T, 23 = Command),
`obs/` (43개 관절, 양손목 자세 등), `processed_actions` (43개 관절 목표), `states/` (로봇·병·책상 상태), `initial_state/`.

## 조정 값 (`camera_teleop_device.py` 의 `CameraTeleopCfg`)

| 값 | 기본 | 뜻 |
|---|---|---|
| `walk_scale` | 2.5 | 사람 이동 대비 로봇 이동 비율 (1.3 m 걸으면 로봇 3.3 m). 실행 때 `CAMERA_TELEOP_WALK_SCALE=2.0 ailab_teleop/record_demos.sh` 처럼 바꿀 수 있음 |
| `turn_scale` | 1.0 | 사람 회전 대비 로봇 회전 비율 |
| `kp_pos`, `kp_yaw` | 1.5 | 위치/방향 추종 게인 |
| `pos_deadband`, `yaw_deadband_deg` | 0.15 m, 8° | 이보다 작은 오차는 무시 (자세 추정 흔들림, 이동 배율만큼 커짐) |
| `max_forward`, `max_left`, `max_turn` | 0.4, 0.3, 0.5 | 속도 한계 (참고 정책 최대값) |
| `arm_scale` | 0.8 | 손목 이동 비율 (G1 팔이 짧음) |
| `body_box_min`, `body_box_max` | (-0.12, -0.17, -0.35) ~ (0.16, 0.17, 0.50) | 손목 목표가 들어갈 수 없는 몸통 영역 (골반 기준, m). 이 환경의 G1 은 자기 충돌이 꺼져 있어 팔이 몸을 뚫을 수 있으므로, 목표를 몸 앞/옆 바깥면으로 밀어냄 |

손 그랩 기준은 터미널 A 옵션 `--grab-close 0.5 --grab-open 0.3`.

## 파일

| 파일 | 내용 |
|---|---|
| `run_fastsam_publisher.py` | D555 + Fast SAM 3D Body 송신기 (골반 위치, 손목 자세, 그랩 포함) |
| `hand_retarget.py` | 손 키포인트 → 손가락 굽힘 → 그랩 판정 |
| `camera_teleop_device.py` | Isaac Lab 원격조작 장치 (`g1_camera`): 스트림 → 23차원 행동 |
| `arena_callback.py` | `record_demos.py` 용 콜백: 장치 등록, 카메라 영상 기록 설정, 성공 판정 연결 |
| `record_demos.sh` | 기록 실행 |
| `convert_to_lerobot.sh`, `lerobot_g1_fast_transport.yaml` | HDF5 → LeRobot 변환 |

시험용 환경 변수 (실제 수집에는 쓰지 않음): `CAMERA_TELEOP_AUTOCAL=N` (N 스텝 후 자동 보정),
`CAMERA_TELEOP_FAKE_SUCCESS_AFTER=N` (N 스텝 후 성공 처리, 저장 경로 시험용).
