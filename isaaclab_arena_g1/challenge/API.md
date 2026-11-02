# g1_fast_transport 정책 작성 안내 (API)

정책 파일 위치, 읽을 수 있는 값과 함수, 명령을 만드는 방법을 정리한 문서임.
과제 설명과 규칙은 [README.md](README.md) 참고. 정책은 고수준과 저수준 두 층으로 나뉨(1장).

## 1. 구조: 고수준과 저수준

메인 프로그램 `challenge_main.py`가 팀 파일을 읽어 팀의 고수준 정책과 저수준 정책을 불러오고, 0.02초(50 Hz)마다
아래 순서로 실행함.

```
관측 ─▶ [고수준 HighLevelPolicy] ─▶ Command(고정 형식) ─▶ [저수준 LowLevelPolicy] ─▶ 43개 관절 위치 목표(고정) ─▶ 로봇
                                                               ▲
                                                     RobotState(로봇 자신의 상태)
```

| 층 | 입력 | 출력 | 정해진 것 |
| --- | --- | --- | --- |
| 고수준 | 관측 + `api`로 읽는 모든 값(읽기만) | `Command` | 출력 형식만. 내부에서 경로·파지·팔 경로 계획을 어떻게 나눌지는 자유 |
| 저수준 | `Command` + `RobotState` | 43개 관절 위치 목표 [rad], (1, 43) | 입출력 형식. 관절 PD 게인·토크 한계는 환경이 고정(모두 같은 모터) |

- 저수준을 `default`로 두면 환경의 내장 제어기(학습된 보행 정책 + 팔 Pink IK + 손 쥐기 매핑)가 Command를 실행함.
- 저수준을 직접 만들면 로봇이 그 저수준의 43개 관절 목표를 받음(`g1_joint_position` embodiment). 더 빠르게 걷는
  보행 정책을 학습해 넣는 것도 허용하며, 순위표는 하나임. 결과 파일에 사용한 고수준·저수준이 기록됨.
- 규칙상 저수준은 Command와 `RobotState`(로봇 자신의 상태)만 사용. 물체·목표 같은 바깥 정보는 고수준이 다룸.

### Command (고수준 → 저수준, 고정)

`isaaclab_arena_g1.challenge.interfaces.Command`. 모든 필드의 맨 앞 차원은 환경 개수(1).

| 필드 | 모양 | 뜻 |
| --- | --- | --- |
| `walk_forward`, `walk_left`, `turn_rate` | (1,) | 걷기 속도 앞 [m/s], 왼쪽 [m/s], 회전 [rad/s], 골반 기준 |
| `pelvis_height` | (1,) | 골반 높이 [m], 서 있을 때 약 0.74 |
| `torso_rpy` | (1, 3) | 몸통 roll, pitch, yaw [rad] |
| `left_wrist_pos`, `right_wrist_pos` | (1, 3) | 손목(`*_wrist_yaw_link`) 목표 위치 [m], 골반 좌표 |
| `left_wrist_quat`, `right_wrist_quat` | (1, 4) | 손목 목표 방향 (x, y, z, w), 골반 좌표 |
| `left_grip`, `right_grip` | (1,) | 손 0 = 펴기, 1 = 쥐기 |

만드는 방법은 5장 참고. 23개 값으로 펼친 순서는 8장 참고.

### RobotState (저수준 입력)

| 필드 | 모양 | 뜻 |
| --- | --- | --- |
| `joint_pos`, `joint_vel` | (1, 43) | 관절 위치 [rad], 속도 [rad/s] (`joint_names` 순서) |
| `joint_names` | list | 43개 관절 이름. 저수준 출력 순서와 같음 |
| `default_joint_pos` | (1, 43) | 기본 서 있는 자세의 관절 위치 |
| `pelvis_quat` | (1, 4) | 골반 방향 (world 기준, x, y, z, w) |
| `pelvis_lin_vel`, `pelvis_ang_vel` | (1, 3) | 골반 선속도·각속도, 골반 좌표 |
| `projected_gravity` | (1, 3) | 골반 좌표에서 본 중력 방향. 똑바로 서 있으면 (0, 0, -1) |
| `dt` | float | 0.02 s |

## 2. 팀 폴더와 실행

팀마다 `isaaclab_arena_g1/challenge/policies/<팀>/` 폴더 하나를 사용. 예시 팀을 복사해서 시작.

```bash
cp -r isaaclab_arena_g1/challenge/policies/example_team isaaclab_arena_g1/challenge/policies/my_team
```

```
policies/my_team/
├── __init__.py
├── team.yaml        # 팀 이름, 고수준·저수준 클래스와 옵션
├── high_level.py    # class MyHighLevel(HighLevelPolicy)
└── low_level.py     # class MyLowLevel(LowLevelPolicy). 내장 제어기를 쓰면 수정할 필요 없음
```

`team.yaml`:

```yaml
team_name: my_team
high_level:
  class: isaaclab_arena_g1.challenge.policies.my_team.high_level.MyHighLevel
  options: {}           # 생성자에 키워드 인자로 전달됨
low_level:
  class: default        # 내장 제어기. 직접 만든 저수준이면 isaaclab_arena_g1.challenge.policies.my_team.low_level.MyLowLevel
  options: {}
```

실행:

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py \
  --team isaaclab_arena_g1/challenge/policies/my_team/team.yaml --num_seeds 3
```

- 화면 표시는 `--viz kit`, 영상 저장은 `--video`, 시드는 `--num_seeds N` 또는 `--seeds 0,3,5`.
- 각 클래스에서 구현할 함수:

```python
class MyHighLevel(HighLevelPolicy):
    def __init__(self, **options): ...                           # team.yaml의 options
    def setup(self, env): ...                                    # 첫 에피소드 전에 한 번(모델 불러오기 등)
    def reset(self): ...                                         # 에피소드 시작마다
    def compute_command(self, env, observation) -> Command: ...  # 0.02초마다

class MyLowLevel(LowLevelPolicy):
    def setup(self, env): ...
    def reset(self): ...
    def compute_joint_targets(self, command, state) -> torch.Tensor: ...  # (1, 43) 관절 목표 [rad]
```

- 고수준에 `phase` 속성(현재 단계 이름)을 두면 단계가 바뀔 때마다 시각, 물체-목표 오차, 골반 위치가 출력됨.

### 제공하는 예시

| 이름 | 종류 | 내용 |
| --- | --- | --- |
| `high_level.baseline.BaselineHighLevel` | 고수준 | 참고 정책. 상태 기계(접근 → 잡기 → 들기 → 물러나기 → 운반 → 놓기) + 이동 계획(경유점 P 제어: 골반 자세·목표점 → 걷기 속도) + 파지 계획(물체 바운딩 박스 → 오른 손목 파지 자세) + 팔 경로(현재·목표 손목 자세 → 최소 저크 직선 궤적, 놓기 전 손목으로 기울기·방향 보정) |
| `high_level.template.TemplateHighLevel` | 고수준 | 물체 앞까지 걸어가서 서기만 함 |
| `default` | 저수준 | 환경 내장 제어기 |
| `low_level.wbc.WbcLowLevel` | 저수준 | 내장 제어기를 팀 저수준 형식(43개 관절 출력)으로 감싼 것. 저수준을 바꿀 때 출발점(예: 다리 관절만 직접 학습한 보행 정책으로 교체) |
| `low_level.stand.HoldPoseLowLevel` | 저수준 | 첫 자세를 유지하기만 하는 가장 작은 예시 |

팀 파일 예시:

- `policies/baseline/team.yaml`: 참고 정책 + 내장 제어기
- `policies/baseline_custom_low_level/team.yaml`: 참고 정책 + `WbcLowLevel`
- `policies/example_team/team.yaml`: `MyHighLevel` + 내장 제어기. 예시 팀의 `MyHighLevel`은 참고 정책을,
  `MyLowLevel`은 내장 제어기를 감싸고 있어서 복사한 상태로도 과제가 끝까지 실행됨. 바꾸고 싶은 부분만 고치면 됨.

### 보행을 직접 학습하려면

이 챌린지 환경은 한 번에 환경 1개만 돌아가서 느리고, 강화학습용으로는 맞지 않음. Isaac Lab처럼 환경 수천 개를
병렬로 돌리는 곳에서 학습한 뒤, 정책을 `LowLevelPolicy`로 감싸서(입력: Command의 걷기 명령 + RobotState,
출력: 관절 목표) 넣는 방식을 권장함. 로봇 모델, 관절 PD 설정, 50 Hz 제어 주기를 8장의 값과 같게 맞춰서 학습할 것.

## 3. 좌표계와 단위

| 이름 | 뜻 |
| --- | --- |
| env 좌표계 | 바닥이 z = 0, 조리대 앞 모서리가 y = -0.65, 로봇은 +y를 보고 시작. 이 과제에서는 world 좌표와 같음 |
| pelvis 좌표계 | 로봇 골반에 붙은 좌표계. x 앞, y 왼쪽, z 위. **Command의 손목 목표는 이 좌표계** |
| 자세(pose) | `(x, y, z, qx, qy, qz, qw)` = 위치 [m] + 쿼터니언 (x, y, z, w 순서) |
| 각도 / 시간 | 라디안 / 초 |
| 텐서 모양 | 맨 앞 차원이 환경 개수(항상 1). 예: 자세는 `(1, 7)` |

## 4. 얻을 수 있는 값

모두 `from isaaclab_arena_g1.challenge import api`로 불러서 사용.

### 물체와 목표

| 값 | 함수 | 모양 | 좌표계·단위 |
| --- | --- | --- | --- |
| 물체 자세 | `api.object_pose(env)` | (1, 7) | env |
| 물체 속도 (선속도 + 각속도) | `api.object_velocity(env)` | (1, 6) | world, m/s, rad/s |
| 물체 바운딩 박스 꼭짓점 8개 | `api.object_corners(env)` | (1, 8, 3) | env |
| 목표 자세 | `api.goal_pose(env)` | (1, 7) | env |
| 목표까지 거리와 회전각 | `api.goal_error(env)` | (1,), (1,) | m, rad |
| 물체 이름 | `api.object_name(env)` | str | `bbq_sauce_bottle_hope_robolab` |

### 로봇

| 값 | 함수 | 모양 | 좌표계·단위 |
| --- | --- | --- | --- |
| 골반 자세 | `api.pelvis_pose(env)` | (1, 7) | env |
| 골반 속도 | `api.pelvis_velocity(env)` | (1, 6) | world |
| 임의 링크의 자세 | `api.body_pose(env, "right_hand_palm_link")` | (1, 7) | env |
| 링크 이름 53개 | `api.body_names(env)` | list | |
| 손목 현재 자세 (`"left"`/`"right"`) | `api.wrist_pose_in_pelvis(env, "right")` | (1, 7) | pelvis (Command와 같은 좌표계) |
| 관절 위치 43개 | `api.joint_positions(env)` | (1, 43) | rad |
| 관절 속도 43개 | `api.joint_velocities(env)` | (1, 43) | rad/s |
| 관절 이름 43개 | `api.joint_names(env)` | list | 다리, 허리, 팔, Dex3 손가락 |

자주 쓰는 링크: `pelvis`, `torso_link`, `left/right_wrist_yaw_link`(손목, Command 손목 목표의 기준점),
`left/right_hand_palm_link`(손바닥), `right_hand_thumb_2_link`, `right_hand_index_1_link`(손가락 끝).

### 가구와 시간

| 값 | 함수 | 모양 | 비고 |
| --- | --- | --- | --- |
| 책상 바운딩 박스 꼭짓점 | `api.desk_corners(env)` | (1, 8, 3) | 바닥~상판(0.75 m), env |
| 조리대 상판 꼭짓점 | `api.counter_top_corners(env)` | (1, 8, 3) | 상판 높이 약 0.92 m, env |
| 에피소드 경과 시간 | `api.elapsed_time(env)` | (1,) | 시뮬레이션 초. 점수에 쓰이는 시간 |
| 제어 주기 | `api.control_dt(env)` | float | 0.02 s |
| 시간 제한 | `api.time_limit(env)` | float | 90 s |

### 좌표 변환과 각도 도우미

| 함수 | 하는 일 |
| --- | --- |
| `api.env_to_pelvis_points(env, points)` | env 좌표의 점 (1, ..., 3)을 pelvis 좌표로 변환. 물체 위치를 손목 목표로 바꿀 때 사용 |
| `api.env_to_pelvis_pose(env, pose)` | env 좌표의 자세 (1, 7)를 pelvis 좌표로 변환 |
| `api.yaw_of(quat)` | 쿼터니언의 방향각(z축 회전) [rad] |
| `api.wrap_angle(angle)` | 각도를 [-π, π) 범위로 |

### 관측 딕셔너리 `observation`

`api` 함수 대신 관측에서 바로 읽어도 됨.

| 그룹 | 내용 |
| --- | --- |
| `observation["challenge"]` (1, 22) | 물체 자세 [0:7], 목표 자세 [7:14], 골반 자세 [14:21], 경과 시간 [21]. `api.read_challenge_observation(observation)`으로 이름 붙여 나눌 수 있음 |
| `observation["policy"]` | `robot_joint_pos` (1, 43), `left/right_wrist_pose_pelvis_frame` (1, 4, 4) 동차 변환 행렬, 직전 액션 `actions` (1, 23) 등 |
| `observation["wbc"]` | `robot_joint_vel` (1, 43), `is_navigating` 등 다리 제어기용 값 |

## 5. 명령(Command) 만들기

고수준은 `Command`를 반환해야 함. `api.make_action`으로 23개 값을 만들고 `Command.from_tensor`로 바꾸는 방법이
가장 간단함. 기본값은 "제자리에 서서 팔 내리고 손 편 상태"이고, 바꿀 값만 넘기면 됨.

```python
command = Command.from_tensor(api.make_action(
    env,
    right_wrist_pos=(0.30, -0.17, 0.10),             # 오른 손목 목표 [m], pelvis 좌표
    right_wrist_quat=(0.0, 0.0, 0.0, 1.0),           # 오른 손목 방향 (x, y, z, w), pelvis 좌표
    right_hand_closed=True,                          # 오른손 쥐기
    walk_forward=0.3, walk_left=0.0, turn_rate=0.0,  # 걷기 [m/s], 회전 [rad/s], pelvis 기준
    pelvis_height=0.74,                              # 골반 높이 [m], 서 있을 때 약 0.74
))
```

`command.to_tensor()`는 반대로 23개 값의 텐서로 바꿈(순서는 8장).

참고 사항:

- 손목 방향이 `(0, 0, 0, 1)`이면 손가락이 앞을 향하고 손바닥이 몸 안쪽을 향함.
- 내장 저수준에서는 팔이 IK로 손목 목표를 따라가고, 다리는 학습된 보행 정책이 걷기 명령을 따름. 목표를 한 번에
  멀리 바꾸면 팔이 급하게 움직이므로 조금씩 옮기는 편이 좋음.
- 물건을 들면 팔이 4~6 cm 정도 처지고 손목이 10° 정도 아래로 숙여짐. 위치 오차는 `api.goal_error`를 보면서
  되먹임으로 줄일 것.
- 걷는 중에 손이 조리대·책상 높이에 있으면 가구에 걸림. 걸을 때는 손을 들거나 몸 쪽으로 붙일 것.

## 6. 예시: 오른손을 물체 앞으로 가져가기

```python
from isaaclab_arena_g1.challenge import api
from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy

class MyHighLevel(HighLevelPolicy):
    def compute_command(self, env, observation) -> Command:
        object_pos = api.object_pose(env)[:, :3]                 # env 좌표
        target = api.env_to_pelvis_points(env, object_pos)[0]    # pelvis 좌표로
        target[0] -= 0.10                                        # 물체 10 cm 앞
        target[1] -= 0.05                                        # 조금 오른쪽
        return Command.from_tensor(api.make_action(env, right_wrist_pos=target))
```

참고 구현: [high_level/baseline.py](high_level/baseline.py) (내부 상태 기계는 [baseline_policy.py](baseline_policy.py)).

## 7. 규칙 요약

- 시뮬레이터 값은 **읽기만** 가능. 물체·로봇을 직접 옮기거나 설정·시간을 바꾸면 실격.
- 매 스텝 관측으로 명령을 계산해야 함. 저장해 둔 명령이나 궤적을 그대로 재생하는 방식은 불가.
- 저수준은 Command와 RobotState(로봇 자신의 상태)만 사용. 출력은 43개 관절 위치 목표이고, 관절 PD 게인·토크 한계는
  모든 팀이 같음.
- 물체만 목표 자세(위치 3 cm, 방향 15° 이내, 0.5초 정지)에 도착하면 성공.
- 환경 옵션은 기본값으로 평가함.

## 8. 출력 형식 요약

### 고수준 출력: `Command`

`command.to_tensor()`로 펼친 23개 값의 순서. 모든 값은 `torch.float32`이고 맨 앞 차원은 1.

| 인덱스 | 필드 | 단위·뜻 | 참고 값 |
| --- | --- | --- | --- |
| 0 | `left_grip` | 왼손 0 = 펴기, 1 = 쥐기 | |
| 1 | `right_grip` | 오른손 0 = 펴기, 1 = 쥐기 | |
| 2:5 | `left_wrist_pos` | 왼 손목 목표 위치 [m], 골반 좌표 | 팔 내린 자세 (0.03, 0.20, -0.07) |
| 5:9 | `left_wrist_quat` | 왼 손목 목표 방향 (x, y, z, w) | (0, 0, 0, 1) = 손가락 앞, 손바닥 몸 안쪽 |
| 9:12 | `right_wrist_pos` | 오른 손목 목표 위치 [m] | 팔 내린 자세 (0.03, -0.20, -0.07) |
| 12:16 | `right_wrist_quat` | 오른 손목 목표 방향 | |
| 16 | `walk_forward` | 앞으로 걷는 속도 [m/s] | 참고 정책 최대 0.4 |
| 17 | `walk_left` | 왼쪽으로 걷는 속도 [m/s] | 참고 정책 최대 0.3 |
| 18 | `turn_rate` | 회전 속도, 반시계 방향 + [rad/s] | 참고 정책 최대 0.5 |
| 19 | `pelvis_height` | 골반 높이 [m] | 서 있을 때 0.74 |
| 20:23 | `torso_rpy` | 몸통 roll, pitch, yaw [rad] | 보통 0 |

### 저수준 출력: 관절 위치 목표

| 항목 | 형식 |
| --- | --- |
| 모양·타입 | (1, 43), `torch.float32`, `state.joint_pos`와 같은 장치 |
| 뜻 | 각 관절의 절대 목표 각도 [rad]. 현재 각도와의 차이나 기본 자세 기준 값이 아님 |
| 순서 | `state.joint_names` 순서 (아래 목록) |
| 주기 | 50 Hz. 한 번 낸 목표를 물리 시뮬레이션 4스텝(200 Hz) 동안 유지 |
| 추종 | 환경의 관절 PD 제어기. 토크 = 강성 × (목표 − 현재) − 감쇠 × 각속도, 토크 한계에서 잘림 |
| 허용 안 함 | 토크 직접 제어, 다른 제어 주기, PD 게인 변경, 일부 관절만 출력 |

| 관절 | 강성 [N·m/rad] | 감쇠 [N·m·s/rad] | 토크 한계 [N·m] | 속도 한계 [rad/s] |
| --- | --- | --- | --- | --- |
| hip yaw / roll / pitch | 150 | 2 | 88 | 32 |
| knee | 300 | 4 | 139 | 20 |
| ankle pitch / roll | 40 | 2 | 50 | 37 |
| waist yaw | 250 | 5 | 88 | 32 |
| waist roll / pitch | 250 | 5 | 50 | 37 |
| shoulder pitch / roll | 100 | 5 | 25 | 37 |
| shoulder yaw, elbow | 40 | 2 | 25 | 37 |
| wrist roll | 20 | 2 | 25 | 37 |
| wrist pitch / yaw | 20 | 2 | 5 | 22 |
| 손가락(모두) | 4 | 0.5 | 5 | 10 |

모든 관절의 armature는 0.03.

```
 0 left_hip_pitch_joint        15 left_shoulder_roll_joint     30 left_hand_middle_0_joint
 1 right_hip_pitch_joint       16 right_shoulder_roll_joint    31 left_hand_thumb_0_joint
 2 waist_yaw_joint             17 left_ankle_roll_joint        32 right_hand_index_0_joint
 3 left_hip_roll_joint         18 right_ankle_roll_joint       33 right_hand_middle_0_joint
 4 right_hip_roll_joint        19 left_shoulder_yaw_joint      34 right_hand_thumb_0_joint
 5 waist_roll_joint            20 right_shoulder_yaw_joint     35 left_hand_index_1_joint
 6 left_hip_yaw_joint          21 left_elbow_joint             36 left_hand_middle_1_joint
 7 right_hip_yaw_joint         22 right_elbow_joint            37 left_hand_thumb_1_joint
 8 waist_pitch_joint           23 left_wrist_roll_joint        38 right_hand_index_1_joint
 9 left_knee_joint             24 right_wrist_roll_joint       39 right_hand_middle_1_joint
10 right_knee_joint            25 left_wrist_pitch_joint       40 right_hand_thumb_1_joint
11 left_shoulder_pitch_joint   26 right_wrist_pitch_joint      41 left_hand_thumb_2_joint
12 right_shoulder_pitch_joint  27 left_wrist_yaw_joint         42 right_hand_thumb_2_joint
13 left_ankle_pitch_joint      28 right_wrist_yaw_joint
14 right_ankle_pitch_joint     29 left_hand_index_0_joint
```

코드에서는 순서를 외우지 말고 `state.joint_names.index("left_knee_joint")`처럼 이름으로 찾을 것.
