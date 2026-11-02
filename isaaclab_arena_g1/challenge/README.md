# G1 빠른 운반 챌린지 (`g1_fast_transport`)

Unitree G1 휴머노이드로 주방 조리대 위 바비큐 소스 병을 집어 방 반대편 책상 위 목표 자리에 옮겨 놓는 과제임.
정해진 시드에서 시뮬레이션 시간 기준으로 가장 빨리 옮긴 팀이 우승함.

참가자는 팀 폴더 하나에 정책 코드를 작성해 제출함. 정책은 고수준(작업 계획)과 저수준(관절 제어) 두 층으로
나뉘고, 저수준은 기본 제공 제어기를 그대로 써도 됨.

- 함수와 값의 자세한 목록: [API.md](API.md)
- 공유 안내 문서: <공유 문서 주소>

## 1. 과제

| 항목 | 내용 |
| --- | --- |
| 장면 | Lightwheel RoboCasa 주방 `one_wall_farmhouse1`. 오른쪽 조리대(높이 0.92 m)에서 방 반대편 책상(높이 0.75 m)까지 약 3.3 m |
| 물체 | 바비큐 소스 병 `bbq_sauce_bottle_hope_robolab` (3.9 × 5.8 × 20.8 cm) |
| 로봇 시작 | 조리대를 보고 선 자세. 방향은 고정, 위치는 시드마다 ±5 cm 무작위 |
| 목표 | 책상 가운데에 세워서 놓기. 방향은 로봇 기준으로 시작과 같음(세계 기준으로는 시작 방향 + 180°). 화면에 반투명 초록 병으로 표시되고 충돌은 없음 |
| 성공 | 병이 목표에서 위치 3 cm, 방향 15° 이내에 멈춘 상태(속도 5 cm/s 미만)로 0.5초 유지 |
| 실패 | 시뮬레이션 시간 90초 초과, 로봇 넘어짐(골반 높이 0.40 m 미만 또는 60° 이상 기울어짐), 병 떨어뜨림(높이 0.5 m 미만) |
| 제어 주기 | 50 Hz (0.02초마다 정책 호출) |

### 점수

- 점수 = 모든 평가 에피소드의 시뮬레이션 시간 평균. 실패한 에피소드는 90초로 계산. 낮을수록 좋음.
- 점수가 같으면 성공 횟수, 그다음 평균 위치 오차 순으로 순위 결정.
- 실제로 걸린 시간(벽시계 시간)과 정책 계산 시간도 기록하지만 점수에는 반영 안 함.
- 순위표는 하나임. 기본 저수준을 쓴 팀과 저수준을 직접 만든 팀을 같이 비교함.

## 2. 설치

필요 환경: Ubuntu 22.04 또는 24.04, NVIDIA RTX GPU(메모리 8 GB 이상, RTX 3090에서 확인), 드라이버 570 이상.

```bash
git clone --recurse-submodules git@github.com:gist-ailab/ailab-challenge-2nd.git
cd ailab-challenge-2nd
uv sync --extra dev
export OMNI_KIT_ACCEPT_EULA=YES
```

- 저장소가 private이라 GitHub 계정에 저장소 접근 권한이 있어야 함. SSH 키가 없으면 `https://github.com/gist-ailab/ailab-challenge-2nd.git`으로 클론.
- `uv`가 없으면 먼저 `curl -LsSf https://astral.sh/uv/install.sh | sh`로 설치.
- `uv sync`가 `.venv/`에 Isaac Sim 6.0, Isaac Lab 3.0을 설치함. 수십 GB를 받아서 처음에는 오래 걸림.
- `OMNI_KIT_ACCEPT_EULA=YES`는 Isaac Sim 라이선스 동의임. `~/.bashrc`에 넣어 두면 편함.
- 주방 장면과 물체 파일은 첫 실행 때 자동으로 받음.

## 3. 실행해 보기

설치가 끝나면 예시 팀으로 한 번 실행해서 장면과 목표 위치를 확인.

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py --viz kit \
  --team isaaclab_arena_g1/challenge/policies/example_team/team.yaml --seeds 0
```

- 예시 팀은 참고 정책(baseline)을 그대로 써서 병을 집어 책상까지 옮김.
- `--viz kit`을 빼면 화면 없이 실행되고 더 빠름.
- 시뮬레이션이 실제 시간보다 느리게(RTX 3090에서 약 0.2배) 돌아서 화면에서는 느리게 보임. 실제 빠르기는 끝에
  출력되는 표의 `sim[s]` 값으로 확인. 실제 속도로 보려면 `--video`로 영상을 저장해서 볼 것(5장).

## 4. 정책 만들기

### 구조

```
관측 ─▶ 고수준 ─▶ Command ─▶ 저수준 ─▶ 43개 관절 위치 목표 ─▶ 로봇
                                ▲
                       로봇 자신의 상태(RobotState)
```

| 층 | 하는 일 | 입력 | 출력 |
| --- | --- | --- | --- |
| 고수준 | 이동 경로, 파지, 팔 경로 계획, 단계 전환 등 작업 전체 | 관측과 `api` 함수로 읽은 값(물체·목표·로봇·가구 자세 등) | `Command` (걷기 속도, 골반 높이, 몸통 각도, 양 손목 목표, 양손 쥐기) |
| 저수준 | 명령을 관절 목표로 바꿔 걷고 팔을 움직임 | `Command`, 로봇 자신의 상태 | 43개 관절 위치 목표 [rad] |

- 저수준을 `default`로 두면 기본 제공 제어기(학습된 보행 정책 + 팔 역기구학 + 손 쥐기)를 씀. 고수준만
  작성할 거면 이걸 쓰면 됨.
- 저수준을 직접 만들면 로봇이 그 저수준이 내는 43개 관절 목표를 그대로 받음. 보행 정책을 따로 학습해서 넣는
  것도 허용함.
- 두 층의 출력 형식(값 순서, 단위, 관절 순서, 관절 PD 설정)은 [API.md](API.md) 8장에 정리되어 있음.

### 팀 폴더 만들기

예시 팀 폴더를 복사해서 시작.

```bash
cp -r isaaclab_arena_g1/challenge/policies/example_team isaaclab_arena_g1/challenge/policies/my_team
```

```
policies/my_team/
├── __init__.py
├── team.yaml        # 팀 이름, 사용할 고수준·저수준 클래스와 옵션
├── high_level.py    # 고수준 (MyHighLevel)
└── low_level.py     # 저수준 (MyLowLevel). 기본 저수준을 쓰면 수정할 필요 없음
```

복사한 뒤 `team.yaml`의 팀 이름과 클래스 경로를 바꿈.

```yaml
team_name: my_team
high_level:
  class: isaaclab_arena_g1.challenge.policies.my_team.high_level.MyHighLevel
  options: {}
low_level:
  class: default
  options: {}
```

- 직접 만든 저수준을 쓰려면 `low_level.class`를
  `isaaclab_arena_g1.challenge.policies.my_team.low_level.MyLowLevel`로 변경.
- `options`에 적은 값은 각 클래스 생성자에 키워드 인자로 전달됨.
- 예시의 `MyHighLevel`은 참고 정책을, `MyLowLevel`은 기본 제어기를 감싸고 있어서 복사한 상태로도 과제가 끝까지
  실행됨. 바꾸고 싶은 부분부터 고쳐 나가면 됨.

### 작성할 함수

```python
from isaaclab_arena_g1.challenge import api
from isaaclab_arena_g1.challenge.interfaces import Command, HighLevelPolicy, LowLevelPolicy


class MyHighLevel(HighLevelPolicy):
    def __init__(self, **options): ...                           # team.yaml의 options
    def setup(self, env): ...                                    # 첫 에피소드 전에 한 번
    def reset(self): ...                                         # 에피소드가 시작될 때마다
    def compute_command(self, env, observation) -> Command: ...  # 0.02초마다


class MyLowLevel(LowLevelPolicy):
    def setup(self, env): ...
    def reset(self): ...
    def compute_joint_targets(self, command, state) -> torch.Tensor: ...  # (1, 43) 관절 목표 [rad]
```

- 값은 `api` 함수로 읽음. 예: `api.object_pose(env)`, `api.goal_pose(env)`, `api.pelvis_pose(env)`,
  `api.wrist_pose_in_pelvis(env, "right")`. 전체 목록은 API.md 4장 참고.
- `Command`는 `Command.from_tensor(api.make_action(env, walk_forward=0.3, right_hand_closed=True))`처럼 만들 수
  있음. 지정하지 않은 값은 제자리에 서 있는 기본값이 됨.
- 에피소드마다 바뀌는 값(상태 기계의 단계, 타이머 등)은 `self`에 저장하고 `reset`에서 초기화.
- 고수준에 `phase` 속성(현재 단계 이름)을 두면 단계가 바뀔 때마다 시각, 물체와 목표 사이 오차, 골반 위치가
  출력되어 디버깅에 쓸 수 있음. 점수와는 관계없음.

## 5. 점수 확인

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py \
  --team isaaclab_arena_g1/challenge/policies/my_team/team.yaml --num_seeds 10
```

| 옵션 | 설명 |
| --- | --- |
| `--num_seeds N` | 시드 0부터 N개 실행 (`--first_seed S`로 시작 번호 변경) |
| `--seeds 3,17,42` | 지정한 시드만 실행. 쉼표로 구분하고 공백은 넣지 않음 |
| `--viz kit` | 화면에 표시 |
| `--video` | 에피소드마다 실제 속도 mp4 영상 저장. 실행 시간이 약 2배로 늘어남 |
| `--video_eye x y z --video_lookat x y z` | 영상 카메라 위치와 바라보는 점 (환경 좌표계, m) |

실행이 끝나면 화면에 결과표가 나오고, `challenge_results/<팀>_<시각>/`에 `results.json`, `results.csv`(그리고
영상)가 저장됨. 순위 점수는 `results.json`의 `summary.score_s`.

```
=== g1_fast_transport | team: baseline (high: ...BaselineHighLevel, low: default) ===
 seed          result   sim[s]  wall[s] pos err[cm] rot err[deg]
----------------------------------------------------------------
    0         success    36.04    275.9         1.9          4.1
    1         success    34.84    333.5         1.0          0.8
    2         success    34.86    341.7         0.5          2.0
    3         success    35.26    216.5         1.5          2.6
----------------------------------------------------------------
success 4/4 | score 35.25 s (lower is better; failure = 90 s) | best 34.84 s | mean success time 35.25 s
```

`result`가 `success`가 아니면 실패 원인이 표시됨: `time_out`(시간 초과), `robot_fell`(넘어짐),
`object_dropped`(병 떨어뜨림), `policy_gave_up`(정책이 스스로 포기).

## 6. 참고 정책 (baseline)

참고 정책은 시뮬레이터에서 병과 목표의 자세를 읽어 움직이는 스크립트 상태 기계임. 조리대로 걸어가 병을 옆에서
잡고, 물러나서 책상 앞까지 걸어간 뒤, 손목으로 병의 기울기와 방향을 맞춰 목표 위에 내려놓음.
시드 0~7에서 8번 중 5번 성공, 성공한 경우 평균 34.4초.

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py \
  --team isaaclab_arena_g1/challenge/policies/baseline/team.yaml --num_seeds 10
```

| 팀 파일 | 고수준 | 저수준 |
| --- | --- | --- |
| `policies/baseline/team.yaml` | 참고 정책 | 기본 제어기 |
| `policies/baseline_custom_low_level/team.yaml` | 참고 정책 | 기본 제어기를 직접 만든 저수준 형식(43개 관절 출력)으로 감싼 것. 저수준을 만들 때 참고용 |
| `policies/example_team/team.yaml` | 참고 정책을 감싼 `MyHighLevel` | 기본 제어기 |

코드 위치: 고수준 `high_level/baseline.py`(내부 동작은 `baseline_policy.py`), 저수준 `low_level/wbc.py`.

## 7. 규칙

1. 시뮬레이터 값은 읽기만 가능. 병이나 로봇을 직접 옮기거나 환경 설정·시간을 바꾸면 실격.
2. 정책은 매 스텝 관측을 보고 명령을 계산해야 함. 미리 저장한 명령이나 궤적을 그대로 재생하는 방식은 불가.
   학습한 신경망, 상태 기계, 경로 계획기 등은 모두 사용 가능.
3. 병만 목표 자세에 도착하면 성공. 손을 놓지 않아도 되고 로봇 위치는 상관없음.
4. 저수준은 `Command`와 로봇 자신의 상태(`RobotState`)만 사용. 출력은 43개 관절 위치 목표이고, 관절 PD 게인과
   토크 한계는 모든 팀이 같음.
5. 환경 옵션(물체, 위치, 책상 높이, 허용 오차, 시간 제한 등)은 기본값으로 평가함. 바꾸는 건 연습할 때만.
6. 평가 시드는 대회 당일 공개.

## 8. 제출

- 제출물은 팀 폴더(`policies/<팀 이름>/`) 전체. 학습한 모델 파일도 이 폴더에 넣고, 경로는 `team.yaml`의
  `options`로 넘길 것.
- 기본 설치에 없는 파이썬 패키지를 쓰면 팀 폴더에 `requirements.txt`를 넣고 제출 전에 주최 측에 알릴 것.
- 제출 전에 `--seeds 0`으로 한 번 실행해서 오류 없이 끝나는지 확인할 것.
- 제출 기한과 방법: <주최 측에서 기입>

## 9. 자주 묻는 질문

**실행하는 데 얼마나 걸림?**
RTX 3090 기준 Isaac Sim 시작에 1~2분, 35초짜리 에피소드 하나에 4~6분 정도. `--video`를 쓰면 약 2배.

**여러 환경을 동시에 돌릴 수 있음?**
한 프로세스에서는 환경 1개만 지원함. GPU 메모리가 24 GB면 프로세스 두 개를 동시에 실행할 수 있음.

**보행 정책을 직접 학습하고 싶음.**
이 환경은 한 번에 1개만 돌아가서 강화학습에는 맞지 않음. 병렬 환경을 쓰는 Isaac Lab 등에서 학습한 뒤
`LowLevelPolicy`로 감싸서 넣을 것. 이때 로봇 모델, 관절 PD 설정, 50 Hz 제어 주기를 이 환경과 같게 맞춰야 학습한
동작이 그대로 재현됨. 값은 API.md 8장 참고.

**목표 위치의 초록색 병이 안 보임.**
`--viz kit` 화면과 `--video` 영상에만 표시됨. 물리적으로는 없는 물체라서 로봇이나 병과 부딪히지 않음.

**`--seeds 0 1 2`처럼 입력했더니 오류가 남.**
시드는 `--seeds 0,1,2`처럼 쉼표로 구분해서 입력.
