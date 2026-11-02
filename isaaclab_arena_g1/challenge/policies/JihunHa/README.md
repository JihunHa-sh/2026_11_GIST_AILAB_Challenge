# JihunHa

학습 없이 매 스텝 병·목표·로봇 위치를 읽어 명령을 계산하는 규칙 기반 고수준 정책. 저수준은 기본 제어기(`default`)를 씀.

## 실행 준비

이 폴더가 `isaaclab_arena_g1/challenge/policies/JihunHa/`에 있어야 함. 아래 명령은 모두 레포 최상위 폴더에서 실행.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## 시뮬레이션 화면으로 보기

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py --viz kit \
  --team isaaclab_arena_g1/challenge/policies/JihunHa/team.yaml --seeds 0
```

- Isaac Sim 창에 로봇이 움직이는 모습이 나옴. 화면은 실제보다 느리게(약 0.2배) 보임.
- 실제 속도로 보려면 `--viz kit` 대신 `--video`를 넣어 mp4로 저장.

## 터미널 평가 (점수 확인)

```bash
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py \
  --team isaaclab_arena_g1/challenge/policies/JihunHa/team.yaml --num_seeds 10
```

- 화면 없이 실행되어 더 빠름. 끝나면 시드별 결과표와 성공 수·평균 시간(`score`)이 출력됨.
- 특정 시드만 돌리려면 `--num_seeds 10` 대신 `--seeds 3,17,42` (쉼표로 구분, 공백 없음).
- 결과는 `challenge_results/JihunHa_<시각>/`에 저장됨.

추가 패키지나 모델 파일은 필요 없음.
