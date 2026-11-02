# JihunHa

학습 없이 매 스텝 병·목표·로봇 위치를 읽어 명령을 계산하는 규칙 기반 고수준 정책. 저수준은 기본 제어기(`default`)를 씀.

## 실행

이 폴더가 `isaaclab_arena_g1/challenge/policies/JihunHa/`에 있어야 함. 레포 최상위 폴더에서 실행.

```bash
export OMNI_KIT_ACCEPT_EULA=YES
.venv/bin/python isaaclab_arena_g1/challenge/challenge_main.py \
  --team isaaclab_arena_g1/challenge/policies/JihunHa/team.yaml --seeds 0
```

- 시드 지정: `--seeds 3,17,42` (쉼표로 구분, 공백 없음)
- 화면에 표시: `--viz kit` 추가
- 영상 저장: `--video` 추가
- 결과는 `challenge_results/JihunHa_<시각>/`에 저장됨.

추가 패키지나 모델 파일은 필요 없음.
