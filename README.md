# 보관용 브랜치 — 머지하지 마세요

A2 가 Colab 에서 한 **손그림 15-joint 포즈 모델 파인튜닝 1차** 기록입니다.
노트북을 잃지 않으려고 깃에 올려 둔 것이고, **서비스 코드가 아닙니다.**

| | |
|---|---|
| 이 브랜치에 있는 것 | 노트북 하나와 이 파일 |
| **`dev` 에 머지** | **하지 않습니다.** 서비스 코드는 `dev` 에 있습니다 |
| 왜 머지하면 안 되나 | 여기에는 `runtime/` `torchserve/` 가 없습니다. 머지하면 지워집니다 |

이 브랜치는 **`dev` 와 히스토리를 공유하지 않습니다**(부모 없는 커밋).
그래서 실수로 머지하려 해도 깃이 `refusing to merge unrelated histories` 로 막습니다.

## 모델을 실제로 쓰려면

평가와 배포 조건은 설계 문서에 있습니다 —
`drawtale-pt_docs` [12 AI 아키텍처](https://github.com/seominji58/drawtale-pt_docs/blob/main/12-ai-architecture.md) 7장.

요약하면 **같은 test split 에서 관절별로 기존 모델과 비교**해야 하고,
모델 서버가 MMPose 0.29 라서 **ONNX 로 내보내야** 올릴 수 있습니다.
