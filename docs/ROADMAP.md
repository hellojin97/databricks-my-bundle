# 로드맵 / 다음 작업 정리

이 프로젝트의 완료된 작업과 다음 세션에서 이어갈 수 있는 태스크를 정리한다.
(프로젝트 분석 → quick-win → 테스트/CI → 현실성 개선 순으로 진행해 왔다.)

## ✅ 완료됨

| 항목 | 내용 |
|------|------|
| README | 비개발자도 이해할 수 있는 README (Mermaid 다이어그램) |
| `--seed` CLI | 실행마다 seed 오버라이드, config 폴백 |
| config 정리 | 미사용 `volumes.orders/events` 제거 |
| 테스트 + CI | pytest 47개(재현성·스키마·FK·dirty-data) + ruff/pytest PR CI |
| 현실성 개선 | 통화별 현지금액(`fx_rate`/`amount_local`) + 시간적 FK 정합성 |
| 버그 수정 | `datetime64[ns]` 오버플로우(미단종 센티넬) 수정 |
| 문서 | Phase 2 events 문제 정의(`phase2-events-problem-definition.md`) |

## 🔜 다음 작업 (우선순위 순)

### 1. Phase 2 — `events`(클릭스트림) 구현 ⭐ 다음 차례
- 문제 정의 완료 → **바로 구현 착수 가능**.
- 상세: [`phase2-events-problem-definition.md`](./phase2-events-problem-definition.md)
- 작업 단위: `generate_events.py` 신규 + `config.yml`/CLI 인자 + `main.py` 통합 + `tests/test_events.py`
- 재사용: `_weighted_prefix_choice`(시간 제약 상품 선택), `make_rng`, `write_parquet`, `inject_nulls`, `HOUR_WEIGHTS`

### 2. Phase 2 — `payments` / `shipments`
- `payments` (orders FK): 결제 승인/실패/환불, 수단별. 엄격한 일관성(`order.status`와 정합).
- `shipments` (orders FK): 발송/도착, 캐리어, 배송 리드타임/지연.
- events 구현에서 만든 패턴(세션/퍼널, 상태 일관성)을 재활용.

### 3. 성능 / 확장성
- `generate_orders._assign_currencies`(orders.py): 300만 행 Python dict 루프 → polars `join` 벡터화.
- `generate_users`: 이메일/생성일/생년월일 Python 루프 → polars 표현식.
- 대규모(수천만 행) 시 단일노드 메모리 점검.

### 4. 운영 / 배포 기능
- **Delta 출력 옵션**: Parquet 대신 Delta로 쓰고 UC 테이블 등록.
- **증분(append) 모드**: 특정 날짜 구간만 생성 → 일일 적재 시뮬레이션.
- **Job 스케줄 트리거**: 매일 자동 생성.
- **데이터 품질 리포트 태스크**: 생성 직후 분포 검증을 Job 태스크로.

### 5. 정리 / 네이밍
- GitHub 레포 이름 변경: `databricks-my-bundle` → **`ecommerce-data-generator`**
  (Settings → Repository name. README는 이미 갱신됨.)
- 선택: 번들명(`hj-krc-dabs`)·패키지명(`ecommerce-generator`)을 새 이름과 통일.

## ⏸️ 보류 (사용자 요청)
- **`prd` 타깃 추가**: 워크플로우(`deploy-bundle.yml`)는 `prd` 배포를 참조하지만
  `databricks.yml`/`variables.yml`에 `prd` 타깃이 없음. → "아직 필요 없음"으로 보류.
