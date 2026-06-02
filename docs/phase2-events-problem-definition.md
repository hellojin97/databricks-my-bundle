# Phase 2 — `events`(클릭스트림) 문제 정의

> 상태: 합의 완료 (구현 전)
> 범위: 이번 단계는 `events`만. `payments` / `shipments` 는 이후 별도 진행.

## 1. 배경 / 목적

Phase 1 은 차원(categories, users, products)과 핵심 팩트(orders, order_items)를 생성한다.
현재 데이터로는 **"무엇이 팔렸나"(orders)** 는 알 수 있지만,
**"왜 / 어떻게 구매에 이르렀나"(행동·퍼널·전환)** 는 분석할 수 없다.

`events` 테이블은 사용자 행동 로그를 추가해 다음 분석을 가능하게 한다.

- 전환율(conversion rate) 분석
- 퍼널 이탈(funnel drop-off) 분석
- 세션 / 검색 행동 분석

## 2. 데이터 모델

`events` — 팩트 테이블, `dt` 로 파티셔닝.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `event_id` | int64 | PK |
| `user_id` | int64 | FK → `users.user_id` (로그인 사용자 기준) |
| `session_id` | str | 세션 묶음 (예: `{user_id}-{n}`) |
| `event_type` | str | 표준 퍼널 세트 (아래) |
| `product_id` | int64 (nullable) | product_view / add_to_cart / purchase 에 설정, 그 외 NULL |
| `search_query` | str (nullable) | search 이벤트에만 설정 |
| `order_id` | int64 (nullable) | **purchase 이벤트에만** — 실제 주문 참조 |
| `event_ts` | datetime64[us] | 발생 시각 |
| `dt` | date | 파티션 키 (`event_ts` 의 날짜) |

### event_type (표준 퍼널 세트)

`page_view`, `product_view`, `search`, `add_to_cart`, `begin_checkout`, `purchase`

## 3. 퍼널 · 일관성 모델 (엄격한 일관성)

### 구매 세션 (purchase session)

- 모든 주문(order)마다 퍼널 1개 생성:
  `page_view → product_view → add_to_cart → begin_checkout → purchase`
- `purchase` 이벤트는 **실제 `order_id` 를 참조**한다.
- `product_view` / `add_to_cart` 는 그 주문의 **order_items 상품**을 포함한다
  (+ 일부 "안 산 상품" 탐색을 섞어 현실감 부여 가능).
- `purchase.event_ts == order.created_at`, 나머지 퍼널 이벤트는 그 **직전** 시각.

### 브라우징 세션 (browsing session)

- 구매로 이어지지 않는 세션도 생성한다 (view / search / cart 후 이탈).
- 이를 통해 **전환율 = 구매세션 / 전체세션** 분석이 가능해진다.

## 4. 규모 정책 (유저 기반 + 설정값)

- 구매 세션 수 = 주문 수 (자동으로 결정됨).
- 브라우징 세션 수 = `config: events.browse_sessions_per_user` 평균(Poisson) × 유저 수
  (세그먼트 가중 적용 가능 — vip 가 더 활발).
- 세션당 이벤트 수 = 퍼널 길이(가변), `events.max_events_per_session` 로 상한.
- **단일노드(F4s, 8GB) 안전선**: 기본값을 보수적으로 잡고(100k 유저 기준 수백만 건 수준),
  `config` / CLI 로 상향 조정 가능하게 한다.

## 5. 반드시 지킬 제약

- **시간적 FK**:
  - `event_ts ≥ user.created_at`
  - 참조 상품은 `created_at ≤ event_ts` 이고 그 시점에 미단종
  - → 기존 `generate_order_items._weighted_prefix_choice` 의 시간 제약 샘플링 재사용
- **재현성**: `base.make_rng(seed)` 사용, 같은 seed → 같은 결과.
- **파티셔닝 / 통합**:
  - `dt` 파티션, `base.write_parquet(..., partition_by=["dt"])` 재사용
  - `main.py` 에 추가 (order_items 이후; users / products / orders / order_items 에 의존)
- **dirty-data**: `search_query` 일부 NULL 등 `base.inject_nulls` 재사용 가능.

## 6. 재사용할 기존 자산

| 자산 | 위치 | 용도 |
|------|------|------|
| `make_rng` | `base.py` | 시드 기반 재현성 |
| `write_parquet` | `base.py` | dt 파티션 쓰기 |
| `inject_nulls` | `base.py` | search_query 등 NULL 주입 |
| `weighted_choice` | `base.py` | event_type / 세그먼트 가중 선택 |
| `_weighted_prefix_choice` | `generate_order_items.py` | 시간 제약 상품 선택 (벡터화) |
| 시간대 가중치 패턴 | `generate_orders.py` (`HOUR_WEIGHTS`) | 이벤트 시각의 시간대 분포 |

## 7. 검증 계획 (`tests/test_events.py` + 스모크)

- 스키마 / 컬럼 타입
- FK 유효성: user / product / order
- 시간적 정합성: `event_ts ≥ user.created_at`, 상품 생성·단종 제약
- purchase ↔ order 매칭 (purchase 이벤트의 order_id 가 실재)
- **세션 내 퍼널 순서**: purchase 가 세션의 마지막, begin_checkout 이 그 직전 등
- 전환율(구매세션/전체세션)이 타당한 범위
- 재현성 (같은 seed → 동일 출력)
- event_type 분포 타당성

## 8. 이번 범위 밖 (다음 단계)

- `payments` (orders FK: 결제 승인/실패/환불, 수단별)
- `shipments` (orders FK: 발송/도착, 캐리어, 리드타임)
