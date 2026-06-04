"""가짜 데이터 생성 메인 엔트리포인트.

콘솔 스크립트(`generate-data`) 및 Databricks python_wheel_task의 진입점.

사용법:
    # 필수: 기간 + 출력 경로
    generate-data --start-date 2025-05-11 --end-date 2026-05-11 \
                  --output-volume /Volumes/<catalog>/<schema>/<volume>/raw

    # 선택: 볼륨/dirty_data 오버라이드 (미지정 시 config.yml 값)
    generate-data --start-date ... --end-date ... --output-volume ... \
                  --users 10000 --products 500 --null-rate-gender 0.1

seed 및 선택 인자의 기본값은 config.yml에서 읽는다 (--config로 교체 가능).
seed는 --seed로 실행마다 오버라이드할 수 있다 (다른 데이터셋 생성용).

Phase 1: 차원 테이블 (categories, users, products)
Phase 2: 팩트 테이블 (orders, order_items, events)
"""
import argparse
import time
from datetime import date
from pathlib import Path

from . import (
    generate_categories,
    generate_events,
    generate_order_items,
    generate_orders,
    generate_payments,
    generate_products,
    generate_shipments,
    generate_users,
)
from .base import load_config, write_parquet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E-Commerce 가짜 데이터 생성기")
    parser.add_argument(
        "--config",
        default=None,
        help="config.yml 경로 (기본: 패키지 내장). seed 및 미지정 옵션의 기본값 출처",
    )
    # 필수 인자 — 매 실행마다 명시적으로 지정
    parser.add_argument("--start-date", required=True, help="분석 기간 시작 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="분석 기간 종료 (YYYY-MM-DD)")
    parser.add_argument(
        "--output-volume", required=True, help="출력 경로 (UC Volume 등 디렉토리)"
    )
    # 선택 인자 — 미지정 시 config.yml 값 사용
    parser.add_argument(
        "--seed", type=int, default=None, help="RNG seed (기본: config seed). 같은 코드로 다른 데이터셋 생성 시 사용"
    )
    parser.add_argument(
        "--users", type=int, default=None, help="유저 수 (기본: config volumes.users)"
    )
    parser.add_argument(
        "--products", type=int, default=None, help="상품 수 (기본: config volumes.products)"
    )
    parser.add_argument(
        "--null-rate-gender", type=float, default=None,
        help="유저 gender NULL 비율 (기본: config dirty_data.null_rate_gender)",
    )
    parser.add_argument(
        "--null-rate-brand", type=float, default=None,
        help="상품 brand NULL 비율 (기본: config dirty_data.null_rate_brand)",
    )
    parser.add_argument(
        "--discontinued-rate", type=float, default=None,
        help="상품 단종 비율 (기본: config dirty_data.discontinued_rate)",
    )
    parser.add_argument(
        "--browse-sessions-per-user", type=float, default=None,
        help="유저당 평균 브라우징 세션 수 (기본: config events.browse_sessions_per_user)",
    )
    parser.add_argument(
        "--max-events-per-session", type=int, default=None,
        help="브라우징 세션당 이벤트 상한 (기본: config events.max_events_per_session)",
    )
    parser.add_argument(
        "--null-rate-search", type=float, default=None,
        help="search_query NULL 비율 (기본: config dirty_data.null_rate_search)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    seed = args.seed if args.seed is not None else cfg["seed"]
    output_dir = Path(args.output_volume)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 선택 인자 해석: CLI 값이 있으면 우선, 없으면 config.yml 기본값
    n_users = args.users if args.users is not None else cfg["volumes"]["users"]
    n_products = args.products if args.products is not None else cfg["volumes"]["products"]
    null_rate_gender = (
        args.null_rate_gender
        if args.null_rate_gender is not None
        else cfg["dirty_data"]["null_rate_gender"]
    )
    null_rate_brand = (
        args.null_rate_brand
        if args.null_rate_brand is not None
        else cfg["dirty_data"]["null_rate_brand"]
    )
    discontinued_rate = (
        args.discontinued_rate
        if args.discontinued_rate is not None
        else cfg["dirty_data"]["discontinued_rate"]
    )
    browse_sessions_per_user = (
        args.browse_sessions_per_user
        if args.browse_sessions_per_user is not None
        else cfg["events"]["browse_sessions_per_user"]
    )
    max_events_per_session = (
        args.max_events_per_session
        if args.max_events_per_session is not None
        else cfg["events"]["max_events_per_session"]
    )
    null_rate_search = (
        args.null_rate_search
        if args.null_rate_search is not None
        else cfg["dirty_data"]["null_rate_search"]
    )

    print("=" * 64)
    print("E-Commerce Fake Data Generation - Phase 1 (Dimensions)")
    print(f"  Period : {start_date} -> {end_date}")
    print(f"  Output : {output_dir.resolve()}")
    print(f"  Seed   : {seed}")
    print("=" * 64)

    total_t0 = time.time()

    # 1. Categories (작고 결정적, 다른 테이블의 FK 기반)
    print("\n[1/3] Categories")
    t0 = time.time()
    categories = generate_categories.generate(seed=seed)
    write_parquet(categories, output_dir / "categories.parquet")
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 2. Users
    print("\n[2/3] Users")
    t0 = time.time()
    users = generate_users.generate(
        n_users=n_users,
        start_date=start_date,
        end_date=end_date,
        null_rate_gender=null_rate_gender,
        seed=seed,
    )
    write_parquet(users, output_dir / "users.parquet")
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 3. Products (categories에 의존)
    print("\n[3/3] Products")
    t0 = time.time()
    products = generate_products.generate(
        n_products=n_products,
        categories_df=categories,
        start_date=start_date,
        end_date=end_date,
        null_rate_brand=null_rate_brand,
        discontinued_rate=discontinued_rate,
        seed=seed,
    )
    write_parquet(products, output_dir / "products.parquet")
    print(f"    elapsed: {time.time() - t0:.2f}s")

    print("\n" + "=" * 64)
    print(f"Phase 1 done in {time.time() - total_t0:.2f}s")
    print("Next: orders + order_items + events + payments + shipments")
    print("=" * 64)

    # 4. Orders (users에 의존, 가장 큰 팩트 테이블)
    print("\n[4/8] Orders")
    t0 = time.time()
    orders = generate_orders.generate(
        users_df=users,
        end_date=end_date,
        seed=seed,
    )
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 5. Order Items (orders, products에 의존, orders도 업데이트)
    print("\n[5/8] Order Items + Update Orders")
    t0 = time.time()
    order_items, orders = generate_order_items.generate(
        orders_df=orders,
        products_df=products,
        seed=seed,
    )
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 6. Events (users/products/orders/order_items에 의존, 클릭스트림)
    print("\n[6/8] Events")
    t0 = time.time()
    events = generate_events.generate(
        users_df=users,
        products_df=products,
        orders_df=orders,
        order_items_df=order_items,
        end_date=end_date,
        seed=seed,
        browse_sessions_per_user=browse_sessions_per_user,
        max_events_per_session=max_events_per_session,
        null_rate_search=null_rate_search,
    )
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 7. Payments (orders에 의존, 주문당 1결제)
    print("\n[7/8] Payments")
    t0 = time.time()
    payments = generate_payments.generate(orders_df=orders, end_date=end_date, seed=seed)
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 8. Shipments (orders에 의존, 발송된 주문만)
    print("\n[8/8] Shipments")
    t0 = time.time()
    shipments = generate_shipments.generate(orders_df=orders, end_date=end_date, seed=seed)
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 이제 다섯 팩트 테이블 저장 (모두 dt 파티션)
    print("\nWriting partitioned facts...")
    write_parquet(orders, output_dir / "orders", partition_by=["dt"])
    write_parquet(order_items, output_dir / "order_items", partition_by=["dt"])
    write_parquet(events, output_dir / "events", partition_by=["dt"])
    write_parquet(payments, output_dir / "payments", partition_by=["dt"])
    write_parquet(shipments, output_dir / "shipments", partition_by=["dt"])


if __name__ == "__main__":
    main()
