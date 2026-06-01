"""가짜 데이터 생성 메인 엔트리포인트.

콘솔 스크립트(`generate-data`) 및 Databricks python_wheel_task의 진입점.

사용법:
    generate-data                          # 패키지 내장 config.yml 사용
    generate-data --output-dir /Volumes/.../raw
    generate-data --config /path/to/config.yml

Phase 1: 차원 테이블 (categories, users, products)
Phase 2: 팩트 테이블 (orders, order_items)
"""
import argparse
import time
from datetime import date
from pathlib import Path

from . import (
    generate_categories,
    generate_order_items,
    generate_orders,
    generate_products,
    generate_users,
)
from .base import load_config, write_parquet


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E-Commerce 가짜 데이터 생성기")
    parser.add_argument(
        "--config",
        default=None,
        help="config.yml 경로 (기본: 패키지에 내장된 config.yml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="출력 디렉토리 (config.yml의 output_dir을 덮어씀)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    start_date = date.fromisoformat(cfg["period"]["start"])
    end_date = date.fromisoformat(cfg["period"]["end"])
    seed = cfg["seed"]
    output_dir = Path(args.output_dir if args.output_dir else cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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
        n_users=cfg["volumes"]["users"],
        start_date=start_date,
        end_date=end_date,
        null_rate_gender=cfg["dirty_data"]["null_rate_gender"],
        seed=seed,
    )
    write_parquet(users, output_dir / "users.parquet")
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 3. Products (categories에 의존)
    print("\n[3/3] Products")
    t0 = time.time()
    products = generate_products.generate(
        n_products=cfg["volumes"]["products"],
        categories_df=categories,
        start_date=start_date,
        end_date=end_date,
        null_rate_brand=cfg["dirty_data"]["null_rate_brand"],
        discontinued_rate=cfg["dirty_data"]["discontinued_rate"],
        seed=seed,
    )
    write_parquet(products, output_dir / "products.parquet")
    print(f"    elapsed: {time.time() - t0:.2f}s")

    print("\n" + "=" * 64)
    print(f"Phase 1 done in {time.time() - total_t0:.2f}s")
    print("Next: orders + order_items + payments + shipments + events")
    print("=" * 64)

    # 4. Orders (users에 의존, 가장 큰 팩트 테이블)
    print("\n[4/4] Orders")
    t0 = time.time()
    orders = generate_orders.generate(
        users_df=users,
        end_date=end_date,
        seed=seed,
    )
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 5. Order Items (orders, products에 의존, orders도 업데이트)
    print("\n[5/5] Order Items + Update Orders")
    t0 = time.time()
    order_items, orders = generate_order_items.generate(
        orders_df=orders,
        products_df=products,
        seed=seed,
    )
    print(f"    elapsed: {time.time() - t0:.2f}s")

    # 이제 두 팩트 테이블 저장 (둘 다 dt 파티션)
    print("\nWriting partitioned facts...")
    write_parquet(orders, output_dir / "orders", partition_by=["dt"])
    write_parquet(order_items, output_dir / "order_items", partition_by=["dt"])


if __name__ == "__main__":
    main()
