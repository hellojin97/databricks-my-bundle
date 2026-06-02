"""공통 유틸리티 모듈.

모든 generator가 공유하는 헬퍼들:
- 시드 기반 RNG (재현성)
- Parquet 쓰기 (선택적 파티셔닝)
- NULL 주입 (dirty data 시뮬레이션)
- 가중 랜덤 선택
"""
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def load_config(path: str | Path | None = None) -> dict:
    """YAML 설정 파일 로드.

    path가 None이면 패키지에 내장된 config.yml을 사용한다.
    wheel로 설치되면 CWD가 불확실하므로 패키지 기준 경로로 찾는다.
    """
    if path is None:
        path = Path(__file__).parent / "config.yml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_rng(seed: int) -> np.random.Generator:
    """시드 기반 RNG 생성. 같은 시드면 항상 같은 결과가 나옴(재현성)."""
    return np.random.default_rng(seed)


def write_parquet(
    df: pl.DataFrame,
    path: str | Path,
    partition_by: list[str] | None = None,
) -> None:
    """DataFrame을 Parquet으로 저장, 선택적으로 파티셔닝.
    
    팩트 테이블(orders, events)은 dt 컬럼으로 파티셔닝 예정.
    차원 테이블(users, products)은 파티셔닝 없이 단일 파일.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_by:
        # polars 네이티브 hive 파티션 쓰기 (pyarrow 불필요)
        df.write_parquet(path, partition_by=partition_by, mkdir=True)
    else:
        df.write_parquet(path)

    size_mb = (path.stat().st_size / 1024 / 1024) if path.is_file() else 0
    print(f"    -> {len(df):>10,} rows | {size_mb:>6.2f} MB | {path}")


def inject_nulls(
    arr: np.ndarray,
    null_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """일부 값을 NULL로 교체. 실적 데이터처럼 누락값 주입용."""
    if null_rate <= 0:
        return arr
    mask = rng.random(len(arr)) < null_rate
    result = arr.astype(object).copy()
    result[mask] = None
    return result


def weighted_choice(
    rng: np.random.Generator,
    choices: list,
    weights: list,
    size: int,
) -> np.ndarray:
    """벡터화된 가중 랜덤 선택. for-loop보다 100배 빠름."""
    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(choices, size=size, p=probs)
