"""유저 차원 테이블 생성기.

실전 느낌을 위한 패턴들:
- 가입일 분포: beta distribution (최근으로 갈수록 가입자 증가)
- 세그먼트: new / regular / vip / churned (가장 분포)
- 국가 분포: 가중치 기반 (미국 비중 높게)
- 이메일 도메인: 실제 분포 흉내
- 5% gender NULL (dirty data)
"""
from datetime import date, datetime, timedelta
from typing import cast

import numpy as np
import polars as pl

from .base import inject_nulls, make_rng, weighted_choice

# 이름 풀 (실제 frequency 기반 US 인구 통계)
FIRST_NAMES_M = [
    "James", "John", "Robert", "Michael", "William", "David", "Joseph",
    "Charles", "Thomas", "Daniel", "Matthew", "Anthony", "Mark", "Steven",
    "Andrew", "Paul", "Joshua", "Kevin", "Brian", "George", "Edward",
    "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
    "Nicholas", "Eric", "Stephen", "Jonathan", "Larry", "Justin", "Scott",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara",
    "Susan", "Jessica", "Karen", "Lisa", "Nancy", "Betty", "Sandra",
    "Margaret", "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Laura",
    "Sharon", "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Kim", "Park", "Choi", "Tanaka", "Suzuki", "Schmidt", "Mueller",
]

COUNTRIES_WEIGHTED = [
    ("US", 0.45), ("UK", 0.10), ("CA", 0.07), ("AU", 0.05),
    ("DE", 0.06), ("FR", 0.05), ("KR", 0.04), ("JP", 0.04),
    ("BR", 0.04), ("IN", 0.05), ("MX", 0.03), ("IT", 0.02),
]

# 세그먼트 분포: 평범한 고객이 다수, vip 소수, churned 일정 비율
SEGMENTS_WEIGHTED = [
    ("new", 0.30),       # 가입했지만 활동 적음
    ("regular", 0.45),   # 일반 고객
    ("vip", 0.10),       # 고액/빈번 구매
    ("churned", 0.15),   # 이탈
]

EMAIL_PROVIDERS_WEIGHTED = [
    ("gmail.com", 0.45), ("yahoo.com", 0.15), ("outlook.com", 0.12),
    ("hotmail.com", 0.08), ("icloud.com", 0.10), ("naver.com", 0.06),
    ("kakao.com", 0.04),
]


def generate(
    n_users: int,
    start_date: date,
    end_date: date,
    null_rate_gender: float = 0.05,
    seed: int = 42,
) -> pl.DataFrame:
    """유저 차원 테이블 생성"""
    rng = make_rng(seed)

    user_ids = np.arange(1, n_users + 1, dtype=np.int64)

    # 성별 (NULL 5% 주입)
    genders_raw = rng.choice(
        np.array(["M", "F", "other"]), size=n_users, p=[0.48, 0.49, 0.03]
    )
    genders = inject_nulls(genders_raw, null_rate=null_rate_gender, rng=rng)

    # 이름: 성별 기반 (벡터화)
    first_names_arr = np.empty(n_users, dtype=object)
    m_mask = genders_raw == "M"
    f_mask = genders_raw == "F"
    o_mask = ~(m_mask | f_mask)
    first_names_arr[m_mask] = rng.choice(FIRST_NAMES_M, size=int(m_mask.sum()))
    first_names_arr[f_mask] = rng.choice(FIRST_NAMES_F, size=int(f_mask.sum()))
    first_names_arr[o_mask] = rng.choice(
        FIRST_NAMES_M + FIRST_NAMES_F, size=int(o_mask.sum())
    )
    last_names = rng.choice(LAST_NAMES, size=n_users)

    # 이메일 도메인 (가중 분포)
    providers = weighted_choice(
        rng,
        [p for p, _ in EMAIL_PROVIDERS_WEIGHTED],
        [w for _, w in EMAIL_PROVIDERS_WEIGHTED],
        n_users
    )
    emails = [
        f"{f.lower()}.{last.lower()}{i}@{p}"
        for f, last, i, p in zip(first_names_arr, last_names, user_ids, providers, strict=False)
    ]

    # 가입일: beta(2, 1.5) - 최근으로 갈수록 가입자 증가 (성장하는 서비스)
    total_days = (end_date - start_date).days
    signup_day_offsets = (rng.beta(2, 1.5, size=n_users) * total_days).astype(int)
    signup_seconds_in_day = rng.integers(0, 86400, size=n_users)

    start_dt = datetime.combine(start_date, datetime.min.time())
    created_at = [
        start_dt + timedelta(days=int(d), seconds=int(s))
        for d, s in zip(signup_day_offsets, signup_seconds_in_day, strict=False)
    ]

    # 국가
    countries = weighted_choice(
        rng,
        [c for c, _ in COUNTRIES_WEIGHTED],
        [w for _, w in COUNTRIES_WEIGHTED],
        n_users,
    )

    # 세그먼트
    segments = weighted_choice(
        rng,
        [s for s, _ in SEGMENTS_WEIGHTED],
        [w for _, w in SEGMENTS_WEIGHTED],
        n_users,
    )

    # 생년월일: 평균 35세, 표준편차 12 (clip 18-75)
    ages = rng.normal(35, 12, size=n_users).clip(18, 75).astype(int)
    birth_years = end_date.year - ages
    birth_months = rng.integers(1, 13, size=n_users)
    birth_days = rng.integers(1, 28, size=n_users) # 28일 max로 안전하게
    birth_dates = [
        date(int(y), int(m), int(d))
        for y, m, d in zip(birth_years, birth_months, birth_days, strict=False)
    ]

    # is_actives: churned면 무조건 False, 나머지는 95% 활성
    is_active = np.where(
        segments == "churned", False, rng.random(n_users) > 0.05
    )

    df = pl.DataFrame({
        "user_id": user_ids,
        "email": emails,
        "first_name": first_names_arr.tolist(),
        "last_name": last_names.tolist(),
        "gender": cast(list, genders.tolist()),
        "birth_date": birth_dates,
        "country": countries.tolist(),
        "segment": segments.tolist(),
        "is_active": is_active.tolist(),
        "created_at": created_at,
    })

    return df