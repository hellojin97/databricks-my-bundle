"""유저 차원 테이블 검증."""
from .conftest import END_DATE, N_USERS, NULL_RATE_GENDER, START_DATE

EXPECTED_COLUMNS = {
    "user_id", "email", "first_name", "last_name", "gender",
    "birth_date", "country", "segment", "is_active", "created_at",
}
VALID_SEGMENTS = {"new", "regular", "vip", "churned"}


def test_schema(users):
    assert set(users.columns) == EXPECTED_COLUMNS


def test_row_count(users):
    assert len(users) == N_USERS


def test_user_id_unique(users):
    assert users["user_id"].n_unique() == N_USERS


def test_segments_valid(users):
    assert set(users["segment"].unique().to_list()) <= VALID_SEGMENTS


def test_gender_null_rate_in_range(users):
    """주입된 gender NULL 비율이 목표치 부근(±2%p)이어야 한다."""
    rate = users["gender"].null_count() / len(users)
    assert abs(rate - NULL_RATE_GENDER) < 0.02


def test_churned_users_are_inactive(users):
    """churned 세그먼트는 무조건 is_active=False (generate_users 규칙)."""
    churned = users.filter(users["segment"] == "churned")
    assert churned["is_active"].sum() == 0


def test_created_at_within_period(users):
    assert users["created_at"].min().date() >= START_DATE
    assert users["created_at"].max().date() <= END_DATE


def test_email_contains_at(users):
    assert all("@" in e for e in users["email"].to_list())
