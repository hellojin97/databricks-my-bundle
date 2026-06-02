"""주문 팩트 테이블 검증."""
from datetime import datetime

from .conftest import END_DATE

EXPECTED_COLUMNS = {
    "order_id", "user_id", "created_at", "status",
    "payment_method", "currency", "fx_rate", "total_amount", "dt",
}
VALID_STATUSES = {
    "pending", "processing", "shipped", "delivered", "cancelled", "refunded",
}


def test_schema(orders):
    assert set(orders.columns) == EXPECTED_COLUMNS


def test_order_id_unique(orders):
    assert orders["order_id"].n_unique() == len(orders)


def test_user_fk_valid(orders, users):
    """모든 주문의 user_id 는 실제 유저여야 한다."""
    user_ids = set(users["user_id"].to_list())
    assert set(orders["user_id"].unique().to_list()) <= user_ids


def test_status_valid(orders):
    assert set(orders["status"].unique().to_list()) <= VALID_STATUSES


def test_created_at_not_after_end_date(orders):
    assert orders["created_at"].max() <= datetime(END_DATE.year, END_DATE.month, END_DATE.day, 23, 59, 59)


def test_order_after_user_signup(orders, users):
    """주문은 유저 가입일 이후에만 발생해야 한다 (시간적 FK)."""
    joined = orders.join(
        users.select(["user_id", "created_at"]).rename({"created_at": "signup_at"}),
        on="user_id",
    )
    assert (joined["created_at"] >= joined["signup_at"]).all()


def test_dt_matches_created_at_date(orders):
    """파티션 키 dt 는 created_at 의 날짜와 일치해야 한다."""
    mismatch = orders.filter(orders["created_at"].dt.date() != orders["dt"])
    assert len(mismatch) == 0
