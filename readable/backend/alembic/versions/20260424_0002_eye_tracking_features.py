"""Add eye tracking feature metrics table."""

from alembic import op
import sqlalchemy as sa


revision = "20260424_0002"
down_revision = "20260423_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "eye_tracking_features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fixation_duration_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("saccade_length", sa.Float(), nullable=False, server_default="0"),
        sa.Column("regression_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("skipped_words", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reading_speed_wpm", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(op.f("ix_eye_tracking_features_id"), "eye_tracking_features", ["id"], unique=False)
    op.create_index(
        op.f("ix_eye_tracking_features_session_id"), "eye_tracking_features", ["session_id"], unique=False
    )
    op.create_index(
        op.f("ix_eye_tracking_features_student_id"), "eye_tracking_features", ["student_id"], unique=False
    )
    op.create_unique_constraint(
        "uq_eye_tracking_features_session_id", "eye_tracking_features", ["session_id"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_eye_tracking_features_session_id", "eye_tracking_features", type_="unique")
    op.drop_index(op.f("ix_eye_tracking_features_student_id"), table_name="eye_tracking_features")
    op.drop_index(op.f("ix_eye_tracking_features_session_id"), table_name="eye_tracking_features")
    op.drop_index(op.f("ix_eye_tracking_features_id"), table_name="eye_tracking_features")
    op.drop_table("eye_tracking_features")
