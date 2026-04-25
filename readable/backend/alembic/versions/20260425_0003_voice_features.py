"""Add voice feature metrics table."""

from alembic import op
import sqlalchemy as sa


revision = "20260425_0003"
down_revision = "20260424_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "voice_features",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("student_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speech_rate_wps", sa.Float(), nullable=False, server_default="0"),
        sa.Column("pause_duration_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("pause_frequency", sa.Float(), nullable=False, server_default="0"),
        sa.Column("mispronunciation_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("repetition_rate", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(op.f("ix_voice_features_id"), "voice_features", ["id"], unique=False)
    op.create_index(op.f("ix_voice_features_session_id"), "voice_features", ["session_id"], unique=False)
    op.create_index(op.f("ix_voice_features_student_id"), "voice_features", ["student_id"], unique=False)
    op.create_unique_constraint("uq_voice_features_session_id", "voice_features", ["session_id"])


def downgrade() -> None:
    op.drop_constraint("uq_voice_features_session_id", "voice_features", type_="unique")
    op.drop_index(op.f("ix_voice_features_student_id"), table_name="voice_features")
    op.drop_index(op.f("ix_voice_features_session_id"), table_name="voice_features")
    op.drop_index(op.f("ix_voice_features_id"), table_name="voice_features")
    op.drop_table("voice_features")
