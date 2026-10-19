from alembic import context
from sqlalchemy import engine_from_config, pool
from backend.app.core.settings import SETTINGS
from backend.app.db import models  # noqa: F401  (register tables)
from backend.app.db.database import Base

config = context.config
config.set_main_option("sqlalchemy.url", SETTINGS.database_url.replace("%", "%%"))
target_metadata = Base.metadata


def run_migrations_online():
    cfg = config.get_section(config.config_ini_section) or {}
    eng = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with eng.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=conn.dialect.name == "sqlite", compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
