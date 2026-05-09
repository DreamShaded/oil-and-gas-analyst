from __future__ import annotations

from src.config import get_settings
from src.ui.chat import build_chat
from src.utils.logging import configure_logging, get_logger


def _ensure_indexed() -> None:
    from pathlib import Path
    log = get_logger("app.bootstrap")
    try:
        from src.rag.retrieval.qdrant_store import QdrantStore
        store = QdrantStore()
        size = store.collection_size()
    except Exception as e:
        log.warning("startup.qdrant_check_failed", error=str(e))
        return
    if size > 0:
        log.info("startup.index_ready", points=size)
        return

    snapshot = Path("data/snapshots") / f"{store.collection}.snapshot"
    if snapshot.exists():
        log.info("startup.restoring_snapshot", path=str(snapshot))
        try:
            if store.restore_snapshot(snapshot):
                log.info("startup.index_ready_from_snapshot", points=store.collection_size())
                return
        except Exception as e:
            log.warning("startup.snapshot_restore_failed", error=str(e))

    log.info("startup.first_run_bootstrap_begin",
             hint="Snapshot не найден. Качаю EIA/OPEC/IEA/Минфин и индексирую. ~3-5 минут.")
    from src.data import bootstrap as bs
    summary = bs.run()
    log.info("startup.first_run_bootstrap_done", **summary)


def run() -> None:
    settings = get_settings()
    configure_logging(env=settings.app_env, level="INFO")
    log = get_logger("app")
    _ensure_indexed()
    log.info("app.start", host=settings.app_host, port=settings.app_port, env=settings.app_env)
    build_chat().launch(server_name=settings.app_host, server_port=settings.app_port)


if __name__ == "__main__":
    run()
