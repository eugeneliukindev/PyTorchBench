import logging
import sys
from pathlib import Path
import pytest
from src.utils import configure_logger


class TestLogger:
    def test_configure_logger(self, tmp_path, caplog):
        log_path = tmp_path / "logs.log"
        caplog.set_level(logging.INFO)
        configure_logger("INFO", log_path)

        # Проверяем, что уровень логов соответствует ожидаемому
        logger = logging.getLogger()
        logger.info("Test message")
        assert "Test message" in caplog.text  # Проверяем, что сообщение записано
        assert logger.getEffectiveLevel() == logging.INFO
