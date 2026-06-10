from __future__ import annotations

import sys
from pathlib import Path

from qsar_tl.config import load_config


def main() -> None:
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QMainWindow,
            QPushButton,
            QPlainTextEdit,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("PySide6 is required to run the desktop GUI.") from exc

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ECOTOX-QSAR Training Console")
            self.resize(1180, 760)

            root = QWidget()
            layout = QHBoxLayout(root)

            self.nav = QListWidget()
            self.nav.addItems(
                [
                    "实验配置",
                    "训练队列",
                    "实时日志",
                    "指标监控",
                    "输出管理",
                    "实验备注",
                ]
            )
            self.nav.setMaximumWidth(180)

            self.pages = QStackedWidget()
            self.pages.addWidget(self._config_page())
            self.pages.addWidget(self._queue_page())
            self.pages.addWidget(self._log_page())
            self.pages.addWidget(self._metrics_page())
            self.pages.addWidget(self._outputs_page())
            self.pages.addWidget(self._notes_page())

            self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
            self.nav.setCurrentRow(0)

            layout.addWidget(self.nav)
            layout.addWidget(self.pages, stretch=1)
            self.setCentralWidget(root)

        def _config_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            title = QLabel("实验配置")
            title.setAlignment(Qt.AlignLeft)
            layout.addWidget(title)
            layout.addWidget(QLabel("第一版将读取 YAML 配置；后续在此页补充表单化参数编辑。"))
            load_button = QPushButton("载入示例配置")
            load_button.clicked.connect(self._load_example_config)
            layout.addWidget(load_button)
            self.config_preview = QPlainTextEdit()
            self.config_preview.setReadOnly(True)
            layout.addWidget(self.config_preview)
            return page

        def _queue_page(self) -> QWidget:
            return self._placeholder_page("训练队列", "后续接入本机与远程训练执行器、任务状态和停止控制。")

        def _log_page(self) -> QWidget:
            return self._placeholder_page("实时日志", "后续显示训练脚本 stdout/stderr 和远程日志同步结果。")

        def _metrics_page(self) -> QWidget:
            return self._placeholder_page("指标监控", "后续读取 metrics 文件，展示 R2、RMSE、MAE、Huber loss。")

        def _outputs_page(self) -> QWidget:
            return self._placeholder_page("输出管理", "后续管理模型目录、图表目录、配置文件和日志文件。")

        def _notes_page(self) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel("实验备注"))
            notes = QPlainTextEdit()
            notes.setPlaceholderText("记录每次训练目的、参数变更、异常现象和科研解释。")
            layout.addWidget(notes)
            return page

        def _placeholder_page(self, title: str, body: str) -> QWidget:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(title))
            layout.addWidget(QLabel(body))
            layout.addStretch()
            return page

        def _load_example_config(self) -> None:
            config_path = Path("configs/experiment.example.yaml")
            config = load_config(config_path)
            self.config_preview.setPlainText(str(config))

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

