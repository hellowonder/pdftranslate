#!/usr/bin/env python3

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ServerConfig:
    name: str
    model: str
    port: int


class VllmSupervisor:
    def __init__(self) -> None:
        self.vllm_bin = os.environ.get("VLLM_BIN", "./.vllm-env/bin/vllm")
        self.startup_timeout = float(os.environ.get("STARTUP_TIMEOUT", "600"))
        self.startup_poll_interval = float(os.environ.get("STARTUP_POLL_INTERVAL", "2"))
        self.cleaned_up = False
        self.processes: list[subprocess.Popen[bytes]] = []
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigterm)

    def _handle_sigint(self, signum: int, frame: Optional[object]) -> None:
        del signum, frame
        self.cleanup(signal.SIGINT)
        raise SystemExit(130)

    def _handle_sigterm(self, signum: int, frame: Optional[object]) -> None:
        del signum, frame
        self.cleanup(signal.SIGTERM)
        raise SystemExit(143)

    def start_server(self, config: ServerConfig) -> subprocess.Popen[bytes]:
        print(f"Starting {config.model} on port {config.port}...")
        process = subprocess.Popen(
            [
                self.vllm_bin,
                "serve",
                config.model,
                "--enable-sleep-mode",
                "--port",
                str(config.port),
            ],
            env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
        )
        self.processes.append(process)
        return process

    def wait_for_server_ready(self, config: ServerConfig, process: subprocess.Popen[bytes]) -> None:
        print(f"Waiting for {config.model} on port {config.port} to become ready...")
        deadline = time.monotonic() + self.startup_timeout

        while True:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{config.model} on port {config.port} exited before becoming ready with status {exit_code}."
                )

            if self._healthcheck(config.port):
                print(f"{config.model} on port {config.port} is ready.")
                return

            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for {config.model} on port {config.port} to become ready.")

            time.sleep(self.startup_poll_interval)

    def send_sleep_command(self, config: ServerConfig) -> None:
        print(f"Sending sleep command to {config.model} on port {config.port}...")
        self._post(f"http://127.0.0.1:{config.port}/sleep?level=1")

    def start_server_and_sleep(self, config: ServerConfig) -> None:
        process = self.start_server(config)
        self.wait_for_server_ready(config, process)
        self.send_sleep_command(config)

    def run(self) -> int:
        servers = [
            ServerConfig(
                name="ocr",
                model=os.environ.get("OCR_MODEL", "deepseek-ai/DeepSeek-OCR"),
                port=int(os.environ.get("OCR_PORT", "8000")),
            ),
            ServerConfig(
                name="translation",
                model=os.environ.get("TRANSLATION_MODEL", "google/gemma-4-26B-A4B-it"),
                port=int(os.environ.get("TRANSLATION_PORT", "8001")),
            ),
        ]

        try:
            for config in servers:
                self.start_server_and_sleep(config)

            return self.monitor_processes()
        except BaseException:
            self.cleanup(signal.SIGTERM)
            raise

    def monitor_processes(self) -> int:
        while True:
            for process in self.processes:
                exit_code = process.poll()
                if exit_code is not None:
                    print(f"A vLLM process exited with status {exit_code}, stopping the other one.")
                    self.cleanup(signal.SIGTERM)
                    return exit_code
            time.sleep(1)

    def cleanup(self, sig: signal.Signals) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True

        for process in self.processes:
            if process.poll() is None:
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    pass

        for process in self.processes:
            try:
                process.wait()
            except Exception:
                pass

    @staticmethod
    def _healthcheck(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    @staticmethod
    def _post(url: str) -> None:
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Unexpected status code {response.status} for {url}")


def main() -> int:
    supervisor = VllmSupervisor()
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
