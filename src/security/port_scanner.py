"""Port scanner using stdlib asyncio + ssl.

Zero third-party imports — pure stdlib (asyncio, socket, ssl, time).

Pattern: ``PortScanner.scan(host, ports) -> list[PortResult]``
"""

from __future__ import annotations

import asyncio
import os
import socket
import ssl
import time
from dataclasses import dataclass, field

# Fail-closed network gate: these utilities are disabled by default.
# Set AURELIUS_SECURITY_NETWORK_SCAN_ENABLED=1 to opt in.
def _network_enabled() -> bool:
    return os.environ.get("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", "").lower() == "1"


class PortScannerError(Exception):
    """Raised for invalid inputs or unrecoverable scanner errors."""


@dataclass
class PortResult:
    """Result of a single-port scan."""

    port: int
    is_open: bool = False
    service: str = ""
    banner: str = ""
    tls: bool = False
    response_time_ms: float = 0.0


@dataclass
class ScanSummary:
    """Aggregate result of a full host scan."""

    host: str = ""
    open_ports: list[PortResult] = field(default_factory=list)
    closed_ports: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def total_ports(self) -> int:
        return len(self.open_ports) + self.closed_ports + len(self.errors)

    @property
    def open_port_numbers(self) -> list[int]:
        return [p.port for p in self.open_ports]


# Well-known service names
_WELL_KNOWN: dict[int, str] = {
    20: "FTP-DATA",
    21: "FTP",
    22: "SSH",
    23: "TELNET",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    465: "SMTPS",
    5432: "PostgreSQL",
    3306: "MySQL",
    6379: "Redis",
    8080: "HTTP-Proxy",
    8443: "HTTPS-Alt",
    9000: "MinIO",
    9200: "Elasticsearch",
}

_SSL_PORTS = {443, 465, 587, 993, 995, 8443}


async def _probe(
    host: str, port: int, timeout: float, grab_banner: bool
) -> PortResult:
    result = PortResult(port=port)
    deadline = time.monotonic() + timeout

    def _service() -> str:
        return _WELL_KNOWN.get(port, "")

    # Plain TCP connect
    try:
        connect = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(connect, timeout=timeout)
        result.is_open = True
        result.service = _service()
        result.tls = port in _SSL_PORTS
        t0 = time.monotonic()
        result.response_time_ms = (t0 - deadline + timeout) * 1000

        if grab_banner and port in (22, 21, 25, 110, 143):
            try:
                banner = await asyncio.wait_for(
                    reader.readline(), timeout=timeout / 2
                )
                result.banner = banner.decode(errors="replace").strip()
            except asyncio.TimeoutError:
                pass

        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
        pass

    return result


async def _scan_async(
    host: str, ports: list[int], concurrency: int, grab_banner: bool, timeout: float
) -> ScanSummary:
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(port: int) -> PortResult:
        async with semaphore:
            return await _probe(host, port, timeout, grab_banner)

    tasks = [asyncio.create_task(bounded(p)) for p in ports]
    raw_results: list[PortResult] = list(await asyncio.gather(*tasks))

    open_ports: list[PortResult] = []
    closed = 0
    errors: list[str] = []

    for pr in raw_results:
        if pr.is_open:
            open_ports.append(pr)
        else:
            closed += 1

    return ScanSummary(
        host=host,
        open_ports=open_ports,
        closed_ports=closed,
        errors=errors,
    )


class PortScanner:
    """High-performance stdlib port scanner.

    Parameters
    ----------
    concurrency : int
        Max simultaneous probes.
    timeout : float
        Per-port connect timeout in seconds.
    grab_banner : bool
        Attempt to read service banner on well-known ports.
    """

    def __init__(
        self,
        concurrency: int = 200,
        timeout: float = 1.5,
        grab_banner: bool = True,
    ) -> None:
        if concurrency < 1:
            raise PortScannerError(f"concurrency must be >= 1, got {concurrency}")
        if timeout <= 0:
            raise PortScannerError(f"timeout must be > 0, got {timeout}")
        self.concurrency = concurrency
        self.timeout = timeout
        self.grab_banner = grab_banner

    def scan(
        self, host: str, ports: list[int] | None = None
    ) -> ScanSummary:
        """Scan *host* across the given *ports* and return the summary.

        Args:
            host: IP address or hostname.
            ports: List of TCP port numbers.  If *None*, scans well-known ports.
        """
        if not _network_enabled():
            raise PortScannerError(
                "Network scanning is disabled by default. "
                "Set AURELIUS_SECURITY_NETWORK_SCAN_ENABLED=1 to opt in."
            )
        if not host:
            raise PortScannerError("host must be a non-empty string")
        if ports is None:
            ports = sorted(_WELL_KNOWN.keys())
        if any(p < 1 or p > 65535 for p in ports):
            raise PortScannerError("port numbers must be in range 1..65535")
        unique = sorted(set(ports))
        return asyncio.run(
            _scan_async(host, unique, self.concurrency, self.grab_banner, self.timeout)
        )


# Thread-blocking helper — avoids ``asyncio.run(None)`` in pytest fixtures
_LOOP = asyncio.new_event_loop()


def scan_thread_safe(
    host: str, ports: list[int] | None = None, timeout: float = 1.5
) -> ScanSummary:
    """Synchronous wrapper suitable for use from pytest/threaded contexts."""
    scanner = PortScanner(timeout=timeout)
    return scanner.scan(host, ports)
