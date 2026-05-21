"""Subdomain enumeration using stdlib only (socket + concurrent.futures).

Zero third-party imports — pure stdlib.

Pattern: ``SubdomainEnumerator.resolve(domain, wordlist) -> list[SubdomainResult]``
"""

from __future__ import annotations

import concurrent.futures
import os
import socket
from dataclasses import dataclass, field

def _network_enabled() -> bool:
    return os.environ.get("AURELIUS_SECURITY_NETWORK_SCAN_ENABLED", "").lower() == "1"


class SubdomainEnumError(Exception):
    """Raised for invalid inputs or network failures."""


@dataclass
class SubdomainResult:
    """One resolved subdomain."""

    name: str
    ip_addresses: list[str] = field(default_factory=list)
    is_cname: bool = False
    error: str = ""


@dataclass
class EnumSummary:
    """Aggregate result of a subdomain enumeration run."""

    domain: str = ""
    found: list[SubdomainResult] = field(default_factory=list)
    not_found: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def total_checked(self) -> int:
        return len(self.found) + self.not_found + len(self.errors)

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.found]


def _resolve_one(domain: str, sub: str, timeout: float) -> SubdomainResult:
    fqdn = f"{sub}.{domain}" if sub else domain
    try:
        # Query A record via getaddrinfo
        infos = socket.getaddrinfo(
            fqdn, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
        ips = sorted({str(info[4][0]) for info in infos if info})
        return SubdomainResult(name=fqdn, ip_addresses=ips)
    except socket.gaierror:
        return SubdomainResult(name=fqdn, error="not_resolved")
    except OSError as exc:
        return SubdomainResult(name=fqdn, error=str(exc))


class SubdomainEnumerator:
    """Resolve a wordlist of subdomains against *domain* via DNS.

    Parameters
    ----------
    concurrency : int
        Worker threads for DNS lookups.
    timeout : float
        Per-name DNS lookup timeout (seconds).
    """

    def __init__(self, concurrency: int = 50, timeout: float = 2.0) -> None:
        if concurrency < 1:
            raise SubdomainEnumError(
                f"concurrency must be >= 1, got {concurrency}"
            )
        if timeout <= 0:
            raise SubdomainEnumError(f"timeout must be > 0, got {timeout}")
        self.concurrency = concurrency
        self.timeout = timeout

    def resolve(
        self, domain: str, wordlist: list[str] | None = None
    ) -> EnumSummary:
        """Resolve *wordlist* against *domain* and return findings.

        Args:
            domain: Bare domain to scan (e.g. ``example.com``).
            wordlist: Subdomain prefixes (e.g. ``["www", "api", "dev"]``).
                      If *None*, an empty default is used.
        """
        if not _network_enabled():
            raise SubdomainEnumError(
                "Network enumeration is disabled by default. "
                "Set AURELIUS_SECURITY_NETWORK_SCAN_ENABLED=1 to opt in."
            )
        if not domain or "/" in domain:
            raise SubdomainEnumError(f"invalid domain: {domain!r}")
        wordlist = wordlist or []
        found: list[SubdomainResult] = []
        not_found = 0
        errors: list[str] = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as pool:
            futures = {
                pool.submit(_resolve_one, domain, sub, self.timeout): sub
                for sub in wordlist
            }
            for fut in concurrent.futures.as_completed(futures, timeout=self.timeout * len(wordlist) + 10):
                try:
                    result = fut.result(timeout=self.timeout)
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                if result.error == "not_resolved":
                    not_found += 1
                elif result.error:
                    errors.append(f"{result.name}: {result.error}")
                else:
                    found.append(result)

        found.sort(key=lambda r: r.name)
        return EnumSummary(
            domain=domain,
            found=found,
            not_found=not_found,
            errors=errors,
        )
