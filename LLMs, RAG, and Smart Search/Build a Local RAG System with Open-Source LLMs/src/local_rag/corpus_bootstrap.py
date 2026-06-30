"""Download and prepare a realistic Linux-focused corpus for local RAG."""

from __future__ import annotations

import asyncio
import tarfile
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from loguru import logger


@dataclass(slots=True)
class DownloadItem:
    """Single downloadable resource."""

    url: str
    relative_path: Path


@dataclass(slots=True)
class ArchiveItem:
    """Archive source with path filters."""

    url: str
    relative_path: Path
    include_prefixes: tuple[str, ...]


class CorpusBootstrapper:
    """One-time corpus bootstrapper.

    Downloads large public Linux documentation corpus with mixed PDF/MD/TXT assets.
    """

    def __init__(
        self,
        documents_dir: Path,
        timeout_seconds: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.documents_dir = documents_dir
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self.files: tuple[DownloadItem, ...] = (
            DownloadItem(
                "https://www.gnu.org/software/bash/manual/bash.pdf",
                Path("gnu_manuals/pdf/bash_reference_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/bash/manual/bash.txt",
                Path("gnu_manuals/txt/bash_reference_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/coreutils/manual/coreutils.pdf",
                Path("gnu_manuals/pdf/coreutils_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/coreutils/manual/coreutils.txt",
                Path("gnu_manuals/txt/coreutils_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/sed/manual/sed.pdf",
                Path("gnu_manuals/pdf/sed_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/sed/manual/sed.txt",
                Path("gnu_manuals/txt/sed_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/grep/manual/grep.pdf",
                Path("gnu_manuals/pdf/grep_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/grep/manual/grep.txt",
                Path("gnu_manuals/txt/grep_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/tar/manual/tar.pdf",
                Path("gnu_manuals/pdf/tar_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/tar/manual/tar.txt",
                Path("gnu_manuals/txt/tar_manual.txt"),
            ),
        )

        self.archives: tuple[ArchiveItem, ...] = (
            ArchiveItem(
                url="https://mirrors.edge.kernel.org/pub/linux/docs/man-pages/man-pages-6.18.tar.xz",
                relative_path=Path("linux/man-pages-6.18.tar.xz"),
                include_prefixes=("man-pages-6.18/man",),
            ),
            ArchiveItem(
                url="https://github.com/systemd/systemd/archive/refs/tags/v261.tar.gz",
                relative_path=Path("linux/systemd-v261.tar.gz"),
                include_prefixes=("systemd-261/docs/",),
            ),
            ArchiveItem(
                url="https://github.com/torvalds/linux/archive/refs/tags/v6.9.tar.gz",
                relative_path=Path("linux/linux-v6.9.tar.gz"),
                include_prefixes=(
                    "linux-6.9/Documentation/admin-guide/",
                    "linux-6.9/Documentation/userspace-api/",
                ),
            ),
        )

    async def bootstrap(self) -> None:
        """Download all configured resources."""

        self.documents_dir.mkdir(parents=True, exist_ok=True)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            file_results = await asyncio.gather(
                *(self._download_file(session, item) for item in self.files),
                return_exceptions=True,
            )
            archive_results = await asyncio.gather(
                *(self._download_archive(session, item) for item in self.archives),
                return_exceptions=True,
            )

        successes = 0
        failures: list[str] = []
        for idx, result in enumerate(file_results, start=1):
            if isinstance(result, Exception):
                failures.append(f"file[{idx}]: {result}")
            elif result:
                successes += 1
        for idx, result in enumerate(archive_results, start=1):
            if isinstance(result, Exception):
                failures.append(f"archive[{idx}]: {result}")
            elif result:
                successes += 1

        for failure in failures:
            logger.error("Corpus bootstrap failure: {}", failure)

        if successes == 0:
            raise RuntimeError(
                "Corpus bootstrap failed: no resources downloaded/extracted successfully."
            )

        logger.info("Corpus bootstrap completed with {} successful resources.", successes)

    async def _download_file(self, session: aiohttp.ClientSession, item: DownloadItem) -> bool:
        target = self.documents_dir / item.relative_path
        if target.exists() and target.stat().st_size > 0:
            logger.info("Skip existing file: {}", target)
            return True

        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading {} -> {}", item.url, target)

        payload = await self._fetch_bytes_with_retries(session, item.url)
        if payload is None:
            logger.error("Failed file download after retries: {}", item.url)
            if target.exists():
                target.unlink(missing_ok=True)
            return False

        target.write_bytes(payload)
        return True

    async def _download_archive(self, session: aiohttp.ClientSession, item: ArchiveItem) -> bool:
        archive_path = self.documents_dir / item.relative_path
        extracted_dir = self.documents_dir / item.relative_path.with_suffix("").with_suffix("")
        done_marker = extracted_dir / ".extracted"

        if done_marker.exists():
            logger.info("Skip existing archive extract: {}", extracted_dir)
            return True

        archive_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading archive {}", item.url)
        payload = await self._fetch_bytes_with_retries(session, item.url)
        if payload is None:
            logger.error("Failed archive download after retries: {}", item.url)
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            return False

        archive_path.write_bytes(payload)

        logger.info("Extracting filtered archive {}", archive_path)
        with tarfile.open(archive_path) as tar:
            members = [
                member
                for member in tar.getmembers()
                if member.name.startswith(item.include_prefixes) and member.isfile()
            ]
            tar.extractall(path=extracted_dir, members=members)  # noqa: S202

        done_marker.write_text("ok", encoding="utf-8")
        return True

    async def _fetch_bytes_with_retries(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> bytes | None:
        for attempt in range(1, self.max_retries + 1):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.read()
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    logger.error(
                        "Download failed (attempt {}/{}): {} ({})",
                        attempt,
                        self.max_retries,
                        url,
                        exc,
                    )
                    return None
                wait_seconds = float(attempt)
                logger.warning(
                    "Download retry {}/{} for {} after error: {}",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )
                await asyncio.sleep(wait_seconds)
        return None
