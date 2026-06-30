"""Download and prepare a realistic mixed-domain corpus for local Document Q&A."""

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

    Builds mixed enterprise corpus across technical docs, policies, research, and finance.
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
            # Technical manuals (PDF + TXT)
            DownloadItem(
                "https://www.gnu.org/software/bash/manual/bash.pdf",
                Path("technical/gnu_manuals/pdf/bash_reference_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/bash/manual/bash.txt",
                Path("technical/gnu_manuals/txt/bash_reference_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/coreutils/manual/coreutils.pdf",
                Path("technical/gnu_manuals/pdf/coreutils_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/coreutils/manual/coreutils.txt",
                Path("technical/gnu_manuals/txt/coreutils_manual.txt"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/sed/manual/sed.pdf",
                Path("technical/gnu_manuals/pdf/sed_manual.pdf"),
            ),
            DownloadItem(
                "https://www.gnu.org/software/sed/manual/sed.txt",
                Path("technical/gnu_manuals/txt/sed_manual.txt"),
            ),
            # Policy / governance PDFs
            DownloadItem(
                "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
                Path("policy/nist/nist_sp_800_53r5.pdf"),
            ),
            DownloadItem(
                "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r3.pdf",
                Path("policy/nist/nist_sp_800_171r3.pdf"),
            ),
            DownloadItem(
                "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
                Path("policy/nist/nist_ai_rm_framework_100_1.pdf"),
            ),
            # Finance / annual reports
            DownloadItem(
                "https://www.berkshirehathaway.com/2023ar/2023ar.pdf",
                Path("finance/reports/berkshire_2023_annual_report.pdf"),
            ),
            DownloadItem(
                "https://www.sec.gov/files/dera_enforcement-and-litigation-quarterly_2023q4.pdf",
                Path("finance/reports/sec_enforcement_q4_2023.pdf"),
            ),
            # Research papers
            DownloadItem(
                "https://arxiv.org/pdf/1706.03762.pdf",
                Path("research/papers/attention_is_all_you_need.pdf"),
            ),
            DownloadItem(
                "https://arxiv.org/pdf/2005.14165.pdf",
                Path("research/papers/language_models_few_shot.pdf"),
            ),
            DownloadItem(
                "https://arxiv.org/pdf/2303.18223.pdf",
                Path("research/papers/sparks_of_agi.pdf"),
            ),
        )

        self.archives: tuple[ArchiveItem, ...] = (
            ArchiveItem(
                url="https://mirrors.edge.kernel.org/pub/linux/docs/man-pages/man-pages-6.18.tar.xz",
                relative_path=Path("technical/linux/man-pages-6.18.tar.xz"),
                include_prefixes=("man-pages-6.18/man",),
            ),
            ArchiveItem(
                url="https://github.com/systemd/systemd/archive/refs/tags/v261.tar.gz",
                relative_path=Path("technical/linux/systemd-v261.tar.gz"),
                include_prefixes=("systemd-261/docs/",),
            ),
            ArchiveItem(
                url="https://github.com/torvalds/linux/archive/refs/tags/v6.9.tar.gz",
                relative_path=Path("technical/linux/linux-v6.9.tar.gz"),
                include_prefixes=(
                    "linux-6.9/Documentation/admin-guide/",
                    "linux-6.9/Documentation/userspace-api/",
                ),
            ),
            ArchiveItem(
                url="https://docs.python.org/3/archives/python-3.12.4-docs-text.tar.bz2",
                relative_path=Path("technical/python/python-3.12.4-docs-text.tar.bz2"),
                include_prefixes=("python-3.12.4-docs-text/",),
            ),
            ArchiveItem(
                url="https://github.com/huggingface/transformers/archive/refs/tags/v4.41.2.tar.gz",
                relative_path=Path("technical/hf/transformers-v4.41.2.tar.gz"),
                include_prefixes=("transformers-4.41.2/docs/source/en/",),
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
