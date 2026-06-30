"""Build the six tutorial notebooks for Project #7.

The notebooks are generated from structured templates so content stays
consistent and can be regenerated after edits.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip() + "\n")


def write_notebook(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    out_path = NOTEBOOKS_DIR / name
    nbf.write(nb, out_path)
    print(f"Wrote {out_path.relative_to(ROOT)}")


def build_nb01() -> None:
    cells = [
        md(
            """
            # 01 - Deployment Foundations and MLOps

            This chapter is zero-to-hero. You will learn **what deployment means**, why local success is not production success, and how MLOps turns experiments into reliable products.

            ## Learning Outcomes
            - Define deployment in AI context.
            - Explain development vs production environments.
            - Understand AI app lifecycle from idea to maintenance.
            - Compare traditional software deployment with AI deployment.
            """
        ),
        md(
            """
            ## Section 1: What Is Deployment?

            ### Definition
            Deployment is process of moving application from developer machine to environment where real users can access it.

            ### Theory
            Deployment couples **code**, **dependencies**, **runtime configuration**, and **infrastructure**.

            ### Motivation
            A model that works on your laptop has zero business value until users can call it safely, repeatedly, and quickly.

            ### Real-World Example
            A startup builds local sentiment model demo. After deployment, product and support teams use the app to triage feedback at scale.

            ### Visual Explanation
            ![Deployment Architecture](../outputs/figures/deployment-architecture.png)

            ### Best Practices
            - Freeze dependency versions.
            - Separate secrets from source code.
            - Validate same startup path locally and in cloud.

            ### Common Mistakes
            - "Works on my machine" assumptions.
            - Missing requirements file.
            - Hardcoded API keys.
            """
        ),
        code(
            """
            from pathlib import Path
            import platform
            import sys

            root = Path.cwd()
            if not (root / "pyproject.toml").exists() and (root.parent / "pyproject.toml").exists():
                root = root.parent

            print("Project root:", root)
            print("Python version:", sys.version.split()[0])
            print("Platform:", platform.platform())
            print("Has pyproject.toml:", (root / "pyproject.toml").exists())
            print("Has app.py:", (root / "app.py").exists())
            """
        ),
        md(
            """
            ## Section 2: Development Environment vs Production Environment

            ### Definition
            - **Development environment:** where engineers write and test code.
            - **Production environment:** where end users interact with deployed system.

            ### Theory
            Production adds constraints: limited CPU/memory, cold starts, security controls, uptime expectations, and external traffic.

            ### Motivation
            Debugging after deployment is expensive. Align environments early to reduce surprises.

            ### Real-World Example
            Local app has unlimited timeout. Cloud host times out at 60s; slow model call fails in production.

            ### Visual Explanation
            ![Cloud Workflow](../outputs/figures/cloud-workflow.png)

            ### Code Explanation
            Next cell compares environment variables and required deployment config.

            ### Best Practices
            - Keep a `.streamlit/secrets.toml.example` template.
            - Validate runtime with smoke tests before deploy.

            ### Common Mistakes
            - Assuming local environment variables exist in cloud.
            - Installing packages manually without lock file.
            """
        ),
        code(
            """
            import os

            required_env = ["HF_API_TOKEN", "OLLAMA_BASE_URL"]
            status = {k: ("set" if os.getenv(k) else "missing") for k in required_env}
            status
            """
        ),
        md(
            """
            ## Section 3: Introduction to MLOps

            ### Definition
            MLOps is discipline that combines ML, software engineering, and operations to deploy and maintain ML systems reliably.

            ### Theory
            Core stages: **model development -> packaging -> deployment -> monitoring -> maintenance**.

            ### Motivation
            AI systems drift. Data changes. Model quality degrades. MLOps creates repeatable process to detect and fix this.

            ### Real-World Example
            Fraud model performs well at launch but misses new fraud patterns 3 months later. Monitoring alerts trigger retraining pipeline.

            ### Visual Explanation
            ![Application Lifecycle](../outputs/figures/application-lifecycle.png)

            ### Best Practices
            - Track baselines and post-deployment metrics.
            - Version model + code + data assumptions together.
            - Add rollback plan before release.

            ### Common Mistakes
            - Treating model artifact as static forever.
            - No observability after launch.
            """
        ),
        code(
            """
            import pandas as pd

            comparison = pd.DataFrame(
                [
                    ["Primary artifact", "Compiled app binary", "Model + app + data assumptions"],
                    ["Failure mode", "Functional bug", "Functional + statistical drift"],
                    ["Validation", "Unit/integration tests", "Tests + offline metrics + monitoring"],
                    ["Release cadence", "Code driven", "Code + data driven"],
                ],
                columns=["Aspect", "Traditional Software", "AI Deployment"],
            )
            comparison
            """
        ),
        md(
            """
            ## Section 4: Deployment Flow Summary

            ### Definition
            End-to-end flow for this project:
            `Developer Machine -> GitHub Repo -> Streamlit Cloud -> Public App`

            ### Theory
            Git acts as source-of-truth. Cloud host rebuilds app from repository state.

            ### Motivation
            Reproducible deployment means anyone can recreate the same application behavior.

            ### Visual Explanation
            ![Git Workflow](../outputs/figures/git-workflow.png)

            ### Best Practices
            - Small commits.
            - CI checks before merge.
            - Pin dependencies.

            ### Common Mistakes
            - Pushing untested code.
            - Deploying directly from unstaged local files.
            """
        ),
    ]
    write_notebook("01_deployment_foundations_and_mlops.ipynb", cells)


def build_nb02() -> None:
    cells = [
        md(
            """
            # 02 - Git and GitHub for AI Deployment

            This chapter teaches Git and GitHub from scratch with deployment context.
            """
        ),
        md(
            """
            ## Section 1: Git Fundamentals

            ### Definition
            Git is distributed version control system that tracks file changes over time.

            ### Theory
            Every commit stores snapshot of project state and metadata (author, timestamp, message).

            ### Motivation
            Deployment platforms pull exact commit state, not loose local files.

            ### Real-World Examples
            - Roll back broken release to previous commit.
            - Compare two model-serving implementations.

            ### Visual Explanation
            ![Git Workflow](../outputs/figures/git-workflow.png)

            ### Best Practices
            - Use meaningful commit messages.
            - Commit frequently in small logical units.

            ### Common Mistakes
            - One huge commit after days of work.
            - Ambiguous messages like `update`.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess
            import tempfile

            demo_dir = Path(tempfile.mkdtemp(prefix="git_demo_"))
            print("Demo repo:", demo_dir)

            commands = [
                ["git", "init"],
                ["git", "config", "user.name", "demo-user"],
                ["git", "config", "user.email", "demo@example.com"],
            ]
            for cmd in commands:
                out = subprocess.run(cmd, cwd=demo_dir, capture_output=True, text=True, check=False)
                print("$", " ".join(cmd))
                print((out.stdout or out.stderr).strip())

            (demo_dir / "README.md").write_text("# Demo Repo\\n")
            subprocess.run(["git", "add", "README.md"], cwd=demo_dir, check=False)
            subprocess.run(["git", "commit", "-m", "feat: add README"], cwd=demo_dir, check=False)

            log = subprocess.run(["git", "log", "--oneline", "-n", "1"], cwd=demo_dir, capture_output=True, text=True, check=False)
            print("Latest commit:", log.stdout.strip())
            """
        ),
        md(
            """
            ## Section 2: Core Git Terms

            ### Definition and Motivation
            - **Repository:** project folder tracked by Git.
            - **Commit:** saved checkpoint.
            - **Branch:** independent line of development.
            - **Push:** upload commits to remote.
            - **Pull:** sync remote changes locally.
            - **Main branch:** default integration branch for releases.

            ### Theory
            Branching allows feature work without destabilizing deployment branch.

            ### Real-World Example
            Build feature in `feature/deploy-monitoring`, then merge to `main` only after tests pass.

            ### Code Explanation
            Next cell prints branch state from current project, if available.

            ### Best Practices
            - Protect `main` with CI checks.
            - Keep branch names task-scoped.

            ### Common Mistakes
            - Working directly on main for risky changes.
            - Force-pushing shared branches.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess

            root = Path.cwd()
            if not (root / ".git").exists() and (root.parent / ".git").exists():
                root = root.parent

            status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, check=False)
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, capture_output=True, text=True, check=False)

            print("Branch:", branch.stdout.strip() or "unavailable")
            print("Status output:\\n", status.stdout.strip() or "(clean or unavailable)")
            """
        ),
        md(
            """
            ## Section 3: GitHub Fundamentals

            ### Definition
            GitHub is remote hosting platform for Git repositories.

            ### Theory
            Deployment platforms integrate with GitHub webhooks and build from selected branch.

            ### Motivation
            No GitHub repo, no Streamlit Cloud auto-deploy workflow.

            ### Real-World Example
            Push commit to `main` -> Streamlit Cloud detects change -> rebuild starts automatically.

            ### Best Practices
            - Add clear README.
            - Use releases/tags for milestone versions.
            - Keep sensitive files out of repository.

            ### Common Mistakes
            - Committing `.streamlit/secrets.toml`.
            - Publicly exposing keys in notebooks.
            """
        ),
        code(
            """
            from pathlib import Path

            root = Path.cwd()
            if not (root / "README.md").exists() and (root.parent / "README.md").exists():
                root = root.parent

            required_files = ["README.md", "app.py", "pyproject.toml", ".gitignore"]
            report = {file: (root / file).exists() for file in required_files}
            report
            """
        ),
    ]
    write_notebook("02_git_and_github_for_ai_deployment.ipynb", cells)


def build_nb03() -> None:
    cells = [
        md(
            """
            # 03 - Build Deployment Candidate Application

            Build and validate a deployment-ready AI app with professional packaging and dependency management.
            """
        ),
        md(
            """
            ## Section 1: Deployment Candidate Definition

            ### Definition
            A deployment candidate is app version that passed local quality gates and is safe to release.

            ### Theory
            Candidate includes source code, tests, dependency lock, configuration templates, and operational docs.

            ### Motivation
            Deploying ad-hoc prototype code creates downtime and debugging chaos.

            ### Real-World Example
            Product team demo scheduled tomorrow. Candidate process prevents last-minute runtime failures.

            ### Visual Explanation
            ![Deployment Architecture](../outputs/figures/deployment-architecture.png)

            ### Best Practices
            - Keep app entrypoint explicit (`app.py`).
            - Use modular structure (`pages`, `components`, `utils`).
            - Add fallback logic for external inference APIs.

            ### Common Mistakes
            - Coupling UI and model code in one large script.
            - No fallback when API quota/rate-limit hits.
            """
        ),
        code(
            """
            from pathlib import Path

            root = Path.cwd()
            if not (root / "streamlit_app").exists() and (root.parent / "streamlit_app").exists():
                root = root.parent

            for path in [
                root / "app.py",
                root / "streamlit_app" / "app.py",
                root / "streamlit_app" / "pages",
                root / "streamlit_app" / "utils" / "models.py",
                root / "tests",
            ]:
                print(path.relative_to(root), "->", "OK" if path.exists() else "MISSING")
            """
        ),
        md(
            """
            ## Section 2: Model Choice Rationale

            ### Definition
            Model selection balances quality, latency, cost, and deployability.

            ### Theory
            This project uses three-tier inference strategy:
            1. Hugging Face hosted models for cloud runtime.
            2. Ollama local models for local experimentation.
            3. Rule-based fallback for guaranteed response.

            ### Motivation
            Free-tier hosting may not support large GPU models. Fallback keeps app available.

            ### Real-World Example
            If external API times out, app still returns output via fallback instead of error page.

            ### Visual Explanation
            ![Fallback Strategy](../outputs/figures/fallback-strategy.png)

            ### Best Practices
            - Expose inference method in UI for transparency.
            - Log latency and fallback usage.

            ### Common Mistakes
            - Silent fallback that hides degraded behavior from maintainers.
            """
        ),
        md(
            """
            ## Section 3: Dependency Management Deep Dive

            ### Definition
            - `requirements.txt`: plain list used by many hosts.
            - `pyproject.toml`: modern project metadata and dependency declaration.
            - `uv.lock`: exact resolved versions for reproducibility.

            ### Theory
            Use `pyproject.toml` + `uv.lock` as source of truth, then export `requirements.txt` for deployment target compatibility.

            ### Motivation
            Reproducible dependencies prevent surprise breakages during cloud build.

            ### Real-World Example
            Unpinned transitive package releases breaking change and deployment fails.

            ### Best Practices
            - Lock dependencies.
            - Keep runtime and dev tooling explicit.
            - Regenerate lock file after dependency changes.

            ### Common Mistakes
            - Editing requirements manually without syncing lock.
            - Mixing `pip`, `conda`, and `uv` in same project.
            """
        ),
        code(
            """
            from pathlib import Path
            import tomllib

            root = Path.cwd()
            if not (root / "pyproject.toml").exists() and (root.parent / "pyproject.toml").exists():
                root = root.parent

            pyproject = tomllib.loads((root / "pyproject.toml").read_text())
            deps = pyproject["project"].get("dependencies", [])
            print("Dependency count:", len(deps))
            print("First five dependencies:")
            for dep in deps[:5]:
                print("-", dep)

            print("uv.lock exists:", (root / "uv.lock").exists())
            print("requirements.txt exists:", (root / "requirements.txt").exists())
            """
        ),
        md(
            """
            ## Section 4: Local Testing Before Deployment

            ### Definition
            Local validation ensures app starts, inference works, and tests pass before cloud push.

            ### Theory
            Fail fast locally, not in production.

            ### Motivation
            Cloud debugging loops are slower and harder due remote logs and limited shell access.

            ### Practical Checklist
            1. Create venv and sync deps.
            2. Run test suite.
            3. Launch Streamlit app.
            4. Verify each user flow.
            5. Confirm secrets are externalized.

            ### Common Mistakes
            - Skipping tests before pushing to main.
            - Testing only one page and deploying all pages.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess

            root = Path.cwd()
            if not (root / "tests").exists() and (root.parent / "tests").exists():
                root = root.parent

            cmd = ["uv", "run", "pytest", "-q"]
            result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
            print("Command:", " ".join(cmd))
            print("Exit code:", result.returncode)
            print("Pytest summary:")
            lines = (result.stdout + "\\n" + result.stderr).splitlines()
            for line in lines[-6:]:
                print(line)
            """
        ),
    ]
    write_notebook("03_building_production_deployment_candidate.ipynb", cells)


def build_nb04() -> None:
    cells = [
        md(
            """
            # 04 - Real Streamlit Community Cloud Deployment

            This chapter walks through full deployment process: repository setup, cloud linking, configuration, verification, and evidence capture.
            """
        ),
        md(
            """
            ## Section 1: What Is Streamlit Community Cloud?

            ### Definition
            Streamlit Community Cloud is managed hosting platform for Streamlit applications with free tier.

            ### Theory
            Cloud runner clones repository, installs dependencies, starts Streamlit using configured entrypoint.

            ### Motivation
            Fast path from local prototype to public URL with minimal infrastructure setup.

            ### Benefits
            - Fast onboarding.
            - GitHub integration.
            - Built-in logs and secret management.

            ### Limitations
            - Resource constraints vs paid cloud.
            - Best for demos, prototypes, lightweight production.

            ### Visual Explanation
            ![Cloud Workflow](../outputs/figures/cloud-workflow.png)

            ### Best Practices
            - Keep startup lightweight.
            - Cache expensive resources.
            - Pin dependencies.

            ### Common Mistakes
            - Missing `requirements.txt`.
            - Wrong app entrypoint path.
            """
        ),
        code(
            """
            from pathlib import Path
            import subprocess

            root = Path.cwd()
            if not (root / "app.py").exists() and (root.parent / "app.py").exists():
                root = root.parent

            checks = {
                "app.py": (root / "app.py").exists(),
                "requirements.txt": (root / "requirements.txt").exists(),
                ".streamlit/config.toml": (root / ".streamlit" / "config.toml").exists(),
                ".streamlit/secrets.toml.example": (root / ".streamlit" / "secrets.toml.example").exists(),
            }
            checks

            gh_status = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True, check=False)
            print("gh auth status exit:", gh_status.returncode)
            print((gh_status.stdout + gh_status.stderr).strip()[:500])
            """
        ),
        md(
            """
            ## Section 2: Step-by-Step Deployment Process

            ### Step 1 - Repository Creation
            Create or use dedicated public repository for this project.

            ### Step 2 - Push Source
            Push branch with tested deployment candidate to GitHub.

            ### Step 3 - Link in Streamlit Cloud
            In Streamlit Cloud dashboard: `New app -> choose repo -> branch -> app path`.

            ### Step 4 - Configure Secrets
            Add `HF_API_TOKEN` and optional `OLLAMA_BASE_URL` in cloud secrets panel.

            ### Step 5 - Deploy and Verify
            Confirm app boots and each tab returns output.

            ### Step 6 - Capture Evidence
            Save deployment URL, build log snippet, and screenshots.

            **Example repository URL (replace with your own):**  
            `https://github.com/<your-user>/<your-repo>`

            If Streamlit Cloud blocks the flow with a sign-in challenge, finish login manually and continue deployment from the same repository.

            ### Code Explanation
            Next cell writes deployment evidence template JSON for reproducible reporting.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            from datetime import datetime, timezone
            import os

            root = Path.cwd()
            if not (root / "outputs").exists() and (root.parent / "outputs").exists():
                root = root.parent

            evidence = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "public_url": os.getenv("DEPLOYED_APP_URL", "NOT_SET"),
                "streamlit_build_status": os.getenv("STREAMLIT_BUILD_STATUS", "unknown"),
                "notes": "Populate env vars during/after real deployment run.",
            }

            out_dir = root / "outputs" / "deployment"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "deployment_evidence.json"
            out_file.write_text(json.dumps(evidence, indent=2))
            print("Wrote", out_file)
            print(json.dumps(evidence, indent=2))
            """
        ),
        md(
            """
            ## Section 3: Troubleshooting Quick Hits During Deployment

            ### Common Failures
            - `ModuleNotFoundError`: dependency missing in `requirements.txt`.
            - App timeout on startup: heavy model initialization without caching.
            - Build fails on wrong Python version.
            - Secret missing (`HF_API_TOKEN`) causing inference errors.

            ### Best Practices
            - Read build logs first.
            - Reproduce same error locally.
            - Fix root cause, then redeploy.

            ### Common Mistakes
            - Re-deploying repeatedly without log analysis.
            - Adding secrets to repository instead of cloud settings.
            """
        ),
    ]
    write_notebook("04_streamlit_cloud_real_deployment.ipynb", cells)


def build_nb05() -> None:
    cells = [
        md(
            """
            # 05 - Troubleshooting, Monitoring, and Optimization

            This chapter focuses on post-deployment reliability: diagnose failures, monitor health, and optimize startup and latency.
            """
        ),
        md(
            """
            ## Section 1: Troubleshooting Framework

            ### Definition
            Troubleshooting is systematic process to detect, isolate, and fix production issues.

            ### Theory
            Use layered diagnosis: infrastructure -> dependencies -> app startup -> inference runtime -> user input.

            ### Motivation
            Random debugging wastes time. Structured triage reduces MTTR (mean time to resolution).

            ### Real-World Example
            User reports blank page. Root cause: import error in deployed environment due missing pinned dependency.

            ### Best Practices
            - Keep reproducible runbook.
            - Capture failing input and stack trace.
            - Validate fix with regression test.

            ### Common Mistakes
            - Hotfixing symptoms instead of root cause.
            - No postmortem documentation.
            """
        ),
        code(
            """
            import pandas as pd

            failure_matrix = pd.DataFrame(
                [
                    ["Missing package", "ModuleNotFoundError", "Add package + pin version"],
                    ["Startup timeout", "App never reaches healthy state", "Cache resources and reduce cold-start work"],
                    ["HF token missing", "401 or fallback-only mode", "Set HF_API_TOKEN in secrets"],
                    ["Bad user input", "Validation error", "Improve input validation and UI messaging"],
                ],
                columns=["Failure", "Signal", "Fix Pattern"],
            )
            failure_matrix
            """
        ),
        md(
            """
            ## Section 2: Monitoring Fundamentals

            ### Definition
            Monitoring tracks app health over time using logs, errors, latency, and user feedback.

            ### Theory
            Minimum set: request latency, error rate, fallback rate, and startup time.

            ### Motivation
            Without monitoring, you discover outages only after user complaints.

            ### Real-World Example
            Latency suddenly doubles after dependency upgrade; monitoring catches regression quickly.

            ### Best Practices
            - Emit structured logs.
            - Surface metrics in UI/admin panel.
            - Track trend, not single datapoint.

            ### Common Mistakes
            - Monitoring only uptime, ignoring model quality and response latency.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import psutil
            import time

            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 ** 2)

            metrics = {
                "timestamp": time.time(),
                "memory_rss_mb": round(mem_mb, 2),
                "cpu_percent": process.cpu_percent(interval=0.1),
            }

            root = Path.cwd()
            if not (root / "outputs").exists() and (root.parent / "outputs").exists():
                root = root.parent

            out_dir = root / "outputs" / "metrics"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "runtime_snapshot.json"
            out_file.write_text(json.dumps(metrics, indent=2))
            print(metrics)
            print("saved to", out_file)
            """
        ),
        md(
            """
            ## Section 3: Production Optimization

            ### Definition
            Optimization improves latency, startup time, and memory efficiency while preserving correctness.

            ### Theory
            Key levers for this app:
            - `st.cache_resource` for expensive model clients.
            - `st.cache_data` for deterministic repeat computations.
            - lightweight fallback for degraded conditions.

            ### Motivation
            Free-tier hosts have strict resource budgets.

            ### Real-World Example
            Cold start drops from 11s to 4s after caching initialization.

            ### Best Practices
            - Measure before and after.
            - Optimize bottlenecks first.
            - Keep correctness tests around optimized path.

            ### Common Mistakes
            - Premature optimization without baseline.
            - Caching mutable objects unsafely.
            """
        ),
        code(
            """
            import time
            from functools import lru_cache

            def expensive_init(delay: float = 0.25) -> str:
                time.sleep(delay)
                return "resource-ready"

            @lru_cache(maxsize=1)
            def cached_expensive_init(delay: float = 0.25) -> str:
                time.sleep(delay)
                return "resource-ready"

            t0 = time.perf_counter()
            expensive_init()
            first_plain = time.perf_counter() - t0

            t1 = time.perf_counter()
            cached_expensive_init()
            first_cached = time.perf_counter() - t1

            t2 = time.perf_counter()
            cached_expensive_init()
            second_cached = time.perf_counter() - t2

            {
                "plain_seconds": round(first_plain, 4),
                "cached_first_seconds": round(first_cached, 4),
                "cached_repeat_seconds": round(second_cached, 4),
            }
            """
        ),
    ]
    write_notebook("05_troubleshooting_monitoring_and_optimization.ipynb", cells)


def build_nb06() -> None:
    cells = [
        md(
            """
            # 06 - Security, Platform Comparison, and Reusable Checklists

            Final chapter: secure your deployment, choose right platform, and use repeatable checklists for future projects.
            """
        ),
        md(
            """
            ## Section 1: Deployment Security Basics

            ### Definition
            Deployment security protects credentials, data, and infrastructure from accidental exposure and abuse.

            ### Theory
            Secrets should flow through environment variables or platform secret managers, never repository commits.

            ### Motivation
            One leaked token can expose inference usage, billing, and private assets.

            ### Real-World Example
            Developer accidentally commits API key; bot scans public repo and key is abused within minutes.

            ### Best Practices
            - Keep `.streamlit/secrets.toml` in `.gitignore`.
            - Add `secrets.toml.example` template.
            - Rotate compromised keys immediately.

            ### Common Mistakes
            - Hardcoding API keys in notebooks.
            - Printing full secrets to logs.
            """
        ),
        code(
            """
            from pathlib import Path

            root = Path.cwd()
            if not (root / ".gitignore").exists() and (root.parent / ".gitignore").exists():
                root = root.parent

            gitignore_text = (root / ".gitignore").read_text()
            checks = {
                "streamlit secrets ignored": ".streamlit/secrets.toml" in gitignore_text,
                "env ignored": ".env" in gitignore_text,
            }
            checks
            """
        ),
        md(
            """
            ## Section 2: Platform Comparison (Snapshot)

            ### Definition
            Platform comparison helps choose host based on cost, complexity, and scalability needs.

            ### Theory
            Beginner-friendly platforms reduce ops burden but may limit advanced scaling and custom runtime controls.

            ### Motivation
            Wrong platform choice creates future migration pain.

            ### Best Practices
            - Start with fastest path to validated user value.
            - Move to heavier infra only when workload needs it.

            ### Common Mistakes
            - Overengineering infra before product validation.
            - Ignoring platform quotas and cold-start behavior.
            """
        ),
        code(
            """
            import pandas as pd

            comparison = pd.DataFrame(
                [
                    ["Streamlit Cloud", "Free tier", "Very easy", "Best for demos and lightweight AI apps", "Low to medium"],
                    ["Hugging Face Spaces", "Free + paid hardware", "Easy", "Great for ML demos and model-centric apps", "Medium"],
                    ["Render", "Free + paid", "Medium", "Good general web deployment", "Medium"],
                    ["Railway", "Usage-based", "Easy to medium", "Fast shipping of web services", "Medium"],
                    ["Fly.io", "Usage-based", "Medium to hard", "Global edge deployment", "High"],
                    ["Docker on VPS/Cloud", "Variable", "Hard", "Max control for production", "Very high"],
                ],
                columns=["Platform", "Cost Model", "Ease of Use", "Strength", "Scalability"],
            )
            comparison
            """
        ),
        md(
            """
            ## Section 3: Reusable Deployment Checklist

            ### Definition
            Checklist is pre-flight and post-flight quality gate for reliable releases.

            ### Theory
            Standardization reduces human error across repeated deployments.

            ### Motivation
            Teams ship faster when release criteria are explicit.

            ### Best Practices
            - Keep checklist versioned in repo.
            - Use same list for every deployment.

            ### Common Mistakes
            - Treating checklist as optional.
            - Skipping monitoring and rollback steps.
            """
        ),
        code(
            """
            from pathlib import Path

            root = Path.cwd()
            if not (root / "docs").exists() and (root.parent / "docs").exists():
                root = root.parent

            checklist = root / "docs" / "deployment_checklist.md"
            print("Checklist exists:", checklist.exists())
            if checklist.exists():
                print("Path:", checklist)
                print("Preview:")
                print("\\n".join(checklist.read_text().splitlines()[:12]))
            """
        ),
        md(
            """
            ## Graduation Notes

            You now have complete deployment pipeline:
            - architecture understanding,
            - Git/GitHub workflow,
            - production packaging,
            - cloud deployment,
            - troubleshooting + monitoring,
            - security + platform tradeoffs.

            Next step is to iterate with real users and evolve from demo reliability to full production SLOs.
            """
        ),
    ]
    write_notebook("06_security_platform_comparison_and_ops_checklists.ipynb", cells)


def main() -> None:
    build_nb01()
    build_nb02()
    build_nb03()
    build_nb04()
    build_nb05()
    build_nb06()


if __name__ == "__main__":
    main()
