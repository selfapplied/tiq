#!/usr/bin/env python3
"""
TIQ - Timeline Interleaver & Quantizer
A reversible Git repository management tool for dirs↔branches superposition.
"""

import os
import sys
import subprocess
import hashlib
import zlib
import tempfile
import shutil
from pathlib import Path
import json
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class Verb(Enum):
    """TIQ operation verbs."""
    SUPERPOSE = "superpose"
    EXTRACT = "extract"
    MAP = "map"
    DIFF = "diff"
    CLASSIFY = "classify"
    EMIT = "emit"
    REFLECT = "reflect"
    REBALANCE = "rebalance"
    ROTATE = "rotate"
    PRUNE = "prune"


@dataclass
class TIQConfig:
    """Configuration for TIQ operations."""
    host: str = "."
    dirs_depth: int = 1
    ignore: Optional[Set[str]] = None
    include_non_git: bool = False
    prefix: Optional[str] = None
    events: Optional[Set[str]] = None  # e.g., {"content", "cadence", "size", "host-moved"}
    
    def __post_init__(self):
        if self.ignore is None:
            self.ignore = {".git", ".venv", "node_modules"}
        # Default event policy: materialize on content change
        if not self.events:
            self.events = {"content"}


@dataclass
class Passport:
    """Passport for a branch with cryptographic identity."""
    branch: str
    tag: str
    head_short: str
    date_short: str


class TIQError(Exception):
    """Base exception for TIQ operations."""
    pass


@dataclass
class MirrorStats:
    """In-memory mirror kernel statistics for a branch.

    Axis equilibrium corresponds to host_head == ghost_head (Fix(I)).
    """
    branch: str
    host_head: str
    ghost_head: str
    ahead: int  # commits in host only (left)
    behind: int  # commits in ghost only (right)
    energy: int
    equilibrium: bool
    fast_forward_possible: bool


class TIQ:
    """Timeline Interleaver & Quantizer - Core implementation."""
    
    def __init__(self, config: TIQConfig):
        self.config = config
        self.host_path = Path(config.host).resolve()
        # Energy threshold (bytes) for swaps; holds if exceeded
        try:
            self.energy_threshold = int(os.environ.get("TIQ_MAX_CHURN", "65536"))
        except Exception:
            self.energy_threshold = 65536
        # In-memory mirror statistics (rebuilt per run)
        self._mirror_stats: List[MirrorStats] = []
        
    def _run_git(self, args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        cmd = ["git"] + args
        cwd = cwd or self.host_path
        
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError:
            raise TIQError("Git not found. Please install Git.")
    
    def _is_git_repo(self, path: Path) -> bool:
        """Check if a directory is a Git repository."""
        return (path / ".git").exists()
    
    def _is_clean_repo(self, path: Path, *, ignore_untracked: bool = False) -> bool:
        """Check if a Git repository is clean (no uncommitted changes).
        If ignore_untracked is True, untracked files are ignored.
        """
        if not self._is_git_repo(path):
            return True
        status_args = ["status", "--porcelain"]
        if ignore_untracked:
            status_args.append("--untracked-files=no")
        returncode, stdout, stderr = self._run_git(status_args, cwd=path)
        if returncode != 0:
            raise TIQError(f"Failed to check git status in {path}: {stderr}")
        return len(stdout) == 0

    def _ensure_host_clean(self):
        """Ensure host repo exists and is clean enough (ignore untracked)."""
        if not self._is_git_repo(self.host_path):
            rc, _, err = self._run_git(["init"], cwd=self.host_path)
            if rc != 0:
                raise TIQError(f"Failed to initialize host repository: {err}")
            print("✓ initialized host repository")
        else:
            if not self._is_clean_repo(self.host_path, ignore_untracked=True):
                raise TIQError("Host repository is not clean. Please commit or stash changes.")
    
    def _sanitize_branch_name(self, name: str) -> str:
        """Sanitize directory name to valid Git branch name."""
        # Replace invalid characters with hyphens
        sanitized = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        # Remove leading/trailing hyphens and collapse multiple hyphens
        sanitized = "-".join(part for part in sanitized.split("-") if part)
        return sanitized or "unnamed"
    
    def _get_directories(self) -> List[Path]:
        """Get first-level subdirectories, excluding ignored ones."""
        if not self.host_path.exists():
            return []
            
        dirs = []
        for item in self.host_path.iterdir():
            if (item.is_dir() and 
                item.name not in (self.config.ignore or set()) and
                not item.name.startswith(".")):
                dirs.append(item)
        
        return sorted(dirs)

    def _get_directories_from(self, root: Path) -> List[Path]:
        """Get first-level subdirectories from an arbitrary root, excluding ignored."""
        if not root.exists():
            return []
        dirs: List[Path] = []
        for item in root.iterdir():
            if (item.is_dir() and
                item.name not in (self.config.ignore or set()) and
                not item.name.startswith(".")):
                dirs.append(item)
        return sorted(dirs)
    
    def _blake3_hash(self, data: bytes) -> str:
        """Compute BLAKE3 hash of data."""
        try:
            import blake3  # type: ignore[import-not-found]
            return blake3.blake3(data).hexdigest()[:8]
        except ImportError:
            # Fallback to SHA-256 if blake3 not available
            return hashlib.sha256(data).hexdigest()[:8]
    
    def _crc16_hash(self, data: bytes) -> str:
        """Compute CRC16 hash of data."""
        return f"{zlib.crc32(data) & 0xFFFF:04x}"

    def _add_remote_once(self, remote_name: str, remote_url: str):
        """Add remote if it doesn't exist."""
        rc, _, _ = self._run_git(["remote", "get-url", remote_name])
        if rc == 0:
            return
        rc2, _, err = self._run_git(["remote", "add", remote_name, remote_url])
        if rc2 != 0:
            raise TIQError(f"Failed to add remote {remote_name}: {err}")
        print(f"✓ added remote {remote_name}")

    def _fetch_remote(self, remote_name: str):
        """Fetch all branches and tags from the given remote."""
        rc, _, err = self._run_git([
            "fetch",
            remote_name,
            "+refs/heads/*:refs/remotes/" + remote_name + "/*",
            "--tags",
        ])
        if rc != 0:
            raise TIQError(f"Failed to fetch {remote_name}: {err}")
        print(f"↷ fetched {remote_name}")

    def _generate_passport(self, branch: str, tree_hash: str, meta: str, ref_hint: Optional[str] = None, pre_head: Optional[str] = None, pre_date: Optional[str] = None) -> Passport:
        """Generate passport for a branch with branch-scoped head/date.
        Prefers pre_head/pre_date if provided. Otherwise, resolves via ref_hint/local refs.
        """
        blake8 = self._blake3_hash(tree_hash.encode())
        crc = self._crc16_hash(meta.encode())
        tag = f"§{blake8}:{crc}"

        # Use precomputed head/date if supplied
        head_short = pre_head or ""
        date_short = pre_date or ""

        # Resolve head deterministically if not provided
        if not head_short:
            if ref_hint:
                rc_h, head_from_hint, _ = self._run_git(["rev-parse", "--short", ref_hint])
                if rc_h == 0 and head_from_hint:
                    head_short = head_from_hint
                if not head_short:
                    rc_hq, head_from_qhint, _ = self._run_git(["rev-parse", "--short", f"refs/remotes/{ref_hint}"])
                    if rc_hq == 0 and head_from_qhint:
                        head_short = head_from_qhint
        if not head_short:
            rc_show, show_out, _ = self._run_git(["show-ref", "--verify", f"refs/heads/{branch}"])
            if rc_show == 0 and show_out:
                full_hash = show_out.split()[0]
                rc_sh, head_short_tmp, _ = self._run_git(["rev-parse", "--short", full_hash])
                if rc_sh == 0 and head_short_tmp:
                    head_short = head_short_tmp
        if not head_short:
            rc1, head_short2, _ = self._run_git(["rev-parse", "--short", branch])
            head_short = head_short2 if rc1 == 0 and head_short2 else head_short
        if not head_short:
            rc1c, head_short3, _ = self._run_git(["rev-parse", "--short", "HEAD"])
            head_short = head_short3 if rc1c == 0 and head_short3 else "unknown"

        # Resolve date if not provided
        if not date_short:
            if ref_hint:
                rc_dh, date_from_hint, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short", ref_hint])
                if rc_dh == 0 and date_from_hint:
                    date_short = date_from_hint
                if not date_short:
                    rc_dhq, date_from_qhint, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short", f"refs/remotes/{ref_hint}"])
                    if rc_dhq == 0 and date_from_qhint:
                        date_short = date_from_qhint
        if not date_short:
            rc2, date_b, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short", branch])
            date_short = date_b if rc2 == 0 and date_b else date_short
        if not date_short:
            rc2b, date_head, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short", "HEAD"])
            date_short = date_head if rc2b == 0 and date_head else "unknown"

        return Passport(branch=branch, tag=tag, head_short=head_short, date_short=date_short)

    def _get_remote_head_or_first(self, remote_name: str) -> str:
        """Select a remote-qualified branch deterministically.
        Prefer refs/remotes/<remote_name>/main or /master; else first alphabetical.
        """
        candidates: List[str] = []
        rc, refs, _ = self._run_git(["for-each-ref", "--format=%(refname:short)", f"refs/remotes/{remote_name}"])
        if rc == 0 and refs:
            for ref in refs.split("\n"):
                ref = ref.strip()
                if ref and ref.startswith(f"{remote_name}/"):
                    candidates.append(ref)
        if not candidates:
            raise TIQError(f"No branches found in remote {remote_name}")
        # Prefer main/master
        for preferred in ("main", "master"):
            target = f"{remote_name}/{preferred}"
            if target in candidates:
                return target
        # Deterministic fallback
        return sorted(candidates)[0]

    # --- ref helpers (no WT writes) ---------------------------------------
    def _resolve_hash(self, ref: str, *, cwd: Optional[Path] = None) -> Optional[str]:
        rc, out, _ = self._run_git(["rev-parse", ref], cwd=cwd or self.host_path)
        return out if rc == 0 and out else None

    def _update_ref(self, ref: str, new_hash: str, old_hash: Optional[str] = None) -> None:
        args = ["update-ref", ref, new_hash]
        if old_hash:
            args.append(old_hash)
        self._run_git(args)

    def _compute_energy(self, old_hash: Optional[str], new_hash: str) -> int:
        if not old_hash or old_hash == new_hash:
            return 0
        rc, out, _ = self._run_git([
            "diff-tree", "-r", "-M90%", "-C90%", "--numstat", old_hash, new_hash
        ])
        if rc != 0 or not out:
            return 0
        energy = 0
        for line in out.splitlines():
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    add = 0 if parts[0] == '-' else int(parts[0])
                    rem = 0 if parts[1] == '-' else int(parts[1])
                    energy += max(0, add) + max(0, rem)
                except Exception:
                    continue
        return int(energy)

    def _branch_exists(self, branch_name: str) -> bool:
        """Return True if local branch exists."""
        rc, _, _ = self._run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"])
        return rc == 0

    def _ahead_behind(self, left: Optional[str], right: Optional[str]) -> Tuple[int, int]:
        """Return (ahead, behind) commit counts using symmetric difference.
        left: host, right: ghost. If either is missing, return (0, 0).
        """
        if not left or not right:
            return 0, 0
        rc, out, _ = self._run_git(
            ["rev-list", "--left-right", "--count", f"{left}...{right}"])
        if rc != 0 or not out:
            return 0, 0
        parts = out.strip().split()
        if len(parts) != 2:
            return 0, 0
        try:
            ahead = int(parts[0])
            behind = int(parts[1])
        except Exception:
            ahead, behind = 0, 0
        return ahead, behind

    def _checkout_branch(self, branch_name: str):
        """Deprecated: no WT writes; keep for compatibility if needed."""
        rc, _, stderr = self._run_git(["symbolic-ref", "HEAD", f"refs/heads/{branch_name}"])
        if rc != 0:
            raise TIQError(f"Failed to set HEAD to {branch_name}: {stderr}")

    def _create_branch(self, branch_name: str, head: str):
        """Create branch ref pointing to head (no WT writes)."""
        if self._branch_exists(branch_name):
            print(f"✓ branch {branch_name} already exists; skipping create")
            return
        self._update_ref(f"refs/heads/{branch_name}", head)
        print(f"✓ created branch {branch_name}")
    
    def _switch_orphan(self, branch_name: str):
        raise TIQError("Snapshot mode requires WT writes, which are disabled by invariants")
    
    def _stage_snapshot(self, source_dir: Path, prefix: Optional[str] = None):
        raise TIQError("Staging snapshots is disabled (no WT writes)")
        
        # Copy files with optional prefix
        if prefix:
            dest_path = self.host_path / prefix
            dest_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, dest_path / source_dir.name, dirs_exist_ok=True)
        else:
            for item in source_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, self.host_path)
                elif item.is_dir():
                    shutil.copytree(item, self.host_path / item.name, dirs_exist_ok=True)
        
        # Stage all files
        returncode, _, stderr = self._run_git(["add", "."])
        if returncode != 0:
            raise TIQError(f"Failed to stage snapshot: {stderr}")
        print(f"✓ staged snapshot of {source_dir}")
    
    def _commit_snapshot(self, message: str):
        raise TIQError("Committing snapshots is disabled (no WT writes)")
    
    def _with_sandbox(self, fn):
        """Deprecated: ghost-refs flow runs in the real host without WT writes."""
        source_root = self.host_path
        return fn(source_root)

    def _promote_sandbox_to_host(self, sandbox: Path, host: Path):
        """Deprecated in ghost-refs flow."""
        return

    def _should_materialize(self, sandbox: Path, host: Path) -> bool:
        """Evaluate -e/--event predicates. If none specified, never materialize."""
        events = self.config.events or set()
        if not events:
            return False
        decisions = []
        # content: any branch tag differs between sandbox and host
        if "content" in events:
            # Compare branch tips by commit id
            rc_s, s_branches, _ = self._run_git(["for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads"], cwd=sandbox)
            rc_h, h_branches, _ = self._run_git(["for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads"], cwd=host)
            sandbox_map = {}
            host_map = {}
            if rc_s == 0 and s_branches:
                for line in s_branches.split("\n"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        sandbox_map[parts[0]] = parts[1]
            if rc_h == 0 and h_branches:
                for line in h_branches.split("\n"):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        host_map[parts[0]] = parts[1]
            diff_exists = any(sandbox_map.get(b) != host_map.get(b) for b in sandbox_map.keys())
            decisions.append(diff_exists)
        # cadence: placeholder (always true for now); can be time-based later
        if "cadence" in events:
            decisions.append(True)
        # size: sandbox .git objects > threshold (placeholder threshold)
        if "size" in events:
            try:
                total = 0
                for root, _dirs, files in os.walk(sandbox / ".git"):
                    for f in files:
                        try:
                            total += (Path(root) / f).stat().st_size
                        except Exception:
                            pass
                decisions.append(total > 0)
            except Exception:
                decisions.append(False)
        # host-moved: host refs changed since last run (we can't persist; treat as true if host has any refs)
        if "host-moved" in events:
            rc_h2, h2, _ = self._run_git(["for-each-ref", "refs/heads", "--count=1"], cwd=host)
            decisions.append(rc_h2 == 0)
        # AND all specified predicates
        return all(decisions) if decisions else False

    def superpose(self) -> List[Passport]:
        """Superpose directories as branches using ghost refs and energy gating."""
        print("↷ syncing directories as ghost refs...")

        def _core(source_root: Path) -> List[Passport]:
            passports: List[Passport] = []
            # reset mirror stats per run
            self._mirror_stats = []
            for dir_path in self._get_directories_from(source_root):
                branch_name = self._sanitize_branch_name(dir_path.name)
                ref_hint = None
                pre_head = None
                pre_date = None
                glyph = "↷"
                reason = ""

                if self._is_git_repo(dir_path):
                    if not self._is_clean_repo(dir_path):
                        raise TIQError(f"Repository {dir_path} is not clean")
                    remote_name = f"tiq/{branch_name}"
                    # Use file:// URL for robustness
                    child_url = f"file://{dir_path}"
                    self._add_remote_once(remote_name, child_url)
                    self._fetch_remote(remote_name)
                    target_ref = self._get_remote_head_or_first(remote_name)
                    # Resolve target full hash
                    full_hash = self._resolve_hash(target_ref)
                    if not full_hash:
                        raise TIQError(f"Failed to resolve target for {branch_name}")
                    # Update ghost ref to target OID
                    ghost_ref = f"refs/tiq/ghost/{branch_name}"
                    self._update_ref(ghost_ref, full_hash)
                    # Decide swap
                    head_ref = f"refs/heads/{branch_name}"
                    old_hash = self._resolve_hash(head_ref)
                    energy = self._compute_energy(old_hash, full_hash)
                    if energy <= self.energy_threshold:
                        if not old_hash:
                            self._update_ref(head_ref, full_hash)
                            glyph = "✓"
                        else:
                            rc_ff, _, _ = self._run_git(["merge-base", "--is-ancestor", old_hash, full_hash])
                            if rc_ff == 0:
                                self._update_ref(head_ref, full_hash, old_hash)
                                glyph = "✓"
                            else:
                                print(f"✖ blocked by host divergence: {branch_name} host={old_hash[:8]} ghost={full_hash[:8]}", file=sys.stderr)
                                glyph = "✖"
                                reason = "div"
                    else:
                        glyph = "⧗"
                        reason = "ener"
                    # Fill passport hints from target
                    rc_hs, pre_head_val, _ = self._run_git(["rev-parse", "--short", full_hash])
                    pre_head = pre_head_val if rc_hs == 0 and pre_head_val else None
                    rc_ds, pre_date_val, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short", full_hash])
                    pre_date = pre_date_val if rc_ds == 0 and pre_date_val else None
                    ref_hint = full_hash
                    # Compute mirror stats
                    host_hash = self._resolve_hash(head_ref) or ""
                    ghost_hash = full_hash or ""
                    ahead, behind = self._ahead_behind(host_hash, ghost_hash)
                    eq = bool(host_hash) and host_hash == ghost_hash
                    rc_ff2, _, _ = self._run_git(
                        ["merge-base", "--is-ancestor", host_hash or "HEAD", ghost_hash]) if host_hash and ghost_hash else (1, "", "")
                    fast_fwd = rc_ff2 == 0
                    self._mirror_stats.append(MirrorStats(
                        branch=branch_name,
                        host_head=(host_hash[:8] if host_hash else ""),
                        ghost_head=(ghost_hash[:8] if ghost_hash else ""),
                        ahead=ahead,
                        behind=behind,
                        energy=energy,
                        equilibrium=eq,
                        fast_forward_possible=fast_fwd,
                    ))
                elif self.config.include_non_git:
                    # No WT writes allowed; hold
                    glyph = "⧗"
                    reason = "plain"
                # Generate passport
                passports.append(self._generate_passport(branch_name, "tree_hash", "meta", ref_hint=ref_hint, pre_head=pre_head, pre_date=pre_date))
                # Emit table row immediately
                head_short = passports[-1].head_short
                print(f"{branch_name} | {glyph} | {head_short} | {passports[-1].tag} | {reason}")
            return passports

        # Ensure host is clean before operations (no uncommitted changes)
        self._ensure_host_clean()
        # Run in-place with ghost refs
        return self._with_sandbox(_core)
    
    def extract(self, branch: str, target: str):
        """Extract branch to standalone repository."""
        print(f"⧗ extracting branch {branch} to {target}")
        
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize target repository
        returncode, _, stderr = self._run_git(["init"], cwd=target_path)
        if returncode != 0:
            raise TIQError(f"Failed to initialize target repository: {stderr}")
        
        # Pull branch from host
        returncode, _, stderr = self._run_git(["pull", str(self.host_path), branch], cwd=target_path)
        if returncode != 0:
            raise TIQError(f"Failed to pull branch {branch}: {stderr}")
        
        print(f"✓ extracted {branch} → {target}")
    
    def map_branches(self) -> List[Passport]:
        """Map all branches with their passports."""
        print("⧗ mapping branches...")
        
        passports = []
        returncode, branches, _ = self._run_git(["branch", "--format=%(refname:short)"])
        
        if returncode == 0 and branches:
            for branch in branches.split("\n"):
                if branch.strip():
                    passport = self._generate_passport(branch.strip(), "tree_hash", "meta")
                    passports.append(passport)
        
        return passports
    
    def diff_branches(self, left: str, right: str):
        """Compare histories between branches."""
        print(f"⧗ comparing {left}..{right}")
        
        returncode, output, stderr = self._run_git(["log", "--oneline", f"{left}..{right}"])
        if returncode != 0:
            raise TIQError(f"Failed to diff branches: {stderr}")
        
        print(output)
    
    # --- mirror kernel / classifier ---------------------------------------
    def classify(self) -> List[MirrorStats]:
        """Classify branches w.r.t. ghost reflections (in-memory tracker).

        Returns the latest mirror stats; also prints a compact table.
        """
        self._ensure_host_clean()
        stats: List[MirrorStats] = []
        # Rebuild from current refs to avoid stale state
        rc, branches_raw, _ = self._run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads"])
        branches = [b.strip() for b in (branches_raw.split(
            "\n") if rc == 0 and branches_raw else []) if b.strip()]
        for branch in sorted(branches):
            head_ref = f"refs/heads/{branch}"
            ghost_ref = f"refs/tiq/ghost/{branch}"
            host_hash = self._resolve_hash(head_ref)
            ghost_hash = self._resolve_hash(ghost_ref)
            ahead, behind = self._ahead_behind(host_hash, ghost_hash)
            energy = self._compute_energy(
                host_hash, ghost_hash or "") if ghost_hash else 0
            eq = bool(host_hash and ghost_hash and host_hash == ghost_hash)
            rc_ff, _, _ = self._run_git(
                ["merge-base", "--is-ancestor", host_hash or "HEAD", ghost_hash]) if host_hash and ghost_hash else (1, "", "")
            fast_fwd = rc_ff == 0
            stats.append(MirrorStats(
                branch=branch,
                host_head=(host_hash[:8] if host_hash else ""),
                ghost_head=(ghost_hash[:8] if ghost_hash else ""),
                ahead=ahead,
                behind=behind,
                energy=energy,
                equilibrium=eq,
                fast_forward_possible=fast_fwd,
            ))
        # Store and print
        self._mirror_stats = stats
        if stats:
            print("\nMirror Kernel (axis equilibrium = host==ghost):")
            print("branch | host | ghost | ahead | behind | energy | eq | ff")
            print("-" * 70)
            for s in stats:
                print(f"{s.branch} | {s.host_head} | {s.ghost_head} | {s.ahead} | {s.behind} | {s.energy} | {int(s.equilibrium)} | {int(s.fast_forward_possible)}")
        else:
            print("No branches to classify.")
        return stats

    def emit_metadata(self, out_path: Optional[str] = None) -> Path:
        """Emit a bash script that verifies invariants then fast-forwards heads.

        Default path: .git/tiq/rebalance.sh. No working tree writes.
        """
        # Build fresh stats (for branch enumeration only)
        _ = self.classify()
        rc, branches_raw, _ = self._run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads"])
        branches = [b.strip() for b in (branches_raw.split(
            "\n") if rc == 0 and branches_raw else []) if b.strip()]
        script_lines: List[str] = []
        script_lines.append("#!/usr/bin/env bash")
        script_lines.append("set -euo pipefail")
        script_lines.append("")
        script_lines.append(
            "# invariant: host repo is clean (ignore untracked)")
        script_lines.append(
            "if [[ -n \"$(git status --porcelain --untracked-files=no)\" ]]; then")
        script_lines.append(
            "  echo '✖ host repository is not clean' >&2; exit 1")
        script_lines.append("fi")
        script_lines.append("")
        for branch in sorted(branches):
            head_ref = f"refs/heads/{branch}"
            ghost_ref = f"refs/tiq/ghost/{branch}"
            script_lines.append(
                f"# {branch}: fast-forward head to ghost if ancestor holds")
            script_lines.append(
                f"HOST=$(git rev-parse {head_ref} 2>/dev/null || echo '')")
            script_lines.append(
                f"GHOST=$(git rev-parse {ghost_ref} 2>/dev/null || echo '')")
            script_lines.append("if [[ -n \"$HOST\" && -n \"$GHOST\" ]]; then")
            script_lines.append(
                "  if git merge-base --is-ancestor \"$HOST\" \"$GHOST\"; then")
            script_lines.append(
                f"    git update-ref {head_ref} \"$GHOST\" \"$HOST\"")
            script_lines.append("  fi")
            script_lines.append("fi")
            script_lines.append("")
        # Write script file
        if out_path:
            target = Path(out_path)
        else:
            git_dir = self.host_path / ".git"
            if not git_dir.exists():
                raise TIQError("Host is not a Git repository (missing .git)")
            target = git_dir / "tiq" / "rebalance.sh"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(script_lines))
        print(f"✓ emitted rebalance script → {target}")
        return target

    def reflect_metadata(self, mode: str = "notes", fmt: str = "json") -> None:
        """Persist mirror stats into Git metadata.

        mode=notes: attach notes under refs/notes/tiq for each branch head.
        mode=config: write stats under git config keys tiq.branch.<name>.*
        mode=both: do both of the above.
        """
        mode_lc = mode.lower()
        if mode_lc not in ("notes", "config", "both"):
            raise TIQError("mode must be one of: notes, config, both")
        if fmt.lower() != "ce1":
            raise TIQError("Only ce1 format is supported for reflect")
        # Ensure repository present and fresh stats
        self._ensure_host_clean()
        stats = self.classify()
        # For each branch, resolve full head hash and attach a note
        for s in stats:
            head_ref = f"refs/heads/{s.branch}"
            full_hash = self._resolve_hash(head_ref)
            if not full_hash:
                continue
            # ce1 compact block only
            eq = 1 if s.equilibrium else 0
            ff = 1 if s.fast_forward_possible else 0
            message = (
                f"CE1{{lens=MirrorKernel|branch={s.branch}|host={s.host_head}|ghost={s.ghost_head}|"
                f"ahead={s.ahead}|behind={s.behind}|energy={s.energy}|eq={eq}|ff={ff}}}"
            )
            # notes
            if mode_lc in ("notes", "both"):
                rc, _, err = self._run_git([
                    "notes", "--ref=refs/notes/tiq", "add", "-f", "-m", message, full_hash
                ])
                if rc != 0:
                    raise TIQError(
                        f"Failed to write note for {s.branch}: {err}")
                print(f"✓ reflected {s.branch} → notes@{full_hash[:8]}")
            # config keys
            if mode_lc in ("config", "both"):
                base = f"tiq.branch.{s.branch}"
                kvs = {
                    f"{base}.host": s.host_head,
                    f"{base}.ghost": s.ghost_head,
                    f"{base}.ahead": str(s.ahead),
                    f"{base}.behind": str(s.behind),
                    f"{base}.energy": str(s.energy),
                    f"{base}.eq": "1" if s.equilibrium else "0",
                    f"{base}.ff": "1" if s.fast_forward_possible else "0",
                    f"{base}.head": full_hash,
                }
                for key, val in kvs.items():
                    rc, _, err = self._run_git(
                        ["config", "--local", "--replace-all", key, val])
                    if rc != 0:
                        raise TIQError(f"Failed to set {key}: {err}")
                print(
                    f"✓ reflected {s.branch} → config tiq.branch.{s.branch}.*")

    def rebalance(self) -> List[str]:
        """Fast-forward branches to their ghost refs when safe.

        Only moves refs/heads/<branch> if host is ancestor of ghost. Ignores energy gating.
        Returns list of updated branch names.
        """
        self._ensure_host_clean()
        updated: List[str] = []
        # Recompute current branches
        rc, branches_raw, _ = self._run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads"])
        branches = [b.strip() for b in (branches_raw.split(
            "\n") if rc == 0 and branches_raw else []) if b.strip()]
        for branch in sorted(branches):
            head_ref = f"refs/heads/{branch}"
            ghost_ref = f"refs/tiq/ghost/{branch}"
            host_hash = self._resolve_hash(head_ref)
            ghost_hash = self._resolve_hash(ghost_ref)
            if not host_hash or not ghost_hash:
                continue
            # Only fast-forward moves
            rc_ff, _, _ = self._run_git(
                ["merge-base", "--is-ancestor", host_hash, ghost_hash])
            if rc_ff != 0:
                continue
            if host_hash == ghost_hash:
                continue
            self._update_ref(head_ref, ghost_hash, host_hash)
            print(f"✓ rebalanced {branch}: {host_hash[:8]} → {ghost_hash[:8]}")
            updated.append(branch)
        if not updated:
            print("No branches eligible for fast-forward rebalancing.")
        return updated

    def _build_rebalance_script_lines(self) -> List[str]:
        """Construct the rebalance script lines (FF-only with invariant checks)."""
        _ = self.classify()
        rc, branches_raw, _ = self._run_git(
            ["for-each-ref", "--format=%(refname:short)", "refs/heads"])
        branches = [b.strip() for b in (branches_raw.split(
            "\n") if rc == 0 and branches_raw else []) if b.strip()]
        script_lines: List[str] = []
        script_lines.append("#!/usr/bin/env bash")
        script_lines.append("set -euo pipefail")
        script_lines.append("")
        script_lines.append(
            "# invariant: host repo is clean (ignore untracked)")
        script_lines.append(
            "if [[ -n \"$(git status --porcelain --untracked-files=no)\" ]]; then")
        script_lines.append(
            "  echo '✖ host repository is not clean' >&2; exit 1")
        script_lines.append("fi")
        script_lines.append("")
        for branch in sorted(branches):
            head_ref = f"refs/heads/{branch}"
            ghost_ref = f"refs/tiq/ghost/{branch}"
            script_lines.append(
                f"# {branch}: fast-forward head to ghost if ancestor holds")
            script_lines.append(
                f"HOST=$(git rev-parse {head_ref} 2>/dev/null || echo '')")
            script_lines.append(
                f"GHOST=$(git rev-parse {ghost_ref} 2>/dev/null || echo '')")
            script_lines.append("if [[ -n \"$HOST\" && -n \"$GHOST\" ]]; then")
            script_lines.append(
                "  if git merge-base --is-ancestor \"$HOST\" \"$GHOST\"; then")
            script_lines.append(
                f"    git update-ref {head_ref} \"$GHOST\" \"$HOST\"")
            script_lines.append("  fi")
            script_lines.append("fi")
            script_lines.append("")
        return script_lines

    def rotate(self, scale: str = "notes", state_branch: str = "tiq/state") -> None:
        """Rotate by scale: notes → config → files (state branch).

        - notes: write CE1 notes per head under refs/notes/tiq
        - metadata: write config keys tiq.branch.*
        - files: materialize CE1 + rebalance.sh into branch state_branch via temp worktree
        """
        mode = scale.lower()
        if mode == "notes":
            self.reflect_metadata(mode="notes", fmt="ce1")
            return
        if mode == "metadata":
            self.reflect_metadata(mode="config", fmt="ce1")
            return
        if mode != "files":
            raise TIQError("scale must be one of: notes, metadata, files")

        # Build contents
        stats = self.classify()
        ce1_lines: List[str] = []
        for s in stats:
            eq = 1 if s.equilibrium else 0
            ff = 1 if s.fast_forward_possible else 0
            ce1_lines.append(
                f"CE1{{lens=MirrorKernel|branch={s.branch}|host={s.host_head}|ghost={s.ghost_head}|ahead={s.ahead}|behind={s.behind}|energy={s.energy}|eq={eq}|ff={ff}}}"
            )
        script_lines = self._build_rebalance_script_lines()

        # Create temporary worktree and commit files to state_branch
        tmpdir = Path(tempfile.mkdtemp(prefix="tiq_state_"))
        try:
            rc, _, _ = self._run_git(
                ["worktree", "add", "-f", "-B", state_branch, str(tmpdir)])
            if rc != 0:
                raise TIQError("Failed to add worktree for state branch")
            tiq_dir = tmpdir / ".tiq"
            tiq_dir.mkdir(parents=True, exist_ok=True)
            (tiq_dir / "mirror.ce1").write_text("\n".join(ce1_lines) + "\n")
            (tiq_dir / "rebalance.sh").write_text("\n".join(script_lines) + "\n")
            try:
                os.chmod(tiq_dir / "rebalance.sh", 0o755)
            except Exception:
                pass
            rc, _, err = self._run_git(["add", "."], cwd=tmpdir)
            if rc != 0:
                raise TIQError(f"Failed to add files in state worktree: {err}")
            rc, _, _ = self._run_git(
                ["commit", "-m", "tiq: rotate(files) update state"], cwd=tmpdir)
        finally:
            try:
                self._run_git(["worktree", "remove", "--force", str(tmpdir)])
            except Exception:
                pass

    # --- prune subset branches --------------------------------------------
    def _list_heads(self) -> List[Tuple[str, str]]:
        rc, out, _ = self._run_git(
            ["for-each-ref", "--format=%(refname:short) %(objectname)", "refs/heads"])
        heads: List[Tuple[str, str]] = []
        if rc == 0 and out:
            for line in out.split("\n"):
                parts = line.strip().split()
                if len(parts) == 2:
                    heads.append((parts[0], parts[1]))
        return heads

    def find_subset_pairs(self) -> List[Tuple[str, str]]:
        """Return pairs (subset_branch, superset_branch) where subset head is ancestor of superset head.

        If heads are equal, deterministically choose the lexicographically larger name as subset.
        """
        heads = self._list_heads()
        name_to_hash = {n: h for n, h in heads}
        branches = sorted(name_to_hash.keys())
        pairs: List[Tuple[str, str]] = []
        for i, a in enumerate(branches):
            ha = name_to_hash[a]
            for b in branches:
                if a == b:
                    continue
                hb = name_to_hash[b]
                if ha == hb:
                    # identical; remove lexicographically larger as subset
                    subset, superset = (b, a) if b > a else (a, b)
                    if subset == a:
                        pairs.append((a, b))
                    continue
                # a ⊆ b if a is ancestor of b
                rc, _, _ = self._run_git(
                    ["merge-base", "--is-ancestor", ha, hb])
                if rc == 0:
                    pairs.append((a, b))
                    break
        return pairs

    def prune(self, apply: bool = False, keep: Optional[Set[str]] = None) -> List[str]:
        """Prune branches whose histories are complete subsets of others.

        Prints plan and, if apply=True, deletes refs/heads/<branch>, corresponding
        refs/tiq/ghost/<branch>, and clears tiq.branch.<branch>.* config keys.
        Returns list of pruned branch names.
        """
        self._ensure_host_clean()
        keep = keep or set()
        pairs = self.find_subset_pairs()
        # choose unique subsets to prune (prefer first detected superset)
        to_prune: Dict[str, str] = {}
        for subset, superset in pairs:
            if subset in keep:
                continue
            if subset not in to_prune:
                to_prune[subset] = superset
        if not to_prune:
            print("No subset branches detected.")
            return []
        for subset, superset in sorted(to_prune.items()):
            print(f"⧗ subset: {subset} ⊆ {superset}")
        pruned: List[str] = []
        if apply:
            for subset, _superset in to_prune.items():
                head_ref = f"refs/heads/{subset}"
                ghost_ref = f"refs/tiq/ghost/{subset}"
                # delete refs if exist
                self._run_git(["update-ref", "-d", head_ref])
                self._run_git(["update-ref", "-d", ghost_ref])
                # remove config keys
                base = f"tiq.branch.{subset}"
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.host"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.ghost"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.ahead"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.behind"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.energy"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.eq"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.ff"])
                self._run_git(
                    ["config", "--local", "--unset-all", f"{base}.head"])
                print(f"✓ pruned {subset}")
                pruned.append(subset)
        return pruned

    def print_table(self, passports: List[Passport]):
        """Print passport table."""
        if not passports:
            print("No branches found.")
            return
        
        print("\nBranch Map:")
        print("dir | type | branch | head | §")
        print("-" * 50)
        
        for passport in passports:
            print(f"{passport.branch} | git | {passport.branch} | {passport.head_short} | {passport.tag}")


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TIQ - Timeline Interleaver & Quantizer")
    parser.add_argument("verb", choices=[v.value for v in Verb], help="TIQ operation verb")
    parser.add_argument("--host", default=".", help="Host repository path")
    parser.add_argument("--include-non-git", action="store_true", help="Include non-Git directories")
    parser.add_argument("--prefix", help="Prefix for snapshot operations")
    parser.add_argument("--branch", help="Branch name for extract operation")
    parser.add_argument("--target", help="Target path for extract operation")
    parser.add_argument(
        "--out", help="Output path for emit operation (default: .git/tiq/mirror.json)")
    parser.add_argument("--format", default="json",
                        help="Output/reflect format (json|ce1)")
    parser.add_argument("--mode", default="notes",
                        help="Reflect mode (notes)")
    parser.add_argument("--left", help="Left branch for diff operation")
    parser.add_argument("--right", help="Right branch for diff operation")
    parser.add_argument("-e", "--event", default="", help="Materialize predicate list: content,cadence,size,host-moved")
    
    args = parser.parse_args()
    
    events = set([e.strip() for e in (args.event or "").split(',') if e.strip()])
    config = TIQConfig(
        host=args.host,
        include_non_git=args.include_non_git,
        prefix=args.prefix,
        events=events
    )
    
    tiq = TIQ(config)
    
    try:
        if args.verb == Verb.SUPERPOSE.value:
            passports = tiq.superpose()
            tiq.print_table(passports)
        elif args.verb == Verb.EXTRACT.value:
            if not args.branch or not args.target:
                raise TIQError("Extract requires --branch and --target")
            tiq.extract(args.branch, args.target)
        elif args.verb == Verb.MAP.value:
            passports = tiq.map_branches()
            tiq.print_table(passports)
        elif args.verb == Verb.DIFF.value:
            if not args.left or not args.right:
                raise TIQError("Diff requires --left and --right")
            tiq.diff_branches(args.left, args.right)
        elif args.verb == Verb.CLASSIFY.value:
            tiq.classify()
        elif args.verb == Verb.EMIT.value:
            tiq.emit_metadata(out_path=args.out)
        elif args.verb == Verb.REFLECT.value:
            tiq.reflect_metadata(mode=args.mode, fmt=args.format)
        elif args.verb == Verb.REBALANCE.value:
            tiq.rebalance()
        elif args.verb == Verb.ROTATE.value:
            tiq.rotate(scale=args.mode)
        elif args.verb == Verb.PRUNE.value:
            # reuse --mode flag as boolean 'apply' if set to 'apply'
            apply = (args.mode or "").lower() in ("apply", "true", "yes", "on")
            tiq.prune(apply=apply)
    except TIQError as e:
        print(f"✖ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
