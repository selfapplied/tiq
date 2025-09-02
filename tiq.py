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
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class Verb(Enum):
    """TIQ operation verbs."""
    SUPERPOSE = "superpose"
    EXTRACT = "extract"
    MAP = "map"
    DIFF = "diff"


@dataclass
class TIQConfig:
    """Configuration for TIQ operations."""
    host: str = "."
    dirs_depth: int = 1
    ignore: Optional[Set[str]] = None
    dry_run: bool = True
    apply: bool = False
    force: bool = False
    include_non_git: bool = False
    prefix: Optional[str] = None
    
    def __post_init__(self):
        if self.ignore is None:
            self.ignore = {".git", ".venv", "node_modules"}


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


class TIQ:
    """Timeline Interleaver & Quantizer - Core implementation."""
    
    def __init__(self, config: TIQConfig):
        self.config = config
        self.host_path = Path(config.host).resolve()
        
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
    
    def _is_clean_repo(self, path: Path) -> bool:
        """Check if a Git repository is clean (no uncommitted changes)."""
        if not self._is_git_repo(path):
            return True
            
        returncode, stdout, stderr = self._run_git(["status", "--porcelain"], cwd=path)
        if returncode != 0:
            raise TIQError(f"Failed to check git status in {path}: {stderr}")
        
        return len(stdout) == 0
    
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
    
    def _blake3_hash(self, data: bytes) -> str:
        """Compute BLAKE3 hash of data."""
        try:
            import blake3
            return blake3.blake3(data).hexdigest()[:8]
        except ImportError:
            # Fallback to SHA-256 if blake3 not available
            return hashlib.sha256(data).hexdigest()[:8]
    
    def _crc16_hash(self, data: bytes) -> str:
        """Compute CRC16 hash of data."""
        return f"{zlib.crc32(data) & 0xFFFF:04x}"
    
    def _generate_passport(self, branch: str, tree_hash: str, meta: str) -> Passport:
        """Generate passport for a branch."""
        tree_data = tree_hash.encode()
        meta_data = meta.encode()
        
        blake3_part = self._blake3_hash(tree_data)
        crc16_part = self._crc16_hash(meta_data)
        
        tag = f"§{blake3_part}:{crc16_part}"
        
        # Get branch head info
        returncode, head_short, _ = self._run_git(["rev-parse", "--short", "HEAD"])
        if returncode != 0:
            head_short = "unknown"
        
        # Get commit date
        returncode, date_short, _ = self._run_git(["log", "-1", "--format=%cd", "--date=short"])
        if returncode != 0:
            date_short = "unknown"
        
        return Passport(
            branch=branch,
            tag=tag,
            head_short=head_short,
            date_short=date_short
        )
    
    def _ensure_host_clean(self):
        """Ensure host repository is clean or initialize it."""
        if not self._is_git_repo(self.host_path):
            if self.config.dry_run:
                print("⧗ would initialize host repository")
                return
            returncode, _, stderr = self._run_git(["init"])
            if returncode != 0:
                raise TIQError(f"Failed to initialize host repository: {stderr}")
            print("✓ initialized host repository")
        else:
            if not self._is_clean_repo(self.host_path):
                raise TIQError("Host repository is not clean. Please commit or stash changes.")
    
    def _add_remote_once(self, remote_name: str, remote_url: str):
        """Add remote if it doesn't exist."""
        returncode, _, _ = self._run_git(["remote", "get-url", remote_name])
        if returncode == 0:
            return  # Remote already exists
        
        if self.config.dry_run:
            print(f"⧗ would add remote {remote_name} -> {remote_url}")
            return
            
        returncode, _, stderr = self._run_git(["remote", "add", remote_name, remote_url])
        if returncode != 0:
            raise TIQError(f"Failed to add remote {remote_name}: {stderr}")
        print(f"✓ added remote {remote_name}")
    
    def _fetch_remote(self, remote_name: str):
        """Fetch all refs from remote."""
        if self.config.dry_run:
            print(f"↷ would fetch {remote_name}")
            return
            
        returncode, _, stderr = self._run_git([
            "fetch", remote_name, 
            "+refs/heads/*:refs/remotes/" + remote_name + "/*",
            "--tags"
        ])
        if returncode != 0:
            raise TIQError(f"Failed to fetch {remote_name}: {stderr}")
        print(f"↷ fetched {remote_name}")
    
    def _get_remote_head_or_first(self, remote_name: str) -> str:
        """Get the head branch of remote or first available branch."""
        # Try to get the default branch
        returncode, head, _ = self._run_git(["symbolic-ref", f"refs/remotes/{remote_name}/HEAD"])
        if returncode == 0:
            return head.replace(f"refs/remotes/{remote_name}/", "")
        
        # Fallback to first branch
        returncode, branches, _ = self._run_git(["branch", "-r", "--format=%(refname:short)"])
        if returncode == 0 and branches:
            for branch in branches.split("\n"):
                if branch.startswith(f"{remote_name}/"):
                    return branch.replace(f"{remote_name}/", "")
        
        raise TIQError(f"No branches found in remote {remote_name}")
    
    def _create_branch(self, branch_name: str, head: str):
        """Create branch from head."""
        if self.config.dry_run:
            print(f"⧗ would create branch {branch_name} from {head}")
            return
            
        force_flag = ["-f"] if self.config.force else []
        returncode, _, stderr = self._run_git(["checkout", "-b", branch_name, head] + force_flag)
        if returncode != 0:
            raise TIQError(f"Failed to create branch {branch_name}: {stderr}")
        print(f"✓ created branch {branch_name}")
    
    def _switch_orphan(self, branch_name: str):
        """Switch to orphan branch."""
        if self.config.dry_run:
            print(f"⧗ would switch to orphan branch {branch_name}")
            return
            
        returncode, _, stderr = self._run_git(["checkout", "--orphan", branch_name])
        if returncode != 0:
            raise TIQError(f"Failed to switch to orphan branch {branch_name}: {stderr}")
        print(f"✓ switched to orphan branch {branch_name}")
    
    def _stage_snapshot(self, source_dir: Path, prefix: Optional[str] = None):
        """Stage snapshot of directory contents."""
        if self.config.dry_run:
            print(f"⧗ would stage snapshot of {source_dir}")
            return
            
        # Clear staging area
        self._run_git(["rm", "-rf", "--cached", "."])
        
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
        """Commit staged snapshot."""
        if self.config.dry_run:
            print(f"⧗ would commit: {message}")
            return
            
        returncode, _, stderr = self._run_git(["commit", "-m", message])
        if returncode != 0:
            raise TIQError(f"Failed to commit snapshot: {stderr}")
        print(f"✓ committed: {message}")
    
    def superpose(self) -> List[Passport]:
        """Superpose directories as branches."""
        print("⧗ superposing directories as branches...")
        
        self._ensure_host_clean()
        passports = []
        
        for dir_path in self._get_directories():
            branch_name = self._sanitize_branch_name(dir_path.name)
            
            if self._is_git_repo(dir_path):
                # Handle Git repository
                if not self._is_clean_repo(dir_path):
                    raise TIQError(f"Repository {dir_path} is not clean")
                
                remote_name = f"tiq/{branch_name}"
                self._add_remote_once(remote_name, str(dir_path))
                self._fetch_remote(remote_name)
                head = self._get_remote_head_or_first(remote_name)
                self._create_branch(branch_name, head)
                
            elif self.config.include_non_git:
                # Handle non-Git directory
                self._switch_orphan(branch_name)
                self._stage_snapshot(dir_path, self.config.prefix)
                self._commit_snapshot(f"import {branch_name} (snapshot)")
            
            # Generate passport
            passport = self._generate_passport(branch_name, "tree_hash", "meta")
            passports.append(passport)
        
        return passports
    
    def extract(self, branch: str, target: str):
        """Extract branch to standalone repository."""
        print(f"⧗ extracting branch {branch} to {target}")
        
        target_path = Path(target)
        target_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.dry_run:
            print(f"⧗ would extract {branch} to {target}")
            return
        
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
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    parser.add_argument("--force", action="store_true", help="Allow force operations")
    parser.add_argument("--include-non-git", action="store_true", help="Include non-Git directories")
    parser.add_argument("--prefix", help="Prefix for snapshot operations")
    parser.add_argument("--branch", help="Branch name for extract operation")
    parser.add_argument("--target", help="Target path for extract operation")
    parser.add_argument("--left", help="Left branch for diff operation")
    parser.add_argument("--right", help="Right branch for diff operation")
    
    args = parser.parse_args()
    
    config = TIQConfig(
        host=args.host,
        dry_run=not args.apply,
        force=args.force,
        include_non_git=args.include_non_git,
        prefix=args.prefix
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
    except TIQError as e:
        print(f"✖ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
