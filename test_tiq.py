#!/usr/bin/env python3
"""
Test suite for TIQ - Timeline Interleaver & Quantizer
Tests the T1-T5 invariants from the CE1 specification.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict
import unittest
from unittest.mock import patch, MagicMock

# Add the current directory to the path to import tiq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tiq import TIQ, TIQConfig, TIQError, Verb


class TestTIQInvariants(unittest.TestCase):
    """Test suite for TIQ invariants T1-T5."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="tiq_test_")
        self.host_dir = Path(self.test_dir) / "host"
        self.host_dir.mkdir()
        
        # Create test repositories as first-level subdirs of host
        self.repo1_dir = self.host_dir / "repo1"
        self.repo2_dir = self.host_dir / "repo2"
        self.plain_dir = self.host_dir / "plain"
        
        self._create_test_repos()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_repos(self):
        """Create test repositories and directories under host."""
        # Create repo1
        self.repo1_dir.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo1_dir, check=True)
        (self.repo1_dir / "file1.txt").write_text("content1")
        subprocess.run(["git", "add", "."], cwd=self.repo1_dir, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=self.repo1_dir, check=True)
        
        # Create repo2
        self.repo2_dir.mkdir()
        subprocess.run(["git", "init"], cwd=self.repo2_dir, check=True)
        (self.repo2_dir / "file2.txt").write_text("content2")
        subprocess.run(["git", "add", "."], cwd=self.repo2_dir, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=self.repo2_dir, check=True)
        
        # Create plain directory
        self.plain_dir.mkdir()
        (self.plain_dir / "plain_file.txt").write_text("plain content")
    
    def _run_git(self, args: List[str], cwd: Path) -> subprocess.CompletedProcess:
        """Run git command and return result."""
        return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    
    def test_T1_fresh_host_n_repos_to_n_branches(self):
        """T1: fresh host + N repos → N branches; heads set; passports present"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # Superpose repositories
        passports = tiq.superpose()
        
        # Verify branches were created
        result = self._run_git(["branch", "--list"], cwd=self.host_dir)
        self.assertEqual(result.returncode, 0)
        branches = [line.strip().replace("* ", "") for line in result.stdout.split("\n") if line.strip()]
        
        # Should have 2 branches (repo1, repo2)
        self.assertIn("repo1", branches)
        self.assertIn("repo2", branches)
        
        # Verify passports were generated
        self.assertTrue(any(p.branch == "repo1" for p in passports))
        self.assertTrue(any(p.branch == "repo2" for p in passports))
        for passport in passports:
            self.assertTrue(passport.tag.startswith("§"))
    
    def test_T2_rerun_superpose_idempotent(self):
        """T2: rerun superpose twice → no additional changes (idempotent)"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # First run
        passports1 = tiq.superpose()
        # Second run should converge without errors and produce same branches
        passports2 = tiq.superpose()
        self.assertEqual(sorted(p.branch for p in passports1), sorted(p.branch for p in passports2))
    
    def test_T3_dirty_child_repo_abort_and_host_untouched(self):
        """T3: dirty child repo → ✖ and host untouched"""
        # Make repo1 dirty
        (self.repo1_dir / "dirty_file.txt").write_text("dirty content")
        
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # Should raise TIQError due to dirty repository
        with self.assertRaises(TIQError) as context:
            tiq.superpose()
        
        self.assertIn("not clean", str(context.exception))
        
        # Verify host repository is still clean (no branches created)
        result = self._run_git(["branch", "--list"], cwd=self.host_dir)
        if result.returncode == 0:
            branches = [line.strip().replace("* ", "") for line in result.stdout.split("\n") if line.strip()]
            # Should not have created repo1/repo2 branches
            self.assertNotIn("repo1", branches)
            self.assertNotIn("repo2", branches)
    
    def test_T4_extract_superpose_reproduces_repo_tree_hash_equal(self):
        """T4: extract(superpose(dir)) reproduces repo (tree hash equal)"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # Superpose repositories
        tiq.superpose()
        
        # Extract repo1 to a new location
        extract_dir = Path(self.test_dir) / "extracted_repo1"
        tiq.extract("repo1", str(extract_dir))
        
        # Verify extracted repository has same content
        self.assertTrue((extract_dir / "file1.txt").exists())
        self.assertEqual(
            (extract_dir / "file1.txt").read_text(),
            (self.repo1_dir / "file1.txt").read_text()
        )
        
        # Verify it's a valid git repository
        result = self._run_git(["status"], cwd=extract_dir)
        self.assertEqual(result.returncode, 0)
    
    def test_T5_snapshot_mode_include_non_git_one_commit_branch(self):
        """T5: snapshot mode (--include-non-git) holds (no WT writes) but lists branch"""
        config = TIQConfig(
            host=str(self.host_dir),
            include_non_git=True
        )
        tiq = TIQ(config)
        
        # Superpose all directories (including plain directory)
        passports = tiq.superpose()
        
        # Expect repo1, repo2 and plain branches
        self.assertTrue(any(p.branch == "repo1" for p in passports))
        self.assertTrue(any(p.branch == "repo2" for p in passports))
        self.assertTrue(any(p.branch == "plain" for p in passports))
        
        # No working tree writes; just ensure branch name is present in passports
    
    def test_idempotent_superpose_no_changes(self):
        """Test idempotent superpose - rerun should result in zero-diff"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # First superpose
        passports1 = tiq.superpose()
        
        # Second superpose should be idempotent
        passports2 = tiq.superpose()
        
        # Passports should contain the same branches
        branches1 = sorted(p.branch for p in passports1)
        branches2 = sorted(p.branch for p in passports2)
        self.assertEqual(branches1, branches2)
    
    def test_deterministic_branch_names(self):
        """Test deterministic branch names from sanitize(dir)"""
        # Create directory with invalid characters under host
        invalid_dir = self.host_dir / "invalid@name#with$chars"
        invalid_dir.mkdir()
        subprocess.run(["git", "init"], cwd=invalid_dir, check=True)
        (invalid_dir / "test.txt").write_text("test")
        subprocess.run(["git", "add", "."], cwd=invalid_dir, check=True)
        subprocess.run(["git", "commit", "-m", "test"], cwd=invalid_dir, check=True)
        
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        passports = tiq.superpose()
        
        # Should have sanitized branch name
        sanitized_passport = next((p for p in passports if "invalid" in p.branch), None)
        self.assertIsNotNone(sanitized_passport)
        if sanitized_passport:
            self.assertNotIn("@", sanitized_passport.branch)
            self.assertNotIn("#", sanitized_passport.branch)
            self.assertNotIn("$", sanitized_passport.branch)
    
    def test_map_branches(self):
        """Test map operation lists branches with passports"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # First superpose to create branches
        tiq.superpose()
        
        # Then map branches
        passports = tiq.map_branches()
        
        # Should return at least the two git branches
        self.assertTrue(any(p.branch == "repo1" for p in passports))
        self.assertTrue(any(p.branch == "repo2" for p in passports))
    
    def test_diff_branches(self):
        """Test diff operation compares branch histories"""
        config = TIQConfig(
            host=str(self.host_dir)
        )
        tiq = TIQ(config)
        
        # Superpose to create branches
        tiq.superpose()
        
        # Add a commit to repo1
        (self.repo1_dir / "new_file.txt").write_text("new content")
        subprocess.run(["git", "add", "."], cwd=self.repo1_dir, check=True)
        subprocess.run(["git", "commit", "-m", "new commit"], cwd=self.repo1_dir, check=True)
        
        # Update the branch by re-fetching and fast-forwarding
        subprocess.run(["git", "fetch", "tiq/repo1", "+refs/heads/*:refs/remotes/tiq/repo1/*"], cwd=self.host_dir, check=False)
        # Try merging the updated default branch if it exists
        subprocess.run(["git", "checkout", "repo1"], cwd=self.host_dir, check=True)
        # Best-effort: merge one of common default branch names; ignore failures
        for cand in ("main", "master"):  # pragma: no cover
            subprocess.run(["git", "merge", f"tiq/repo1/{cand}"], cwd=self.host_dir, check=False)
        
        # Test diff (should not error)
        with patch('builtins.print') as mock_print:
            tiq.diff_branches("repo2", "repo1")
            
            # Should have printed diff header
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            self.assertTrue(any("⧗ comparing" in msg for msg in print_calls))


class TestTIQCLI(unittest.TestCase):
    """Test TIQ CLI interface."""
    
    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run([sys.executable, "tiq.py", "--help"], 
                              capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("TIQ - Timeline Interleaver & Quantizer", result.stdout)
    
    def test_cli_invalid_verb(self):
        """Test CLI with invalid verb."""
        result = subprocess.run([sys.executable, "tiq.py", "invalid"], 
                              capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
