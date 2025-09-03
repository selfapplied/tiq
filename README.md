# TIQ - Timeline Interleaver & Quantizer

⧗ A reversible Git repository management tool for dirs↔branches superposition.

## Overview

TIQ (Timeline Interleaver & Quantizer) is a sophisticated Git repository management tool that enables reversible superposition of directories as branches. It allows you to:

- **Superpose**: Convert subdirectories into branches within a host repository
- **Extract**: Extract branches back into standalone repositories  
- **Map**: List all branches with their cryptographic passports
- **Diff**: Compare histories between branches
- **Classify**: In-memory mirror-kernel classification of branches vs ghost refs
- **Emit**: Write mirror-kernel metadata to disk (JSON under .git)
- **Reflect**: Persist mirror metadata into Git notes (no history rewrite)
 - **Rebalance**: Fast-forward heads to ghost refs when safe (no rewrite)

## Key Features

- **Reversible Operations**: All operations maintain full reversibility
- **No History Rewrite**: Only adds refs/objects, never rewrites history
- **Cryptographic Passports**: Each branch gets a unique cryptographic identity
- **Clean State Enforcement**: Aborts on dirty repositories to maintain invariants
- **Deterministic Branch Names**: Sanitizes directory names to valid Git branch names
- **Non-Git Directory Support**: Can snapshot plain directories as orphan branches

## CE1 Specification

TIQ is defined by the following CE1 specification:

```ce1
CE1{
  name=TIQ                      // Timeline Interleaver & Quantizer
  glyph=⧗                       // tick
  version=0.1
  intent=dirs↔branches reversible.superposition

  scope{
    host="."                    // aggregator repo (init if absent)
    dirs.depth=1                // only first-level subdirs
    ignore={".git",".venv","node_modules"}
  }

  verbs{
    superpose  = "dirs → branch-per-dir (no rewrite)"
    extract    = "branch → standalone repo at target"
    map        = "list branches + heads + passports"
    diff       = "compare histories: left..right"
  }

  flags{
    --include-non-git           // snapshot plain dirs as orphan branches
    --prefix=<path>             // snapshot under subpath
    -e, --event                 // materialization predicates (content,cadence,size,host-moved)
  }

  passport{
    tag = § blake3(tree(branch))[:8] ":" crc16(meta)
    cadence = per-branch-touch
  }
}
```

## Installation

### Requirements

- Python 3.7+
- Git
- Optional: `blake3` for enhanced cryptographic hashing

### Install Dependencies

```bash
# Install with pip
pip install -r requirements.txt

# Or install blake3 for enhanced hashing
pip install blake3
```

## Usage

### Basic Commands

```bash
# Superpose directories as branches (sandbox-first)
python tiq.py superpose

# Include non-Git directories
python tiq.py superpose --include-non-git

# Map all branches with passports
python tiq.py map

# Extract a branch to standalone repository
python tiq.py extract --branch my-branch --target /path/to/extract

# Compare branch histories
python tiq.py diff --left branch1 --right branch2

# Classify mirror equilibrium (host vs ghost)
python tiq.py classify

# Emit a safe rebalance script (verifies invariants + FF-only)
python tiq.py emit --out .git/tiq/rebalance.sh

# Reflect mirror metadata into Git notes (CE1 format)
python tiq.py reflect --mode notes --format ce1

# Rebalance (FF-only) branches to their ghost refs
python tiq.py rebalance
```

### Command Line Options

- `--host <path>`: Host repository path (default: ".")
- `--include-non-git`: Include non-Git directories as orphan branches
- `--prefix <path>`: Prefix for snapshot operations
- `-e, --event`: Materialize predicate list: `content,cadence,size,host-moved`

## Examples

### Superposing Multiple Repositories

```bash
# Given directory structure:
# project/
# ├── frontend/     (Git repo)
# ├── backend/      (Git repo)  
# └── docs/         (plain directory)

# Superpose all as branches (materialization is event-driven)
python tiq.py superpose --include-non-git -e content

# Result: project becomes a Git repo with branches:
# - frontend
# - backend  
# - docs (orphan branch with snapshot)
```

### Extracting a Branch

```bash
# Extract the frontend branch to a new location
python tiq.py extract --branch frontend --target ../frontend-extracted
```

### Mapping Branches

```bash
# List all branches with their passports
python tiq.py map

# Output:
# Branch Map:
# dir | type | branch | head | §
# ----------------------------------
# frontend | git | frontend | a1b2c3d | §f8a2b1c4:3e7d
# backend | git | backend | e4f5g6h | §9c3d4e5f:2a8b
# docs | snapshot | docs | i7j8k9l | §1b2c3d4e:5f6g
```

## Invariants

TIQ maintains several critical invariants:

1. **Idempotent Superpose**: Rerunning superpose results in zero-diff
2. **Reversible Operations**: `extract(superpose(dir))` reproduces original repo
3. **No History Rewrite**: Only adds refs/objects, never rewrites
4. **Clean State Enforcement**: Aborts on dirty repositories
5. **Deterministic Branch Names**: Sanitizes directory names consistently

### Emergent properties

- **Deterministic convergence**: Repeated superpose drives the host to a fixed point; subsequent runs plan no ops beyond changed children.
- **Reversible views**: Branches behave like lossless views over directories; extract∘superpose preserves trees.
- **Stable identities**: Passports `§{blake3:crc}` expose content drift at a glance while staying compact.
- **Functorial mapping**: The mapping directory → branch preserves structure; fetch updates propagate without history rewrite.
- **Object-store economy**: Git deduplicates objects across branches, making the superposed aggregate compact.
- **Naming lattice**: `sanitize(dir)` yields deterministic branch names; collisions resolve predictably.
- **Minimal-churn cadence**: Per-branch touch highlights only changed dirs in maps and reports.
- **Diff-as-dynamics**: `left..right` operates as a temporal arrow between snapshots/children inside the host index.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest test_tiq.py -v

# Run specific test
python -m pytest test_tiq.py::TestTIQInvariants::test_T1_fresh_host_n_repos_to_n_branches -v
```

The test suite covers all T1-T5 invariants from the CE1 specification:

- **T1**: Fresh host + N repos → N branches; heads set; passports present
- **T2**: Rerun superpose --dry after --apply → no planned ops  
- **T3**: Dirty child repo → ✖ and host untouched
- **T4**: Extract(superpose(dir)) reproduces repo (tree hash equal)
- **T5**: Snapshot mode (--include-non-git) → 1-commit branch with dir tree

## Passport System

Each branch receives a cryptographic passport with:

- **Tag**: `§{blake3_hash[:8]}:{crc16_hash}`
- **Head Short**: Short commit hash
- **Date Short**: Commit date in YYYY-MM-DD format

The passport provides a unique, verifiable identity for each branch.

## Error Handling

TIQ enforces strict error handling:

- **Dirty Repositories**: Aborts if any repository has uncommitted changes
- **Missing Dependencies**: Graceful fallbacks for optional dependencies
- **Invalid Operations**: Clear error messages for invalid operations
- **Clean Abort**: Never leaves repositories in inconsistent states

## Development

### Project Structure

```
tiq/
├── tiq.ce1          # CE1 specification
├── tiq.py           # Core implementation
├── test_tiq.py      # Test suite
├── README.md        # This file
├── pyproject.toml   # Python project configuration
├── Makefile         # Build and test commands
└── LICENSE          # License file
```

## Mirror Kernel (CE1: lens=MirrorKernel)

The mirror-kernel tracks Git metadata in memory by reflecting each branch `refs/heads/<b>` against its ghost `refs/tiq/ghost/<b>` produced during superpose. It reports per-branch statistics:

- branch | host | ghost | ahead | behind | energy | eq | ff
- **eq**: 1 if host head equals ghost head (axis equilibrium)
- **ahead/behind**: commit counts (host-only vs ghost-only) via `rev-list --left-right --count`
- **energy**: approximate churn computed from `diff-tree --numstat`
- **ff**: 1 if host is ancestor of ghost (fast-forward possible)

Run:

```bash
python tiq.py classify
```

This is read-only and maintains all invariants: no working tree writes, no history rewrite.

### Emitting Mirror Metadata

By default, the emit command writes to `.git/tiq/mirror.json`:

```bash
python tiq.py emit
```

Custom path and (currently only) JSON format:

```bash
python tiq.py emit --out /tmp/mirror.json --format json
```

### Building

```bash
# Run tests
make test

# Check code quality
make lint

# Build documentation
make docs

# Clean build artifacts
make clean
```

## License

TIQ is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

TIQ is inspired by the principles of reversible computation and the need for clean, deterministic repository management. The CE1 specification format provides a formal foundation for the tool's behavior and invariants.

---

*"Every reversible plan can be cast as a TIQ operation. If Git comprises the universal version control system, then TIQ occupies the space of reversible repository transformations."*
