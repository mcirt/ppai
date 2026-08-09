# Puzzle Pyramid AI — JavaScript Solver v0.1

This is a **mechanical Puzzle Pyramid conversion** of the working `PyramidSolitaire_v1.5.5` browser solver.

## What changed

- Card ranks were replaced by Puzzle Pyramid emoji identities (`emoji1` … `emoji18`).
- Complementary pairs are:
  - emoji1 ↔ emoji2
  - emoji4 ↔ emoji5
  - emoji7 ↔ emoji8
  - emoji10 ↔ emoji11
  - emoji13 ↔ emoji14
  - emoji16 ↔ emoji17
- Unmatched/clearable singles are: emoji3, emoji6, emoji9, emoji12, emoji15, emoji18.
- `KING` was replaced by canonical `CLEAR` behavior.
- Hold matches are logged/represented as `MATCH_HOLD`.
- Tile IDs remain canonical integers 1–52.
- The original beam/full-plan search and 8260 perfect-score probe are preserved.

## Files

- `index.html` — enter/upload a 52-tile Puzzle Pyramid board.
- `game.html` — manually play the puzzle in JavaScript.
- `solver.html` — calculate and visually replay a JavaScript solution.
- `images/` — Puzzle Pyramid emoji assets.
- `tests/replay-demos.mjs` — regression test against the six verified human demonstrations from the Python PPAI project.

## Run

The files are standalone browser files. Open `index.html` in a browser, enter or upload a puzzle, then choose **Play Puzzle** or **Solve Puzzle**.

For the regression test (requires Node.js):

```bash
node tests/replay-demos.mjs
```

Expected verified scores:

- 7210
- 7910
- 7410
- 7010
- 7310
- 8260

## Scope of v0.1

This version intentionally does **not** add LITP reasoning, endgame deduction, BCS strategy, inventory molding, learning, or demonstration lookup. The goal is first to prove that the working Pyramid Solitaire search can operate under Puzzle Pyramid rules.
