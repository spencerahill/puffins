---
name: wrap-up
description: Use when user says "wrap up", "close session", "end session",
  or invokes /wrap-up — runs end-of-session checklist
disable-model-invocation: true
---

# Session Wrap-Up

Adapted from u/ai_grand_master's "Self-Improvement Loop" skill:
https://www.reddit.com/r/ClaudeCode/comments/1r89084/

Run four phases in order. Each phase is conversational and inline. Present
results as you go.

## Phase 1: Ship It

**Tests:**
1. Run `pytest` and report pass/fail status
2. If tests fail, stop here — do not commit broken code. Report failures and
   ask the user how to proceed.

**Commit:**
3. Run `git status` and `git diff --stat` in the repo root
4. If uncommitted changes exist:
   - Show the diff summary and draft a descriptive commit message
   - Ask the user for approval before committing
   - If approved, commit and push
5. If no uncommitted changes, say so and move on

**Task cleanup:**
6. Check the task list for in-progress or stale items
7. Mark completed tasks as done, flag orphaned ones

## Phase 2: Remember It

Review what was learned during the session. For each piece of knowledge,
decide where it belongs:

- **Auto memory** — Debugging insights, patterns discovered, project quirks
- **CLAUDE.md** — Permanent project conventions, architecture decisions
- **`.claude/rules/`** — Topic-specific rules scoped to file types via
  `paths:` frontmatter
- **`CLAUDE.local.md`** — Personal WIP context, local URLs, ephemeral notes

**Decision framework:**
- Permanent project convention? -> CLAUDE.md or `.claude/rules/`
- Scoped to specific file types? -> `.claude/rules/` with `paths:` frontmatter
- Pattern or insight discovered during work? -> Auto memory
- Personal or ephemeral context? -> `CLAUDE.local.md`

Save findings to the appropriate locations.

## Phase 3: Review & Recommend

Analyze the conversation for self-improvement findings. If the session was
short or routine with nothing notable, say "Nothing to improve" and move on.

**Finding categories:**
- **Skill gap** — Things Claude struggled with, got wrong, or needed multiple
  attempts
- **Friction** — Repeated manual steps, things user had to ask for explicitly
  that should have been automatic
- **Knowledge** — Facts about the project, preferences, or setup that Claude
  didn't know but should have
- **Automation** — Repetitive patterns that could become skills, hooks, or
  scripts

**Do not auto-apply.** Present all proposed changes (to CLAUDE.md, rules,
memory, or skill/hook specs) and ask the user for approval before applying.
Then apply only the approved changes and show a summary:

Findings:

1. [Category]: Description of finding
   -> [Target] Proposed change

2. [Category]: Description of finding
   -> [Target] Proposed change

---
No action needed:

3. [Category]: Description — already documented or not actionable

## Phase 4: Session Log

Append a session summary to the auto memory file `session-log.md`. Each entry
should include:

- **Date**: Today's date
- **Accomplished**: Bullet points of what was done
- **Decisions**: Key decisions made during the session
- **Next steps**: Open questions or suggested follow-up work
