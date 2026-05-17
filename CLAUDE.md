# Claude Code Instructions for this Project

## You must always:
1. Read `doc/06_coding_conventions.md` before any code change.
2. Write tests before implementation when working on solver modules.
3. Use existing types from `src/leo_alloc/utils/config.py`; do not redefine them.
4. Follow the terminology table in `doc/00_research_context.md`.
5. Keep each function under 50 lines unless there is a documented reason.
6. Keep each module file under 500 lines unless there is a documented reason.

## You must never:
1. Modify `ScenarioInstance` without explicit user approval.
2. Create new source files outside the structure in `doc/01_system_architecture.md`.
3. Use `print()` for logging; use `leo_alloc.utils.logging.get_logger`.
4. Catch generic `Exception`; catch specific exception types.
5. Use `np.random` global state; use `np.random.default_rng(seed)`.
6. Silently relax hard handover budgets.

## When unsure:
Ask a concise question before making an architectural change.
