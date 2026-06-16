"""Agent handoff-contract analysis (Box 3) — catch broken data contracts statically.

Multi-agent pipelines pass state between agents via `output_key` (produced) and
`{placeholder}` references in instructions (consumed). When those drift, an agent
references a key nobody produces, or produces one nobody consumes — silent bugs
that don't crash but corrupt the run.

`analyze_pipeline(root_agent)` walks an ADK agent tree in execution order and reports:
- **dangling_reference** — an agent references `{key}` not produced by any upstream
  agent (a hard contract violation).
- **orphan_output** — an agent produces a key no downstream agent references via a
  `{placeholder}` (informational: may be consumed by a tool via session state, or
  be a terminal output).

This is the structural layer of contract testing. Semantic completeness (e.g. an
instruction promising "news" the tools never produce) needs human review or an
LM-judge — see docs/AGENT_CONTRACT_TESTING.md.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


@dataclass
class ContractIssue:
    agent: str
    kind: str  # "dangling_reference" | "orphan_output"
    detail: str


@dataclass
class ContractReport:
    agents: list[str]
    produced: dict  # output_key -> producing agent
    referenced: dict  # key -> [referencing agents]
    issues: list[ContractIssue] = field(default_factory=list)

    @property
    def dangling(self) -> list[ContractIssue]:
        return [i for i in self.issues if i.kind == "dangling_reference"]

    @property
    def orphans(self) -> list[ContractIssue]:
        return [i for i in self.issues if i.kind == "orphan_output"]

    def ok(self) -> bool:
        """Pass when there are no dangling references (orphans are informational)."""
        return not self.dangling

    def summary(self) -> str:
        verdict = "OK" if self.ok() else "FAIL"
        return (
            f"[{verdict}] {len(self.agents)} agents · {len(self.dangling)} dangling refs "
            f"· {len(self.orphans)} orphan outputs"
        )


def _flatten(agent, out: list) -> list:
    """Depth-first flatten to leaf agents, preserving sequential execution order."""
    subs = getattr(agent, "sub_agents", None)
    if subs:
        for sub in subs:
            _flatten(sub, out)
    else:
        out.append(agent)
    return out


def analyze_pipeline(root) -> ContractReport:
    leaves = _flatten(root, [])
    produced: dict[str, str] = {}
    referenced: dict[str, list[str]] = defaultdict(list)
    issues: list[ContractIssue] = []

    for agent in leaves:
        name = getattr(agent, "name", "?")
        instruction = getattr(agent, "instruction", "") or ""
        for ref in sorted(set(_PLACEHOLDER.findall(instruction))):
            referenced[ref].append(name)
            if ref not in produced:
                issues.append(
                    ContractIssue(
                        name,
                        "dangling_reference",
                        f"references {{{ref}}} which no upstream agent produces",
                    )
                )
        out_key = getattr(agent, "output_key", None)
        if out_key:
            produced[out_key] = name

    for key, producer in produced.items():
        if key not in referenced:
            issues.append(
                ContractIssue(
                    producer,
                    "orphan_output",
                    f"produces '{key}' but no agent references {{{key}}} "
                    "(may be consumed by a tool via state, or be a terminal output)",
                )
            )

    return ContractReport(
        agents=[getattr(a, "name", "?") for a in leaves],
        produced=produced,
        referenced=dict(referenced),
        issues=issues,
    )
