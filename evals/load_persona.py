import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Persona:
    id: str
    category: str
    role: str
    disposition: str
    sessions: list[dict]
    rubric: list[str]
    mode: str = "auto"
    checks: list[dict] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)
    approval: dict = field(default_factory=dict)
    deny_rules: list[str] = field(default_factory=list)


def load_personas() -> deque[Persona]:
    directory = Path(__file__).resolve().parent / "personas"
    personas: deque[Persona] = deque()
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        personas.append(Persona(**data))
    return personas
