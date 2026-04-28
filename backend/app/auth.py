"""Auth stub for v1 scaffold.

Real implementation will validate Auth0-issued JWTs and return the
authenticated user. For now this returns a fixed test user so the rest
of the scaffold compiles and can be exercised end-to-end.
"""

import uuid
from dataclasses import dataclass


@dataclass
class CurrentUser:
    id: uuid.UUID
    email: str
    role: str
    company_id: uuid.UUID


_TEST_USER = CurrentUser(
    id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
    email="test@proofread.local",
    role="producer",
    company_id=uuid.UUID("00000000-0000-0000-0000-000000000aaa"),
)


def get_current_user() -> CurrentUser:
    return _TEST_USER
