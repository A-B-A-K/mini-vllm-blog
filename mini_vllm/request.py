"""State tracked by the inference engine for one generation request"""

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(Enum):
    """The stages a request moves through inside the engine."""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    """Store the tokens, output limit, and lifecycle of one request"""

    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list, init=False)
    status: RequestStatus = field(
        default=RequestStatus.WAITING,
        init=False,
    )

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError(
                "prompt token ids must contain at least one token"
            )

        if self.max_new_tokens < 1:
            raise ValueError("max new tokens must be at least 1")

    @property
    def token_ids(self) -> list[int]:
        """Return the prompt followed by every token generated so far"""
        return self.prompt_token_ids + self.generated_token_ids

    def start(self) -> None:
        """Move a waiting request into the running state"""
        if self.status is not RequestStatus.WAITING:
            raise RuntimeError("only a waiting request can start")
        self.status = RequestStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        """Record one generated token and finish at the request limit"""
        if self.status is not RequestStatus.RUNNING:
            raise RuntimeError(
                "Tokens can only be added to running requests"
            )

        self.generated_token_ids.append(token_id)
        if len(self.generated_token_ids) == self.max_new_tokens:
            self.status = RequestStatus.FINISHED
