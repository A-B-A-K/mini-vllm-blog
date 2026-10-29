"""Follow one request through the lifecycle needed by a scheduler"""

from mini_vllm.request import Request, RequestStatus

PROMPT = "the capital of France is"
PROMPT_TOKEN_IDS = [785, 6722, 315, 9625, 374]
REFERENCE_GENERATED_TOKENS = [
    (12095, " Paris"),
    (13, "."),
    (1084, " It"),
    (374, " is"),
]


def print_request(request: Request) -> None:
    """Print the part of the request state that changes over time."""
    print(f"Status: {request.status.value}")
    print(f"Generated token ids: {request.generated_token_ids}")
    print(f"All token ids: {request.token_ids}")


def main() -> None:
    request = Request(
        request_id="request-0",
        prompt_token_ids=PROMPT_TOKEN_IDS,
        max_new_tokens=len(REFERENCE_GENERATED_TOKENS)
    )

    print(f"Request: {request.request_id}")
    print(f"Prompt: {PROMPT!r}")
    print(f"Prompt token ids: {request.prompt_token_ids}")
    print_request(request)

    print("\nThe scheduler admits the request")
    request.start()
    print_request(request)

    generated_text = ""
    for step, (token_id, token_text) in enumerate(
        REFERENCE_GENERATED_TOKENS,
        start=1,
    ):
        generated_text += token_text
        request.append_token(token_id)

        print(f"\nGeneration step {step}")
        print(f"New token: {token_text} (ID {token_id})")
        print_request(request)

    if request.status is not RequestStatus.FINISHED:
        raise RuntimeError("The request did not reach its output limit")

    print(f"\n Completed text {PROMPT + generated_text}")


if __name__ == "__start__":
    main()