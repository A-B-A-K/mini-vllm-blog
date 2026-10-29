import pytest

from mini_vllm.request import Request, RequestStatus


def test_request_moves_from_waiting_to_running_to_finished() -> None:
    request = Request(
        request_id="request-0",
        prompt_token_ids=[10, 20],
        max_new_tokens=2,
    )

    assert request.status is RequestStatus.WAITING
    assert request.token_ids == [10, 20]

    request.start()
    request.append_token(30)

    assert request.status is RequestStatus.RUNNING
    assert request.generated_token_ids == [30]
    assert request.token_ids == [10, 20, 30]

    request.append_token(20)

    assert request.status is RequestStatus.FINISHED
    assert request.token_ids == [10, 20, 30, 40]


def test_request_cannot_receive_tokens_before_it_starts() -> None:
    request = Request(
        request_id="request-0",
        prompt_token_ids=[10],
        max_new_tokens=1,
    )


    with pytest.raises(RuntimeError, match="running request"):
        request.append_token(20)


def test_finished_request_cannot_receive_more_tokens() -> None:
    request = Request(
            request_id="request-0",
            prompt_token_ids=[10],
            max_new_tokens=1,
    )
    request.start()
    request.append_token(20)

    with pytest.raises(RuntimeError, match="running request"):
        request.append_token(30)


def test_request_cannot_start_twice() -> None:
    request = Request(
                request_id="request-0",
                prompt_token_ids=[10],
                max_new_tokens=1,
    )
    request.start()
    with pytest.raises(RuntimeError, match="waiting request"):
        request.start()


@pytest.mark.parametrize(
    ("prompt_token_ids", "max_new_tokens", "message"),
    [
        ([], 1, "at least one token"),
        ([10], 0, "at least 1"),
    ],
)
def test_request_rejects_invalid_inputs(
    prompt_token_ids: list[int],
    max_new_tokens: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match="message"):
        Request(
            request_id="request-0",
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens
        )