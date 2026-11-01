import pytest

from mini_vllm.request import Request, RequestStatus
from mini_vllm.scheduler import Scheduler


def make_request(request_id: str) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[10],
        max_new_tokens=2,
    )


def test_scheduler_admits_requets_in_fifo_order() -> None:
    scheduler = Scheduler(max_batch_size=2)
    requests = [
        make_request("request-0"),
        make_request("request-1"),
        make_request("request-2"),
    ]
    for request in requests:
        scheduler.add_request(request)

    running_requests = scheduler.schedule()

    assert [request.request_id for reqeust in running_requests] == [
        "request-0",
        "request-1",
    ]

    assert [
        request.request_id for reqeust in scheduler.waiting_requests
    ] == ["request-2"]
    assert requests[0].status is RequestStatus.RUNNING
    assert requests[1].status is RequestStatus.RUNNING    
    assert requests[2].status is RequestStatus.WAITING


def test_scheduler_fills_only_the_available_batch_slots() -> None:
    scheduler = Scheduler(max_batch_size=2)
    first_request = make_request("request-0")
    scheduler.add_request(first_request)
    scheduler.schedule()

    scheduler.add_request(make_request("request-1"))
    scheduler.add_request(make_request("request-2"))
    running_requests = scheduler.schedule()

    assert [ request.request_id for request in running_requests] == [
        "request-0",
        "request-1",
    ]

    assert [
        request.request_id for request in scheduler.waiting_requests
    ] == ["request-2"]


def test_scheduler_requires_at_least_one_batch_slot() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        Scheduler(max_batch_size=0)


def test_scheduler_only_accepts_waiting_requests() -> None:
    scheduler = Scheduler(max_batch_size=1)
    request = make_request("request-0")
    request.start()

    with pytest.raises(RuntimeError, match="waiting requests"):
        scheduler.add_request(request)