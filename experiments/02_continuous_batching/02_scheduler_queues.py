"""Admit waiting requests into a fixed-capacity running batch"""

from mini_vllm.request import Request
from mini_vllm.scheduler import Scheduler


def print_scheduler(scheduler: Scheduler) -> None:
    """Print the requests waiting for and occupying batch slots"""
    waiting = [
        request.request_id for request in scheduler.waiting_requests
    ]
    running = [
        request.request_id for request in scheduler.running_requests
    ]

    print(f"Waiting requests: {waiting}")
    print(f"Running requests: {running}")


def main() -> None:
    scheduler = Scheduler(max_batch_size=2)
    requests = [
        Request("request-0", [10, 11], max_new_tokens=4),
        Request("request-1", [20, 21, 22], max_new_tokens=4),
        Request("request-2", [30], max_new_tokens=4),
    ]

    for request in requests:
        scheduler.add_request(request)

    print("Before scheduling")
    print_scheduler(scheduler)

    scheduler.schedule()

    print("\nAfter scheduling")
    print_scheduler(scheduler)

    print("\nRequest statuses")
    for request in requests:
        print(f"{request.request_id}: {request.status.value}")


if __name__ == "__main__":
    main()