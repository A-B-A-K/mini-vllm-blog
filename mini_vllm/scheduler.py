"""Minimal request scheduler for continuous batching"""

from collections import deque

from mini_vllm.request import Request, RequestStatus


class Scheduler:
    """Admit waiting requests into a fixed-capacity running batch"""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")

        self.max_batch_size = max_batch_size
        self.waiting_requests: deque[Request] = deque()
        self.running_requests: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Place a new request at the end of the waiting queue"""
        if request.status is not RequestStatus.WAITING:
            raise RuntimeError(
                "Only waiting requests can be added to the scheduler"
            )
        self.waiting_requests.append(request)

    def schedule(self) -> list[Request]:
        """Fill every available batch slot in fifo order"""
        available_slots = self.max_batch_size - len(self.running_requests)
        request_to_admit = min(
            available_slots,
            len(self.waiting_requests),
        )

        for _ in range(request_to_admit):
            request = self.waiting_requests.popleft()
            request.start()
            self.running_requests.append(request)

        return list(self.running_requests)

    def remove_finished_requests(self) -> list[Request]:
        """Release batch slots occupied by completed requests"""
        finished_requests = [
            request
            for request in self.running_requests
            if request.status is RequestStatus.FINISHED
        ]
        self.running_requests = [
            request
            for request in self.running_requests
            if request.status is not RequestStatus.FINISHED
        ]
        return finished_requests
