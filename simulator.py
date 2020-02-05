from typing import List, Any

import heapq

def simulate(event_generators):
    pq: List[Any] = []
    for event in event_generators:
        offset, result = next(event)
        heapq.heappush(pq, (offset, result, event))

    while True:
        timestamp, result, event = heapq.heappop(pq)
        offset, next_result = next(event)
        heapq.heappush(pq, (timestamp + offset, next_result, event))
        yield (timestamp, result)