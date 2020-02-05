import scipy
import heapq

class Simulation(object):

    def __init__(self, event_generators) -> None:
        super().__init__()
        self.pq = []
        for event in event_generators:
            offset, result = next(event)
            heapq.heappush(self.pq, (offset, result, event))

    def __next__(self):
        timestamp, result, event = heapq.heappop(self.pq)
        offset, next_result = next(event)
        heapq.heappush(self.pq, (timestamp + offset, next_result, event))
        yield (timestamp, result)


