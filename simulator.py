from typing import List, Any, Tuple

import heapq
from scipy import stats
import numpy as np


def simulate(event_generators, initial_time=0):
    """ Run a simulation with the supplied
        event generators and starting time """
    pq: List[Tuple[int, Any]] = []
    for event in event_generators:
        offset, result = next(event)
        heapq.heappush(pq, (offset + initial_time, result, event))

    while True:
        timestamp, result, event = heapq.heappop(pq)
        offset, next_result = event.send(timestamp)
        heapq.heappush(pq, (timestamp + offset, next_result, event))
        yield (timestamp, result)



def makeprng(func):
    import time
    def call_with_prng(*args, prng=None, seed=None, **kwargs):
        if prng is None:
            if seed is None:
                seed = int(time.time()) & ((1 << 32) - 1)
            prng = np.random.RandomState(seed)
        return func(*args, prng=prng, seed=seed, **kwargs)
    return call_with_prng


@makeprng
def legitimate_entry_types(prng=None, seed=None):
    size = 256

    entry_types = ["contactless", "chip_and_pin", "swipe", "manual", "online"]
    entry_probs = [0.25, 0.2, 0.15, 0.05, 0.35]

    while True:
        stream = [entry_types[i] for i in prng.choice(len(entry_types), p=entry_probs, size=size)]
        yield from stream



def makedist(dist_cls, seed=None, prng=None, **kwargs):
    d = dist_cls(**kwargs)
    d.random_state = (seed and seed) or prng.randint((1 << 32) - 1)
    return d


@makeprng
def merchant_stream(common_merchants, all_merchants, fav_percentage=0.2, probs=[0.6,0.37,0.03], prng = None, seed=None):
    favorite_merchants = prng.choice(common_merchants,
                                          size=int(len(common_merchants) * fav_percentage))
    merchants = [favorite_merchants, common_merchants, all_merchants]
    while True:
        pool = merchants[prng.choice(len(merchants), p=probs)]
        yield int(prng.choice(pool))


@makeprng
def legitimate_user_stream(user_id, transactions_per_day=12, start_timestamp=0, amount_means=[20, 100, 500],
                           amount_probs=[0.9, 0.075, 0.025], prng=None, seed=None):
    amounts = transaction_amounts(amount_means, amount_probs, seed=prng.randint((1 << 32) - 1))
    entry_types = legitimate_entry_types(seed=prng.randint((1 << 32) - 1))
    merchants = merchant_stream(common_merchants, np.arange(MERCHANT_COUNT), seed=prng.randint((1 << 32) - 1))

    SECONDS_PER_DAY = 86400
    SECONDS_PER_HOUR = 60 * 60
    loc = SECONDS_PER_DAY // transactions_per_day
    p = 1 / (loc / 10)

    geom = makedist(stats.geom, prng=prng, p=p, loc=loc)

    # choose an arbitrary timezone offset (in seconds)
    tz_offset = (user_id % 24) * SECONDS_PER_HOUR
    WAKEUP_TIME = 7 * SECONDS_PER_HOUR
    SLEEP_TIME = 21 * SECONDS_PER_HOUR

    ts = start_timestamp
    snooze = False

    while True:
        amount = next(amounts)
        entry = next(entry_types)
        foreign = entry == "online" and prng.choice([True, False], p=[0.4, 0.6])

        merchant_id = next(merchants)

        offset = geom.rvs()

        localtime = (ts + tz_offset) % SECONDS_PER_DAY
        if localtime < WAKEUP_TIME or localtime > SLEEP_TIME:
            snooze = True

        while snooze:
            localtime = (ts + tz_offset + offset) % SECONDS_PER_DAY
            if localtime > WAKEUP_TIME and localtime < SLEEP_TIME:
                snooze = False
            offset += geom.rvs()

        result = {
            "user_id": user_id,
            "amount": amount,
            "merchant_id": merchant_id,
            "entry": entry,
            "foreign": foreign
        }
        ts = yield (offset, ("legitimate", *result.values()))


@makeprng
def fraud_entry_types(prng=None, seed=None):
    size = 256

    entry_types = ["contactless", "chip_and_pin", "swipe", "manual", "online"]
    entry_probs = [0.05, 0.05, 0.05, 0.35, 0.5]

    while True:
        stream = [entry_types[i] for i in prng.choice(len(entry_types), p=entry_probs, size=size)]
        yield from stream


@makeprng
def fraudulent_user_stream(user_id, transactions_per_day=12, transactions_per_burst=10, amount_means=[5, 10, 20],
                           amount_probs=[0.2, 0.2, 0.6], prng=None, seed=None):
    amounts = transaction_amounts(amount_means, amount_probs, seed=prng.randint((1 << 32) - 1))
    entry_types = fraud_entry_types(seed=prng.randint((1 << 32) - 1))

    SECONDS_PER_DAY = 86400
    loc = SECONDS_PER_DAY // (transactions_per_day * transactions_per_burst)
    p = 1 / 10

    poisson = makedist(stats.poisson, prng=prng, mu=transactions_per_burst)
    foldnorm = makedist(stats.foldnorm, prng=prng, c=1.8, loc=SECONDS_PER_DAY * 30, scale=1 << 20)
    geom = makedist(stats.geom, prng=prng, p=p, loc=loc)
    while True:
        # consider also np.floor(stats.gamma.rvs(a=6.4, loc=SECONDS_PER_DAY * 90, scale=SECONDS_PER_DAY, size=1))
        fraud_delay = np.floor(foldnorm.rvs())
        fraud_delay = int(fraud_delay)

        fraud_count = poisson.rvs()

        ams = [next(amounts) for _ in range(fraud_count)]
        ens = [next(entry_types) for _ in range(fraud_count)]
        fs = prng.choice([True, False], p=[0.3, 0.7], size=fraud_count)

        m_ids = prng.choice(MERCHANT_COUNT, size=fraud_count)
        offsets = geom.rvs(size=fraud_count)
        offsets[0] += fraud_delay

        for offset, amount, merchant_id, entry, foreign in zip(offsets, ams, m_ids, ens, fs):
            result = {
                "user_id": user_id,
                "amount": amount,
                "merchant_id": merchant_id,
                "entry": entry,
                "foreign": foreign
            }
            yield (offset, ("fraud", *result.values()))



def setup(user_count = 100000, seed = None):
    MERCHANT_COUNT = 20000

    # a small percentage of merchants account for most transactions
    COMMON_MERCHANT_COUNT = MERCHANT_COUNT // 21
    prng = np.random.RandomState(seed or 0xda7aba5e)

    common_merchants = prng.random.choice(MERCHANT_COUNT,
                                          size=COMMON_MERCHANT_COUNT,
                                          replace=True)

    user_seeds = prng.random.randint(1<<32, size=user_count)

