#!/usr/bin/env python
# coding: utf-8

# # Generating synthetic payments data
# 
# In this notebook, we'll build up a very simple simulator to generate payments data corresponding to legitimate and fraudulent transactions.  (There are many ways you could improve this generator and we'll call some of them out.)  We'll start by building up some functionality to run simulations in general.
# 
# ## An (extremely) basic discrete-event simulation framework
# 
# The next function is all you need to run simple discrete-event simulations.  Here's how to use it:
# 
# - you'll define several streams of events, each of which is modeled by a Python generator,
# - each event stream generator will `yield` a tuple consisting of *an offset* (the amount of time that has passed since the last event of that type) and *a result* (an arbitrary Python value associated with the event),
# - the generator produced by the `simulate` function will yield the next event from all event streams indefinitely.

# In[ ]:


import heapq

def simulate(event_generators, initial_time=0):    
    def setup_e(e, i):
        offset, result = next(e)
        return ((offset + i), result, e)
    
    pq = [setup_e(event, initial_time)
          for event in event_generators]
    heapq.heapify(pq)
    
    while True:
        timestamp, result, event = pq[0]
        offset, next_result = event.send(timestamp)
        heapq.heappushpop(pq, (timestamp + offset, next_result, event))
        yield (timestamp, result)


# It may be easier to see how this works with an example.  In the next three cells, we 
# 
# 1. define a generator for event streams, which samples interarrival times from a Poisson distribution and returns a predefined string as the result at each event,
# 2. set up a simulation with four streams, each of which has a different distribution of interarrival times and value, and
# 3. take the first twenty events from the simulation

# In[ ]:


from scipy import stats

def bedrockstream(mu, name):
    while True:
        offset, = stats.poisson.rvs(mu, size=1)
        x = yield (offset, name)


# In[ ]:


sim = simulate([bedrockstream(10, "fred"), 
                bedrockstream(12, "betty"), 
                bedrockstream(20, "wilma"), 
                bedrockstream(35, "barney")])


# In[ ]:


for i in range(20):
    print(next(sim))


# ### Sidebar:  repeatability and pseudorandom number generation
# 
# There are a couple of small problems here:  
# 
# 1.  We aren't seeding our random number generator, which means that our results won't be deterministic (in general, we'd like simulations to be deterministic so we can replay them with different parameters or policies), and
# 2.  we're using the same random number generator for every user in the stream, which means (among other things) that any long-period autocorrelations in our pseudorandom number stream will show up in our simulation results.  (It also means that the behavior of any given user will depend on how many users there are in the simulation!)
# 
# We can solve both problems by using a separate generator for each user and seeding it.  Scipy will let us create a stream of numbers sampled from a given distribution as an object, but it won't let us pass in a seed to the constructor.  However, we can set the seed after we create the stream, like this:

# In[ ]:


import numpy as np

def det_bedrockstream(mu, name, seedpart=0xda7aba5e):
    
    # scipy doesn't let us specify a seed in the constructor...
    poisson = stats.poisson(mu)
    
    # ...so we'll set one up after creating the object
    seed = (hash(name) ^ seedpart) & (1 << 31 - 1)
    poisson.random_state = np.random.default_rng(seed=seed)
    
    while True:
        offset, = poisson.rvs(size=1)
        x = yield (offset, name)
        
sim = simulate([det_bedrockstream(10, "fred"), 
                det_bedrockstream(12, "betty"), 
                det_bedrockstream(20, "wilma"), 
                det_bedrockstream(35, "barney")])

for i in range(20):
    print(next(sim))


# ## Modeling transactions
# 
# The first thing we need to do is to decide what data we'll generate for each transaction.  Some interesting possibilities include:
# 
# - user ID
# - merchant ID
# - merchant type
# - transaction amount (assuming a single currency)
# - card entry mode (e.g., contactless, chip and pin, swipe, card manually keyed, or online transaction)
# - foreign transaction (whether or not the user's home country matches the country in which the transaction is taking place)
# 
# We'll also generate a label for each transaction (`legitimate` or `fraud`).  We'll start with a very basic user event stream generator:  all of the transactions we generate will be legitimate, and we won't do anything particularly interesting with most of the fields.  We also won't bother making this very basic simulation deterministic.

# In[ ]:


MERCHANT_COUNT = 20000

# a small percentage of merchants account for most transactions
COMMON_MERCHANT_COUNT = MERCHANT_COUNT // 21

np.random.seed(0xda7aba5e)

common_merchants = np.random.choice(MERCHANT_COUNT, 
                                    size=COMMON_MERCHANT_COUNT, 
                                    replace=True)

def basic_user_stream(user_id, mu, seed=None):
    
    favorite_merchants = np.random.choice(common_merchants,
                                         size=len(common_merchants) // 5)
    while True:
        amount = 100.00
        entry = "chip_and_pin"
        foreign = False
        
        merchant_id, = np.random.choice(favorite_merchants, size=1)
        offset = stats.poisson.rvs(mu)
        result = {
            "user_id": user_id,
            "amount": amount,
            "merchant_id": merchant_id,
            "entry": entry,
            "foreign": foreign
        }
        yield (offset, ("legitimate", *result.values()))


# In[ ]:


sim = simulate([basic_user_stream(1, 700), basic_user_stream(2, 105), basic_user_stream(3, 40)])


# In[ ]:


for i in range(20):
    print(next(sim))


# ## Exercise:  some quick improvements
# 
# 1.  Users don't always just buy things from a few favorite merchants.  Change `basic_user_stream` so that they occasionally buy from any merchant.
# 2.  Most people buy many inexpensive things and relatively few expensive things.  Use this insight to generate (more) realistic transaction amounts.
# 3.  Some small percentage of online sales will be foreign transactions.  When a user is traveling abroad, nearly all of his or her transactions will be foreign transactions.  Add some state to `basic_user_stream` to model occasional international travel.

# ## Building a better transaction stream
# 
# We'll start by building a generator to build a mixture model we can use to make several kinds of transactions:  small, medium, and large.  We'll need to do a bit of extra work to make this generator deterministic, but we'll start by showing a nondeterministic version.

# In[ ]:


def nd_transaction_amounts(means, percentages, distribution=None):
    size = 256
    
    if distribution is None:
        distribution = lambda m, sz: stats.gamma.rvs(a=1.1, scale=min(m, 750), loc=m, size=sz)
    
    while True:
        streams = [distribution(m * 100, size) for m in means]
        stream = np.floor(np.choose(np.random.choice(len(means), p=percentages, size=size), streams)) / 100
        
        yield from stream


# In[ ]:


import pandas as pd

import altair as alt
alt.data_transformers.enable('json')

amt = nd_transaction_amounts([5, 15, 50], [0.5, 0.35, 0.15])
amounts = [next(amt) for i in range(80000)]

source = pd.DataFrame({"amounts": amounts})

alt.Chart(source).mark_bar().encode(
    alt.X("amounts", bin=alt.Bin(maxbins=100)),
    y='count()'
)


# We can also plot a broader distribution of transactions:

# In[ ]:


get_ipython().run_cell_magic('time', '', 'amt = nd_transaction_amounts([5, 10, 15, 20, 50, 100], \n                          [0.35, 0.25, 0.15, 0.1, 0.1, 0.05])\namounts = [next(amt) for i in range(40000)]\n\nsource = pd.DataFrame({"amounts": amounts})\n\nalt.Chart(source).mark_bar().encode(\n    alt.X("amounts", bin=alt.Bin(maxbins=100)),\n    y=\'count()\',\n)')


# Let's make that transaction-amount generator deterministic.

# In[ ]:


import time

def transaction_amounts(means, percentages, seed=None):
    def mkgamma(m, rng):
        while True:
            yield from stats.gamma.ppf(rng.uniform(size=1024), a=1.1, scale=min(m, 750), loc=m)
    
    if seed is None:
        seed = int(time.time()) & ((1 << 32) - 1)
        
    prng = np.random.default_rng(seed=seed)
    
    distributions = [mkgamma(m * 100, prng) for m in means]
    
    while True:
        streams = [next(d) for d in distributions]
        yield (np.floor(np.choose(prng.choice(len(means), p=percentages), streams)) / 100)


# Finally, we'll sanity-check the deterministic version to make sure it behaves the same way as the non-determinstic version.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\namt = transaction_amounts([5, 10, 15, 20, 50, 100], \n                          [0.35, 0.25, 0.15, 0.1, 0.1, 0.05],\n                          seed=0x12341234)\namounts = [next(amt) for i in range(40000)]\n\nsource = pd.DataFrame({"amounts": amounts})\n\nalt.Chart(source).mark_bar().interactive().encode(\n    alt.X("amounts", bin=alt.Bin(maxbins=100)),\n    y=\'count()\',\n)')


# Next up, we'll make a generator to create the entry types.  
# 
# You may have noticed that we need to do some extra work at the beginning of each generator function to make it deterministic -- specifically, something like this:
# 
#     if seed is None:
#         seed = int(time.time()) & ((1 << 32) - 1)
#     
#     prng = np.random.default_rng(seed=seed)
#     
# This code sets up a pseudorandom number generator, either seeded with an explicit value (if supplied) or with the time (if no seed is supplied).  We shouldn't be writing this code more than once, so let's set up some code to automatically add this to every function that will need its own seeded PRNG.  We'll use Python's [decorator](https://www.python.org/dev/peps/pep-0318/) facility for this purpose.

# In[ ]:


def makeprng(func):
    import time
    def call_with_prng(*args, prng=None, seed=None, **kwargs):
        if prng is None:
            if seed is None:
                seed = int(time.time()) & ((1 << 32) - 1)
            prng = np.random.default_rng(seed=seed)
        return func(*args, prng=prng, seed=seed, **kwargs)
    return call_with_prng


# Here's our decorator in action:

# In[ ]:


@makeprng
def legitimate_entry_types(prng=None, seed=None):    
    entry_types = ["contactless", "chip_and_pin", "swipe", "manual", "online"]
    entry_probs = [0.25,0.2,0.15,0.05,0.35]

    while True:
        yield entry_types[prng.choice(len(entry_types), p=entry_probs)]


# We'll also declare a simple function to make it easy to generate an instance of a SciPy distribution class with a given seed:

# In[ ]:


def makedist(dist_cls, seed=None, prng=None, **kwargs):
    d = dist_cls(**kwargs)
    d.random_state = (seed and seed) or prng.integers((1 << 32) - 1)
    return d


# ...and one for selecting merchants (primarily a user's favorite merchants):

# In[ ]:


@makeprng
def merchant_stream(common_merchants, all_merchants, fav_percentage=0.2, probs=[0.6,0.37,0.03], prng = None, seed=None):
    favorite_merchants = prng.choice(common_merchants,
                                          size=int(len(common_merchants) * fav_percentage))
    merchants = [favorite_merchants, common_merchants, all_merchants]
    while True:
        pool = merchants[prng.choice(len(merchants), p=probs)]
        yield int(prng.choice(pool))


# We can combine all of these to generate a stream of legitimate activity for a single user:

# In[ ]:


@makeprng
def legitimate_user_stream(user_id, transactions_per_day=12, start_timestamp=0, amount_means=[20,100,500], amount_probs=[0.9,0.075,0.025], prng=None, seed=None, common_merchants=None, merchant_count=20000):
    if common_merchants is None:
        # this case means that "common" merchants are unique to each user
        common_merchants = prng.choice(merchant_count, size=merchant_count // 21)
    
    amounts = transaction_amounts(amount_means, amount_probs, seed=prng.integers((1<<32)-1))
    entry_types = legitimate_entry_types(seed=prng.integers((1<<32)-1))
    merchants = merchant_stream(common_merchants, np.arange(merchant_count), seed=prng.integers((1<<32)-1))
    
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
        


# In[ ]:


sim = simulate([legitimate_user_stream(i, common_merchants=common_merchants) for i in [1,6,9,14]])

for i in range(200):
    if i < 10 or i > 190:
        print(next(sim))
    elif i == 10:
        print("...")


# We can visualize the behavior of users in each time zone to see when they're awake:

# In[ ]:


prng = np.random.default_rng(seed=0xcfe1a77e)
seeds = prng.integers((1<<32) - 1, size=72)

sim = simulate([legitimate_user_stream(i, seed=seeds[i]) for i in range(72)])

results = [(offset, tup[1]) for offset, tup in [next(sim) for _ in range(60000)]]
    
source = pd.DataFrame({"hours": [(t[0] % 86400) // 3600 for t in results], 
                       "tzs": [t[1] % 24 for t in results]})

alt.Chart(source).mark_area().encode(
    alt.X("hours"),
    y='count()', color="tzs"
).interactive()


# ## Simulating fraud
# 
# We'll start with some basic assumptions:  
# 
# 1. fraudulent transactions are equally likely to happen at any arbitrary merchant,
# 2. fraudulent transactions are typically for small dollar amounts,
# 3. fraudulent transactions are rare overall, but when they occur, several will occur close together,
# 4. fraudulent transactions are far more likely to be certain entry types (manual or online) or foreign transactions, and
# 5. fraudulent transactions occur without regard for the user's typical schedule.
# 
# These will guide our design of a fraudulent transaction generator.  We'll simulate parallel and independent streams of legitimate and fraudulent transactions for each user.

# In[ ]:


@makeprng
def fraud_entry_types(prng=None, seed=None):
    entry_types = ["contactless", "chip_and_pin", "swipe", "manual", "online"]
    entry_probs = [0.05,0.05,0.05,0.35,0.5]

    while True:
        yield entry_types[prng.choice(len(entry_types), p=entry_probs)]
        
@makeprng
def fraudulent_user_stream(user_id, transactions_per_day = 12, transactions_per_burst=10, amount_means=[5,10,20], amount_probs=[0.2, 0.2, 0.6], prng=None, seed=None, merchant_count=20000):
    amounts = transaction_amounts(amount_means, amount_probs, seed=prng.integers((1<<32) - 1))
    entry_types = fraud_entry_types(seed=prng.integers((1<<32) - 1))
    
    SECONDS_PER_DAY = 86400
    loc = SECONDS_PER_DAY // (transactions_per_day * transactions_per_burst)
    p = 1 / 10

    poisson = makedist(stats.poisson, prng=prng, mu=transactions_per_burst)
    foldnorm = makedist(stats.foldnorm, prng=prng, c=1.8, loc=SECONDS_PER_DAY * 30, scale=1 << 20)
    geom = makedist(stats.geom, prng=prng, p=p, loc=loc)
    while True:
        # consider also np.floor(stats.gamma.rvs(a=6.4, loc=SECONDS_PER_DAY * 90, scale=SECONDS_PER_DAY, size=1))
        fraud_delay = np.floor(foldnorm.rvs())
        fraud_delay = max(int(fraud_delay), 1)
        
        fraud_count = max(poisson.rvs(), 1)
        
        ams = [next(amounts) for _ in range(fraud_count)]
        ens = [next(entry_types) for _ in range(fraud_count)]
        fs = prng.choice([True, False], p=[0.3, 0.7], size=fraud_count)
        
        m_ids = prng.choice(merchant_count, size=fraud_count)
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


# Let's sanity-check the output of our fraudulent transaction generator.

# In[ ]:


sim = simulate([fraudulent_user_stream(1, seed=123, merchant_count=MERCHANT_COUNT), 
                fraudulent_user_stream(2, seed=456, merchant_count=MERCHANT_COUNT), 
                fraudulent_user_stream(3, seed=789, merchant_count=MERCHANT_COUNT)])

for i in range(200):
    v = next(sim)
    if i < 10 or i > 190:
        print(v)
    elif i == 10:
        print("...")


# In[ ]:


sim = simulate([legitimate_user_stream(1, seed=123, common_merchants=common_merchants, merchant_count=MERCHANT_COUNT), 
                legitimate_user_stream(2, seed=456, common_merchants=common_merchants, merchant_count=MERCHANT_COUNT), 
                legitimate_user_stream(3, seed=789, common_merchants=common_merchants, merchant_count=MERCHANT_COUNT),
                fraudulent_user_stream(1, seed=321, merchant_count=MERCHANT_COUNT), 
                fraudulent_user_stream(2, seed=654, merchant_count=MERCHANT_COUNT), 
                fraudulent_user_stream(3, seed=987, merchant_count=MERCHANT_COUNT)])
count = 0
STEPS = 50000

for i in range(STEPS):
    result = next(sim)
    if result[1][0] == 'fraud':
        count += 1

print("%.02f%% of transactions were fraudulent" % (count / STEPS * 100))


# # Generating a file of synthetic transactions
# 
# 

# In[ ]:


import time
import itertools

def setup(user_count = 50000, merchant_count = 80000, common_fraction = 21, seed = None):
    # a small percentage of merchants account for most transactions
    common_merchant_count = merchant_count // common_fraction
    prng = np.random.default_rng(seed=(seed or 0xda7aba5e))

    common_merchants = prng.choice(merchant_count,
                                   size=common_merchant_count,
                                   replace=True)

    legitimate_user_seeds = prng.integers(1<<32, size=user_count)
    fraud_user_seeds = prng.integers(1<<32, size=user_count)
    
    legitimate_streams = [legitimate_user_stream(uid, seed=seed, transactions_per_day=5 + prng.integers(13),
                                                 common_merchants=common_merchants, 
                                                 merchant_count=merchant_count,
                                                 amount_means=[3 * (prng.integers(7) + 1), 
                                                               5 * (prng.integers(20) + 4), 
                                                               20 * (prng.integers(35) + 4)]) 
                          for uid, seed in enumerate(legitimate_user_seeds)]
    fraud_streams = [fraudulent_user_stream(uid, seed=seed, merchant_count=merchant_count) 
                     for uid, seed in enumerate(fraud_user_seeds) if prng.integers(10) < 6]
    print("%d legitimate users and %d fraud streams" % (len(legitimate_streams), len(fraud_streams)))
    return simulate(itertools.chain(legitimate_streams, fraud_streams), initial_time=int(time.time()))


# In[ ]:


RECORD_COUNT = 500000000
simulation = setup(seed=0x20200213)

with open("fraud-huge.csv", "w") as f:
    f.write("timestamp,label,user_id,amount,merchant_id,trans_type,foreign\n")
    for i in range(RECORD_COUNT):
        v = next(simulation)
        if i % 10000 == 0:
            print("generating record %d" % i)
        f.write(("%d," % v[0]) + ",".join([str(val) for val in v[1]]) + "\n")


# # Saving to parquet
# 
# 

# In[ ]:


pd.read_csv("fraud-huge.csv").to_parquet("fraud-huge-all.parquet")


# In[ ]:




