#!/usr/bin/env python
# coding: utf-8

# # Loading data

# In[1]:


from pyspark.sql import SparkSession
session = SparkSession.builder.getOrCreate()

import time, timeit
output_ts = int(time.time())

df = session.read.parquet("fraud-256m-partitioned2.parquet")


# ## Cleaning data
# 
# These data are mostly clean but we need to add a new field for transaction interarrival time.

# In[2]:


df.printSchema()


# In[3]:


import pyspark.sql.window as W
import pyspark.sql.functions as F

interarrival_spec = W.Window.partitionBy("user_id").orderBy("timestamp")
overall_spec = W.Window.orderBy("timestamp")

df_interarrival = df.withColumn(
    "previous_timestamp", 
    F.lag(df["timestamp"]).over(
        interarrival_spec
    )
).withColumn(
    "interarrival",
    (F.col("timestamp") - F.col("previous_timestamp")).cast("int")
)


# In[8]:


if False:

    split_point = int(df_interarrival.count() * 0.7)

    df_interarrival_split = df_interarrival.withColumn(
        "amount_rank_user_rolling",
        (F.rank().over(rollingUserSpec) / 
         F.count("user_id").over(rollingUserSpec)).cast("float") 
    ).withColumn("transactions_in_rolling_window",
        F.count("user_id").over(rollingUserSpec)   
    )
    
    df_interarrival_train = df_interarrival_split.where(F.col("observation_number") <= split_point)
    df_interarrival_test = df_interarrival_split.where(F.col("observation_number") > split_point)


# In[9]:


# never computed; an option for comparison

if False:
    df_dist_unused = df_interarrival_train.        withColumn("amount_quantile",
            F.cume_dist().over(
                W.Window.partitionBy("user_id").orderBy("amount")
            )
        )


# In[14]:


session.conf.set("spark.rapids.sql.castFloatToIntegralTypes.enabled", True)

amount_cents = (F.col("amount") * 100).cast("int")

rollingUserSpec =     W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        -1000,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )


userSpec =     W.Window.partitionBy("user_id").orderBy(
        amount_cents
    )

toPresentUserSpec =     W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        W.Window.unboundedPreceding,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )


rollingUserSpec =     W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        -1000,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )


overallSpec =     W.Window.orderBy(
        amount_cents
    )

# not identical to cume_dist; this rank is the fraction of 
# transactions that are strictly less than the current row

# XXX:  need to censor overall and user-overall quantiles with train/test split

df_dist = df_interarrival.    withColumn("amount_rank_user",
        (F.rank().over(userSpec) / 
         F.count("user_id").over(userSpec)).cast("float")
    ).withColumn("amount_rank_overall",
         (F.rank().over(overallSpec) / 
         F.count("user_id").over(overallSpec)).cast("float")       
    ).withColumn("amount_rank_user_to_present",
        (F.rank().over(toPresentUserSpec) / 
         F.count("user_id").over(toPresentUserSpec)).cast("float") 
    ).withColumn("user_transaction_count_to_present",
         F.count("user_id").over(toPresentUserSpec)   
    )


# In[13]:


df_dist.printSchema()


# In[ ]:


df_out = df_dist.drop(
    "previous_timestamp"
).withColumn(
    "amount", 
    F.col("amount").cast("float")
).withColumn(
    "user_id", 
    F.col("user_id").cast("int")
).withColumn(
    "merchant_id", 
    F.col("merchant_id").cast("int")
)


# In[ ]:


interarrival_calc = timeit.timeit(lambda: df_interarrival.write.parquet(f"fraud-interarrival-{output_ts}.parquet"), number=1)
quantile_calc = timeit.timeit(lambda: df_out.write.parquet(f"fraud-cleaned-{output_ts}.parquet"), number=1)


# In[ ]:


df_out.sample(fraction=0.05).write.parquet(f"fraud-cleaned-{output_ts}-sample.parquet")


# In[ ]:


print(f"time to compute interarrivals:  {interarrival_calc}")
print(f"time to compute quantiles:  {quantile_calc}")

