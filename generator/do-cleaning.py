#!/usr/bin/env python
# coding: utf-8

# Copyright (c) 2020–2021, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import re
import json

options = {}

WORKLOAD_VERSION = '0.1'

def register_options(**kwargs):
    global options
    for k, v in kwargs.items():
        options[k] = v

def _register_session(s):
    global session
    session = s

def read_df(session, fn):
    global options
    kwargs = {}
    input_kind = options["input_kind"]

    if input_kind == "csv":
        kwargs["header"] = True
    return getattr(session.read, input_kind)("%s.%s" % (fn, input_kind), **kwargs)

def write_df(df, name):
    global options
    output_kind = options["output_kind"]
    output_mode = options["output_mode"]
    output_prefix = options["output_prefix"]
    
    name = "%s.%s" % (name, output_kind)
    if output_prefix != "":
        name = "%s%s" % (output_prefix, name)
    kwargs = {}
    if output_kind == "csv":
        kwargs["header"] = True
    getattr(df.write.mode(output_mode), output_kind)(name, **kwargs)

app_name = "fraud-output"

default_output_file = "fraud-output-"
default_output_prefix = ""
default_input_prefix = ""
default_output_mode = "overwrite"
default_output_kind = "parquet"
default_input_kind = "parquet"

parser = parser = argparse.ArgumentParser()
parser.add_argument('--output-file', help='base name for preprocessed output data (default="%s"; "interarrival" or "cleaned" will be appended)' % default_output_file, default=default_output_file)
parser.add_argument('--output-mode', help='Spark data source output mode for the result (default: overwrite)', default=default_output_mode)
parser.add_argument('--input-file', help='default input file (e.g., "fraud-raw.parquet"; the default is empty)', default="")
parser.add_argument('--output-prefix', help='text to prepend to every output file basename (e.g., "hdfs:///fraud-processed-data"; the default is empty)', default=default_output_prefix)
parser.add_argument('--output-kind', help='output Spark data source type for the result (default: parquet)', default=default_output_kind)
parser.add_argument('--input-kind', help='Spark data source type for the input (default: parquet)', default=default_input_kind)
parser.add_argument('--summary-prefix', help='text to prepend to analytic reports (e.g., "reports/"; default is empty)', default='')
parser.add_argument('--report-file', help='location in which to store a performance report', default='report.txt')
parser.add_argument('--log-level', help='set log level (default: OFF)', default="OFF")
parser.add_argument('--coalesce-output', help='coalesce output to NUM partitions', default=0, type=int)

if __name__ == '__main__':
    import pyspark
    import os

    failed = False
    
    args = parser.parse_args()

    session = pyspark.sql.SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

    session.sparkContext.setLogLevel(args.log_level)

    session

    register_options(
        app_name = app_name,
        output_prefix = args.output_prefix,
        output_mode = args.output_mode,
        output_kind = args.output_kind,
        input_kind = args.input_kind,
        output_file = args.output_file,
        coalesce_output = args.coalesce_output,
    )


    import timeit
    
    import pyspark.sql.window as W
    import pyspark.sql.functions as F

    df = read_df(session, args.input_file)
    tt_spec = W.Window.partitionBy(F.lit("")).orderBy("trans_type")
    interarrival_spec = W.Window.partitionBy("user_id").orderBy("timestamp")
    overall_spec = W.Window.orderBy("timestamp")

    trans_types = df.select("trans_type").distinct().select(F.row_number().over(tt_spec).alias("index"), "trans_type")
    df = df.join(trans_types, "trans_type").withColumn("trans_type_index", F.col("index")).drop("trans_type", "index")
    df = df.withColumn("amount", df["amount"].cast("float"))

    df_interarrival = df.withColumn(
        "previous_timestamp", 
        F.lag(df["timestamp"]).over(
            interarrival_spec
        )
    ).withColumn(
        "interarrival",
        (F.col("timestamp") - F.col("previous_timestamp")).cast("int")
    )

    session.conf.set("spark.rapids.sql.castFloatToIntegralTypes.enabled", True)

    amount_cents = (F.col("amount") * 100).cast("int")

    rollingUserSpec = W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        -1000,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )


    userSpec = W.Window.partitionBy("user_id").orderBy(
        amount_cents
    )

    toPresentUserSpec = W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        W.Window.unboundedPreceding,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )

    rollingUserSpec = W.Window.partitionBy("user_id").orderBy(
        F.col("timestamp")
    ).rowsBetween(
        -1000,
        W.Window.currentRow
    ).orderBy(
        amount_cents
    )

    overallSpec = W.Window.orderBy(
        amount_cents
    )

    df_dist = df_interarrival.withColumn("amount_rank_user",
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

    interarrival_calc = timeit.timeit(lambda: write_df(df_interarrival, "%s-interarrival" % options["output_file"]), number=1)
    quantile_calc = timeit.timeit(lambda: write_df(df_out, "%s-cleaned" % options["output_file"]), number=1)

    first_line = "Completed payments fraud transaction preprocessing (version %s; %d transactions) \n" % (WORKLOAD_VERSION, df.count())

    first_line += 'Calculating interarrival times took %.02f seconds\n' % interarrival_calc
    first_line += 'Calculating transaction percentiles took %.02f seconds; configuration follows:\n\n' % quantile_calc
    print(first_line)

    with open(args.report_file, "w") as report:
        report.write(first_line + "\n")
        for conf in session.sparkContext.getConf().getAll():
            report.write(str(conf) + "\n")
            print(conf)
    
    session.stop()    



