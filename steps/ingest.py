
def ingest(table_name):
    df = spark.table(table_name).toPandas()
    return df