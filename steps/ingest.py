from pyspark.sql import SparkSession

def ingest(spark, table_name):
    """Load a Unity Catalog table as a pandas DataFrame."""
    # spark = SparkSession.builder.getOrCreate()   
    df = spark.table(table_name).toPandas()
    return df