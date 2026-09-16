from pyspark.sql import SparkSession
from abc import ABC, abstractmethod
import argparse
import pandas as pd
import yaml
import sys
from datetime import datetime, timezone
from trainers.transformer import Transformer
import logging

logger = logging.getLogger("transform")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def validate_transformer_class(train_df, val_df, test_df, transformer):
    required_methods = ["fit_transform", "transform"]
    missing = [m for m in required_methods if not hasattr(transformer, m) or not callable(getattr(transformer, m))]
    if missing:
        raise AttributeError(
            f"{type(transformer).__name__} is missing required method(s): {', '.join(missing)}. "
            f"Transformer classes must implement {required_methods} as functions of the class "
            f" 'Transformer()' to be used in this pipeline."
        )
    
    train_transformed_df = transformer.fit_transform(train_df)
    val_transformed_df = transformer.transform(val_df)
    test_transformed_df = transformer.transform(test_df)

    transformed_dfs = [("train", train_transformed_df),
                       ("val", val_transformed_df),
                       ("test", test_transformed_df)]

    for name, df in transformed_dfs:
        if df is None:
            raise ValueError(f"transform on {name} returned None")
        if hasattr(df, "shape"):
            if df.shape[0] == 0:
                raise ValueError(f"transform on {name} returned 0 rows")
            if df.shape[1] == 0:
                raise ValueError(f"transform on {name} returned 0 columns")
        elif len(df) == 0:
            raise ValueError(f"transform on {name} returned an empty result")

    return train_transformed_df , val_transformed_df, test_transformed_df

def transform(train_table, val_table, test_table, cfg):
    train_df = spark.table(train_table).toPandas()
    val_df = spark.table(val_table).toPandas()
    test_df = spark.table(test_table).toPandas()
    transformer = Transformer()

    train_df_transformed, val_df_transformed, test_df_transformed = validate_transformer_class(train_df, val_df, test_df, transformer)

    # validate = True # config this
        # if validate:#
    # I think the transform step should be more of a validation step and the actual transformation actually happens in train.py
        # train_df_transformed = spark.createDataFrame(transformer.fit_transform(train_df))
        # val_df_transformed = spark.createDataFrame(transformer.transform(val_df))
        # test_df_transformed = spark.createDataFrame(transformer.transform(test_df))

    # for df, table in [
    #     (train_df_transformed, train_table),
    #     (val_df_transformed, val_table),
    #     (test_df_transformed, test_table),
    # ]:
        
    #     df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(str(table))

    #     spark.sql(f"""
    #         ALTER TABLE {table}
    #         SET TBLPROPERTIES (
    #             'data_transformed' = 'true',
    #             'transformer' = 'Transformer',
    #             'transformed_at' = '{datetime.now(timezone.utc).isoformat()}'
    #         )
    #     """) # i think we should add tags here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_table", required=True)
    parser.add_argument("--val_table", required=True)
    parser.add_argument("--test_table", required=True)

    with open("configs/03_transform_cfg.yml") as f:
        cfg = yaml.safe_load(f)
    args = parser.parse_args()

    transform(args.train_table, args.val_table, args.test_table, cfg)
