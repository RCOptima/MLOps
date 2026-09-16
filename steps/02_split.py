import logging
import argparse
import yaml
from sklearn.model_selection import train_test_split

logger = logging.getLogger("split")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def split(ingested_data_path,
          train_table,
          test_table,
          val_table,
          cfg):
        
    split_ratios = cfg['split_ratios']
    random_state = cfg.get("random_state", None)
    if random_state:
        random_state = int(random_state)
    shuffle = cfg.get("shuffle") # bool?
    validate = cfg.get("validate", True)

    pdf = spark.table(ingested_data_path).toPandas()
    split = [float(x) for x in split_ratios.split(',')]

    if len(split) not in [2, 3]:
        raise ValueError(f"Split ratios have to be 2 or 3, this is {len(split)}")
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"Split doesn't add up to 1. It adds to {sum(split)}")

    if validate and len(split) == 3:
        logger.info(f'Validation is enabled, splitting the df as '
                    f'{split[0] * 100}% - train, {split[1] * 100}% validation'
                    f' and {split[2] * 100}% test')
        val_size = split[1]
        test_size = split[2]

        train_df, test_val_df = train_test_split(
            pdf,
            test_size=test_size + val_size,
            random_state=random_state,
            shuffle=shuffle
        )

        val_df, test_df = train_test_split(
            test_val_df,
            test_size=val_size / (val_size + test_size),
            random_state=random_state,
            shuffle=shuffle
        ) # test here to make sure the split works properly
        (
            spark.createDataFrame(val_df)
            .write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(f"{val_table}")
        )
    else:
        if len(split) == 3:
            logger.warning(f"Validate is set to False but data is split in 3 - {split_ratios}, using default 0.8,0.2")
            split = [0.8, 0.2]

        test_size = split[1]
        logger.info(f'Validate is disabled, splitting the df as {split}')
        train_df, test_df = train_test_split(
            pdf,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle
        )
    (
        spark.createDataFrame(test_df)
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{test_table}")
    )

    (
        spark.createDataFrame(train_df)
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{train_table}")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingested_data_path", required=True)
    parser.add_argument("--train_table", required=True)
    parser.add_argument("--test_table", required=True)
    parser.add_argument("--val_table", required=False, default=None)
    args = parser.parse_args()

    with open("configs/02_split_cfg.yml") as f:
        cfg = yaml.safe_load(f)

    split(
        args.ingested_data_path,
        args.train_table,
        args.test_table,
        args.val_table,
        cfg
    )