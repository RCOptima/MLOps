from abc import ABC, abstractmethod
import pandas as pd

class BaseTransformer(ABC):
    """
    Every dataset-specific transformer subclasses this and implements
    fit() and transform(). Everything else (save/load/CLI/logging)
    is handled once, here, for all of them.
    """
 
    # Bump this if a transformer's output shape/logic changes in a way
    # that would make old fitted artifacts incompatible.
    version: str
 
    @abstractmethod
    def fit(self, df: pd.DataFrame, y=None) -> "BaseTransformer":
        """Learn any parameters needed (e.g. imputation values, encoders)
        from the training split ONLY. Must return self."""
        raise NotImplementedError
 
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted transform to a dataframe (train,
        val, test, or future production data) and return the result."""
        raise NotImplementedError
 
    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df).transform(df)


# example
class Transformer(BaseTransformer):
    """Median-impute a numeric column, one-hot encode a categorical
    column, derive a feature."""
 
    version = "1.0"
 
    def __init__(self, numeric_col: str = "monthly_spend", cat_col: str = "plan_type"):
        pass
        # self.numeric_col = numeric_col
        # self.cat_col = cat_col
        # self._median: float | None = None
        # self._categories: list[str] | None = None
 
    def fit(self, df: pd.DataFrame, y=None) -> "Transformer":
        # self._median = df[self.numeric_col].median()
        # self._categories = sorted(df[self.cat_col].dropna().unique().tolist())
        return self
 
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
        # if self._median is None or self._categories is None:
        #     raise RuntimeError("Transformer must be fit() before transform().")
 
        # out = df.copy()
        # out[self.numeric_col] = out[self.numeric_col].fillna(self._median)
 
        # for cat in self._categories:
        #     out[f"{self.cat_col}_{cat}"] = (out[self.cat_col] == cat).astype(int)
        # out = out.drop(columns=[self.cat_col])
 
        # out["spend_is_high"] = (out[self.numeric_col] > self._median * 2).astype(int)
        # return out