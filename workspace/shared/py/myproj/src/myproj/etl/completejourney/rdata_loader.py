from abc import ABC, abstractmethod
import os
from pathlib import Path

import rdata

class RDataLoadError(Exception): 
    def __init__(self, msg):
        self._msg = msg

    def __str__(self):
        return self._msg

class RDataLoadStrategy(ABC):
    @abstractmethod
    def load(self, path: Path, logical_name: str):
        pass


class RdaLoadStrategy(RDataLoadStrategy):
    def load(self, path: Path, logical_name: str):
        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)

        # .rda は dict で返る: {"campaigns": DataFrame} のような形
        return converted[logical_name]


class RdsLoadStrategy(RDataLoadStrategy):
    def load(self, path: Path, logical_name: str):
        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)

        # .rds はオブジェクト自体、今回なら DataFrame が返る
        return converted


class RDataLoader:
    def __init__(self, strategy: RDataLoadStrategy):
        self._strategy = strategy

    def load(self, file_path: Path, logical_name: str):
        return self._strategy.load(file_path, logical_name)


def build_rdata_loader(file_type: str) -> RDataLoader:
    if file_type == "rda":
        return RDataLoader(RdaLoadStrategy())
    if file_type == "rds":
        return RDataLoader(RdsLoadStrategy())
    raise RDataLoadError(f"拡張子は rda または rds を指定してください: {file_type}")


class ExecuteRDataLoader: 
    def __init__(self, path_project_base: Path | None = None): 
        self.path_project_base = path_project_base or Path("/loc0/bigbrother/repositories/0015_zenn/workspace")
        self.path_raw = self.path_project_base / "shared/data/00_raw/completejourney/rdata"
        self.path_interim = self.path_project_base / "shared/data/10_interim/completejourney"

    def build_file_path(self, logical_name: str, file_type: str) -> Path:
        return self.path_raw / f"{logical_name}.{file_type}"

    def load_dfs(self, data_names: dict): 
        dfs = {}
        for name, file_type in data_names.items():
            print(name, file_type)
            file_path = self.build_file_path(name, file_type)
            loader = build_rdata_loader(file_type)
            dfs[name] = loader.load(file_path, name)
        
        return dfs

    def write_down_dfs(self, dfs): 
        os.makedirs(self.path_interim, exist_ok=True)

        for k, v in dfs.items(): 
            v.to_parquet(self.path_interim / f"{k}.parquet")

if __name__ == "__main__":
    erdl = ExecuteRDataLoader()

    data_names = {
            "campaign_descriptions": "rda", 
            "campaigns": "rda", 
            "coupon_redemptions": "rda", 
            "coupons": "rda", 
            "demographics": "rda", 
            "products": "rda", 
            "transactions": "rds", 
            "promotions": "rds"
    }
    dfs = erdl.load_dfs(data_names)
    
    erdl.write_down_dfs(dfs)
