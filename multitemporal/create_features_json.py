import json
from pathlib import Path


def make_features_json(tile_id: str, tmp_dir: str) -> None:
    """"
    Tile id is the current tile we are processing
    input_dir is the temporary dir to save the files to
    """
    output_dict = {
        "projname": "mt_brazil",
        "projdir": f"{tmp_dir}/data",
        "outdir": f"{tmp_dir}/result",
        "dperframe": 1,
        "sources":
            [
                {
                    "name": "evi2",
                    "regexp": f"^veg_indices_evi2_modis_(\\d{8})_{tile_id}.tif$",
                    "bandnum": 1,
                    "scale": 0.0001
                },
                {
                    "name": "mask",
                    "regexp": f"^br_cropmask_(\\d{8})_{tile_id}.tif$",
                    "bandnum": 1
                }
            ],
        "steps":
            [
                {
                    "module": "interpolate",
                    "params": [],
                    "inputs": "evi2",
                    "output": False
                },
                {
                    "module": "shifttime",
                    "params": [244],
                    "inputs": "interpolate",
                    "output": False
                },
                {
                    "name": "evi_season",
                    "module": "trimyr",
                    "params": [1, 1],
                    "inputs": "shifttime",
                    "output": False
                },
                {
                    "name": "mask_season",
                    "module": "trimyr",
                    "params": [2, 2],
                    "inputs": "mask",
                    "output": False
                },
                {
                    "module": "features",
                    "params": [0.5],
                    "inputs": ["evi_season", "mask_season"],
                    "output": True
                }
            ]
    }

    output_path = Path(f'{tmp_dir}/features.json')
    with output_path.open('w') as fp:
        json.dump(output_dict, fp)
