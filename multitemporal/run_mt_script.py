import glob
import json
import tempfile
from pathlib import Path
import pandas as pd

import click

from telluslabs.s3 import S3Path

from multitemporal.create_features_json import make_features_json
from multitemporal.mt import main as mt
import sys


def run_tile(tmp_dir: str):
    make_features_json(tmp_dir=tmp_dir)
    # call the cli command to do the work
    # this is pretty hacky way to get it to work
    sys.argv = ['', '--nproc', '6', '--ymd', '--nongips', '--conf', f'{tmp_dir}/features.json']
    mt()


def download_files(year: int, tile_id: str, tmp_dir: str):
    inventory_s3 = S3Path(bucket='tl-octopus', key=f'raw/inventory/MCD43A4_{year - 1}0901_{year}0901_evi2.json')
    tile_id_split = tile_id.split('_')
    new_tile_id = f'h{int(tile_id_split[0]):2d}_v{int(tile_id_split[1]):2d}'
    mask_s3_path = S3Path(bucket='tl-octopus', key=f'raw/raster/br_crop_mask')

    input_path = Path(tmp_dir) / 'data'
    output_path = Path(tmp_dir) / 'result'
    # create the inputs if they dont exist
    input_path.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)

    print(f'processing {tile_id} {year}')
    with tempfile.NamedTemporaryFile() as tmpfile:
        inventory_s3.download_to(Path(tmpfile.name))
        inventory = json.load(open(tmpfile.name))

    paths = {}
    for item in inventory['evi2']['dates']:
        date = item['date']
        for p, exists in zip(item['paths'], item['exists']):
            if exists and p.endswith(f'_{tile_id}.tif'):
                paths[date] = p

    if len(paths) == 0:
        raise Exception(f'No data for {tile_id}')

    paths = pd.DataFrame(pd.Series(paths, name='path')).sort_index().reset_index()
    for idx, row in paths.iterrows():
        s3path = S3Path.from_str(row['path'])
        print(f'downloading {s3path} to {input_path}')
        s3path.download_to(input_path)

    # also download masks to download location - mt needs two mask years even if only using the last one
    for y in range(year - 1, year + 1):
        input_mask_path = mask_s3_path.join(str(y), f'{new_tile_id}.tif')
        local_path = input_path / f'br_cropmask_{y}0101_{tile_id}.tif'
        input_mask_path.download_to(local_path)

    model_file_path = S3Path.from_str('s3://tl-octopus/user/damien/brazil_crop_mask/'
                                      'rf_2004-2016_varsel_0-7samp_255trees_20md_mfsqrt.sav')
    model_file_path.download_to(Path(tmp_dir))

    print('downloading finished. files downloaded are here: ')
    for f in glob.glob(f'{input_path}/*'):
        print(f)


def upload_outputs(year: int, tile_id: str, tmp_dir: str):
    output_dir = Path(tmp_dir) / 'result'
    output_file = output_dir.joinpath(f'mt_brazil_classify_brazil.tif')
    s3_path = S3Path(bucket='tl-octopus', key=f'raw/raster/br_crop_type/{year}/{tile_id}.tif')
    s3_path.upload_path(output_file)


@click.command('run_mt')
@click.option('--tile_id', help="Tile id str should be in form (12_10) for tile h12v10.", required=True, type=str)
@click.option('--year', help="Harvest year to process", required=True, type=int)
def run_mt(tile_id: str, year: int):
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_files(year=year, tile_id=tile_id, tmp_dir=tmp_dir)
        run_tile(tmp_dir=tmp_dir)
        upload_outputs(year=year, tile_id=tile_id, tmp_dir=tmp_dir)


if __name__ == '__main__':
    run_mt()
