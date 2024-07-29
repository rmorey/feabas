from pathlib import Path
from typing import NamedTuple
import pandas as pd
import numpy as np
import cv2
import os

from feabas import config

DEFAULT_SUBTILE_OVERLAP = 0.2
DEFAULT_SUPERTILE_OVERLAP = 0.2

# Generate supertile map from stage_positions.csv
def generate_supertile_map(section_path):
    csv_path = Path(section_path) / 'metadata' / 'stage_positions.csv'
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(str(csv_path))
    
    # Correct the column names by removing leading spaces
    df.columns = [col.strip() for col in df.columns]
    
    # Normalize the stage_x_nm and stage_y_nm values
    df['norm_stage_x'] = df['stage_x_nm'].rank(method='dense').astype(int) - 1
    df['norm_stage_y'] = df['stage_y_nm'].rank(method='dense').astype(int) - 1
    
    # Determine the dimensions of the 2D array
    max_x = df['norm_stage_x'].max()
    max_y = df['norm_stage_y'].max()
    
    # Initialize an array of shape (max_y+1, max_x+1) with None values
    arr = np.full((max_y+1, max_x+1), None)
    
    # Populate the array with tile_id values using numpy's advanced indexing
    arr[df['norm_stage_y'].values, df['norm_stage_x'].values] = df['tile_id'].values
    
    # Reverse the order of the rows in the array
    supertile_map = arr[::-1]
    
    return supertile_map

def generate_tile_id_map(supertile_map):

    # Cricket subtile order
    SUBTILE_MAP = [[6, 7, 8], [5, 0, 1], [4, 3, 2]]

    tile_id_map = []
    for supertile_row in supertile_map:
        for subtile_row in SUBTILE_MAP: 
            current_row = []
            for supertile in supertile_row:
                if supertile is not None:
                    for subtile in subtile_row:
                        current_row.append(f"{supertile:04}_{subtile}")
            tile_id_map.append(current_row)
    return np.array(tile_id_map)



def get_subtile_pos(supertile_map, subtile_size, subtile_overlap=DEFAULT_SUBTILE_OVERLAP, supertile_overlap=DEFAULT_SUPERTILE_OVERLAP):

    supertile_pos = {}
    supertile_x, supertile_y = 0, 0
    supertile_size = 3 * subtile_size - 2 * subtile_size * subtile_overlap

    for row in supertile_map:
        for supertile in row:
            supertile_pos[supertile] = (supertile_x, supertile_y)
            supertile_x += supertile_size * (1 - supertile_overlap)
        supertile_y += supertile_size * (1 - supertile_overlap)
        supertile_x = 0


    SUBTILE_ID_TO_XY = {
        0: (1,1),
        1: (2,1),
        2: (2,2),
        3: (1,2),
        4: (0,2),
        5: (0,1),
        6: (0,0),
        7: (1,0),
        8: (2,0)
    }

    tile_id_map = generate_tile_id_map(supertile_map)

    subtile_pos = {}
    for row in tile_id_map:
        for tile in row:
            supertile, subtile_y = tile.split('_')
            supertile_x, supertile_y = supertile_pos[int(supertile)]
            dx, dy = SUBTILE_ID_TO_XY[int(subtile_y)]
            subtile_x = int(supertile_x + dx * subtile_size * (1 - subtile_overlap))
            subtile_y = int(supertile_y + dy * subtile_size * (1 - subtile_overlap))
            subtile_pos[tile] = (subtile_x, subtile_y)
     
    return subtile_pos


class StitchConfig(NamedTuple):
    """Configuration for stitching a single TEM section"""

    section_dir: str
    """Path to the section directory as a string, e.g., '/scratch/tem-data/bladeseq-2024.07.02-11.01.33/s013-2024.07.02-11.01.33'"""

    resolution: int
    """Resolution in nanometers, e.g., 4"""

    subtile_size: int
    """Size of subtiles in pixels, e.g., 6000"""

    subtile_overlap: float
    """Fraction of overlap between subtiles, e.g., 0.08"""

    supertile_overlap: float
    """Fraction of overlap between supertiles, e.g., 0.06"""

    file_ext: str
    """File extension for image files, e.g., 'bmp'"""

def gen_stitch_coords(config: StitchConfig, output_file: str):
    """
    Generate stitch coordinates based on the given configuration and save them to a file.

    Args:
        config (StitchConfig): The configuration object containing the stitch parameters.
        output_file (str): The path to the output file where the stitch coordinates will be saved.

    Returns:
        None
    """
    section_dir = Path(config.section_dir)

    tile_root_dir = section_dir / 'subtiles'

    supertile_map = generate_supertile_map(section_dir)
    tile_coordinates = get_subtile_pos(supertile_map, config.subtile_size, config.subtile_overlap, config.supertile_overlap)
    
    # File content preparation
    file_content = [
        "{ROOT_DIR}\t" + f"{tile_root_dir}",
        "{RESOLUTION}\t" + str(config.resolution),
        "{TILE_SIZE}\t" + "\t".join(map(str, (config.subtile_size, config.subtile_size))),
    ]
    file_content.extend([f"tile_{tile_id}.{config.file_ext}\t{coord_x}\t{coord_y}" for tile_id, (coord_x, coord_y) in tile_coordinates.items()])
    
    # Joining content into a single string
    file_content_str = "\n".join(file_content)
    
    # Writing the content to the file
    with open(output_file, 'w') as file:
        file.write(file_content_str)


# /scratch/zhihaozheng/mec/acqs/3-complete/Part1_reel1068_blade2_20231010/bladeseq-2023.10.10-10.14.43/s3429-2023.10.10-10.14.43/metadata/stage_positions.csv

def get_image_dimension(image_path):
    image = cv2.imread(image_path)
    if image is not None:
        height, width, _ = image.shape
        assert width == height
        return width
    else:
        raise ValueError(f"Failed to load image at path: {image_path}")

def make_stitch_coord_from_local_blade_path(blade_path: str|Path, stitch_coord_path: str|Path):

    blade_path = Path(blade_path)
    if not blade_path.exists():
        raise ValueError(f"{blade_path} does not exist.")
    
    if blade_path.name.contains("bladeseq"):
        contents = os.listdir(blade_path)
        if len(contents) != 1:
            raise ValueError(f"Expected one directory in {blade_path}, but found {contents}.")
        blade_path = blade_path / contents[0]
   
    stage_positions_path = blade_path / 'metadata/stage_positions.csv'
    if not stage_positions_path.exists():
        raise ValueError(f"Path {stage_positions_path} does not exist.")
    
    stitch_coord_path = Path(stitch_coord_path)
    if os.path.exists(stitch_coord_path):
        raise ValueError(f"{stitch_coord_path} already exists.")
    
    some_tile = os.listdir(blade_path / 'subtiles')[0]
    subtile_size = get_image_dimension(blade_path / 'subtiles' / some_tile)
    file_ext = some_tile.split('.')[-1]

    stitch_config = StitchConfig(
        section_dir=blade_path,
        resolution=config.data_resolution(),
        subtile_size=subtile_size,
        subtile_overlap=DEFAULT_SUBTILE_OVERLAP, # todo: configurable
        supertile_overlap=DEFAULT_SUPERTILE_OVERLAP,
        file_ext=file_ext
    )

    gen_stitch_coords(stitch_config, stitch_coord_path)