from pandas import read_csv
from skimage.io import imread
import numpy as np
from pandarallel import pandarallel
from tqdm import tqdm

def read_stitched_imagepositions(txt_path):
    """simple helper function to read stitched position from ZEISS multiSEM. 
    Note that the col names is hard coded, which could be risky in the future.

    Parameters
    ----------
    txt_path (Path): Path to *stitched_imagepositions.txt

    Returns
    -------
    image_positions (pandas DF): image positions as a pandas dataframe

    """
    # this is based on ZEISS file headers, might change in the future
    col_names = ["relative_path", "centre_y", "centre_x", "centre_z"]
    image_positions = read_csv(txt_path, header=None, sep='\t', names=col_names)

    return image_positions


def get_bmp_name(rel_bmp_path):
    # simple helper function
    png_p = rel_bmp_path.split('\\')
    return png_p[1]

def get_hexagon(bmp_name):
    # simple helper function
    hex = bmp_name.split('_')
    return hex[1]

def get_region_id(bmp_name):
    # simple helper function
    id = bmp_name.split('_')
    return id[0]

def get_tile_number(bmp_name):
    tile_n = bmp_name.split('_')
    return tile_n[2]

def get_absolute_path(parent_path, relative_path):
    return parent_path.joinpath(relative_path)

def get_info_from_path(positions_df, section_path):
    # getting paths sorted
    positions_df['abs_path'] = positions_df.apply(lambda x: get_absolute_path(section_path, x['relative_path']), axis=1)
    # getting hexagon information
    positions_df['bmp'] = positions_df['relative_path'].apply(get_bmp_name)
    positions_df['id'] = positions_df['bmp'].apply(get_region_id)
    positions_df['hexagon'] = positions_df['bmp'].apply(get_hexagon)
    positions_df['tile_number'] = positions_df['bmp'].apply(get_tile_number)

    return positions_df



def simple_shift(val, shift):
    # simple helper function
    return val-shift

def translation00(df):
    # simple helper function that translates the array to 0,0
    df00 = df.copy()
    min_x = df.min()["corner_x"]
    min_y = df.min()["corner_y"]

    df00['corner_x'] = df00.apply(lambda x: x["corner_x"] - min_x, axis=1)
    df00['centre_x'] = df00.apply(lambda x: x["centre_x"] - min_x, axis=1)
    df00['corner_y'] = df00.apply(lambda x: x["corner_y"] - min_y, axis=1)
    df00['centre_y'] = df00.apply(lambda x: x["centre_y"] - min_y, axis=1)

    return df00

def get_info_from_image(positions_df):
    """We load a single tile, assume all others have same size and calculate tile positions accordingly 
    via the centgre positions"""
    
    img_tile = imread(positions_df['abs_path'][0])

    bmp_x = img_tile.shape[0]
    bmp_y = img_tile.shape[1]

    print(f"'.bmp' img_tile size: {bmp_x}, {bmp_y}, and dtype: {img_tile.dtype}")

    positions_df['size_x'] = bmp_x
    positions_df['size_y'] = bmp_y

    positions_df['corner_x'] = positions_df.apply(lambda x: int(x['centre_x']- x['size_x']/2), axis=1)
    positions_df['corner_y'] = positions_df.apply(lambda x: int(x['centre_y']- x['size_y']/2), axis=1)

    return positions_df


def get_mean(im_path):
    """helper function that gets mean of png tile"""
    im = imread(im_path)
    return np.mean(im).astype(im.dtype)

def get_median(im_path):
    """helper function that gets median of png tile"""
    from skimage.io import imread
    from numpy import median
    # simple helper function
    im = imread(im_path)
    return median(im).astype(im.dtype)

def get_quantile(im_path, q = 0.3):
    """helper function that gets quantile of png tile"""
    from skimage.io import imread
    from numpy import quantile
    # simple helper function
    im = imread(im_path)
    return quantile(im, q).astype(im.dtype)

def get_intensity_correction(positions_df, method='q30'):
    """calculates the intensity correction based on the desired method. 
    The plan is to correct for differences in intensities among tiles"""

    pandarallel.initialize(use_memory_fs=False, progress_bar=True)
    # problems with parallel apply, I have to read more.

    #positions_df['mean_int'] = positions_df.apply(lambda x: get_mean(section_path.joinpath(x['bmp_name'])), axis=1)
    positions_df['median_int'] = positions_df['abs_path'].parallel_apply(get_median)
    positions_df['q0p3_int'] = positions_df['abs_path'].parallel_apply(get_quantile)

    # simple intensity correction
    # full_img_mean = positions_df['mean_int'].mean()
    full_img_med = positions_df['median_int'].median()
    full_img_q = positions_df['q0p3_int'].median()

    print(f'target median: {full_img_med}')

    if method=='q30':
        positions_df['int_corr'] = full_img_q / positions_df['q0p3_int']
    elif method=='median':
        positions_df['int_corr'] = full_img_med / positions_df['median_int']
    else:
        raise('problems with method')
    
    return positions_df


# related to the creation of the zarr array
def optimal_size(current_size, res_levels):
    """helper function that asses the best size given the desired resolution level""" 
    
    div_factor = np.power(2,res_levels)
    rem = np.remainder(current_size, div_factor)

    print(f'current size: {current_size}, factor: {div_factor}, reminder: {rem}')

    if rem > 0:
        extra = div_factor-rem
    else:
        extra = 0

    print(f'we need to add: {extra}, so new size is: {current_size+extra}')

    return current_size+extra

def flat_field_correction(img, g_sigma=10):
    from skimage.filters import gaussian
    from numpy import reshape, hstack, dot, transpose, divide, ones, mgrid
    from numpy.linalg import pinv
    smooth_img = gaussian(img,sigma=g_sigma,preserve_range=True)
    xdim = img.shape[0]
    ydim = img.shape[1]
    npix = xdim*ydim

    XX, YY = mgrid[:xdim, :ydim]

    X = reshape(XX, (npix,1))
    Y = reshape(YY, (npix,1))
    ZYX = hstack((ones((npix, 1)), Y, X))

    ZZ = reshape(smooth_img, (npix,1))

    theta = dot(dot( pinv(dot(ZYX.transpose(), ZYX)), ZYX.transpose() ), ZZ)

    plane = reshape(dot(ZYX, theta), img.shape)
    
    plane = divide(plane, plane.mean())

    img_corr = divide(img.astype(float), plane).astype(img.dtype)

    return img_corr