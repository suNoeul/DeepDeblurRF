import os
import shutil
import subprocess

####################################################################
########     COLMAP WRAPPER FOR DeepDeblurRF (CVPR 2025)     #######
####################################################################

def run_colmap(basedir, match_type):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    database_path = os.path.join(basedir, 'database.db')
    image_path = os.path.join(basedir, 'images')
    sparse_path = os.path.join(basedir, 'sparse')
    os.makedirs(sparse_path, exist_ok=True)

    # --- Step 1: Feature extraction ---
    feature_extractor_args = [
        'xvfb-run', '-a',
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_path,
        '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
        '--ImageReader.single_camera', '1',
        '--SiftExtraction.use_gpu', '0'                     # Forced to CPU mode and run without OpenGL
    ]
    feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True)
    logfile.write(feat_output)
    print('[COLMAP] Features extracted.')

    # --- Step 2: Feature matching (guided matching) ---
    exhaustive_matcher_args = [
        'xvfb-run', '-a',
        'colmap', match_type,
        '--database_path', database_path,
        '--SiftMatching.guided_matching', '1',
        '--SiftMatching.use_gpu', '0'
    ]
    match_output = subprocess.check_output(exhaustive_matcher_args, universal_newlines=True)
    logfile.write(match_output)
    print('[COLMAP] Features matched.')

    # --- Step 3: Sparse map reconstruction --- 
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', image_path,
        '--output_path', sparse_path,
        '--Mapper.min_num_matches', '16',
        '--Mapper.tri_min_angle', '1.5',
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
    ]
    map_output = subprocess.check_output(mapper_args, universal_newlines=True)
    logfile.write(map_output)
    logfile.close()    
    print('[COLMAP] Sparse map created.')
    print(f'[COLMAP] Finished, see {logfile_name} for logs.')