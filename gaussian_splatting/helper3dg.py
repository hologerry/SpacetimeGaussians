#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import shutil


def get_render_parts(render_pkg):
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


def get_colmap_single_n3d(folder, offset):
    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = f"colmap feature_extractor --database_path {db_file} --image_path {input_image_folder}"

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = f"colmap exhaustive_matcher --database_path {db_file}"
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    # threshold is from   https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/scripts/local_colmap_and_resize.sh#L62
    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP"
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_im_undistort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = (
        "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
    )

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    # Copy each file from the source directory to the destination directory
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_im_distort(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = (
        "colmap feature_extractor SiftExtraction.max_image_size 6000 --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
    )

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )  # --blank_pixels 1
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    remove_input = "rm -r " + input_image_folder
    exit_code = os.system(remove_input)
    if exit_code != 0:
        exit(exit_code)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)


def get_colmap_single_techni(folder, offset):

    folder = os.path.join(folder, "colmap_" + str(offset))
    assert os.path.exists(folder)

    db_file = os.path.join(folder, "input.db")
    input_image_folder = os.path.join(folder, "input")
    distorted_model = os.path.join(folder, "distorted/sparse")
    step2model = os.path.join(folder, "tmp")
    if not os.path.exists(step2model):
        os.makedirs(step2model)

    manual_input_folder = os.path.join(folder, "manual")
    if not os.path.exists(distorted_model):
        os.makedirs(distorted_model)

    feature_extract = "colmap feature_extractor --database_path " + db_file + " --image_path " + input_image_folder

    exit_code = os.system(feature_extract)
    if exit_code != 0:
        exit(exit_code)

    feature_matcher = "colmap exhaustive_matcher --database_path " + db_file
    exit_code = os.system(feature_matcher)
    if exit_code != 0:
        exit(exit_code)

    tri_and_map = (
        "colmap point_triangulator --database_path "
        + db_file
        + " --image_path "
        + input_image_folder
        + " --output_path "
        + distorted_model
        + " --input_path "
        + manual_input_folder
        + " --Mapper.ba_global_function_tolerance=0.000001"
    )

    exit_code = os.system(tri_and_map)
    if exit_code != 0:
        exit(exit_code)
    print(tri_and_map)

    img_undist_cmd = (
        "colmap"
        + " image_undistorter --image_path "
        + input_image_folder
        + " --input_path "
        + distorted_model
        + " --output_path "
        + folder
        + " --output_type COLMAP "
    )
    exit_code = os.system(img_undist_cmd)
    if exit_code != 0:
        exit(exit_code)
    print(img_undist_cmd)

    files = os.listdir(folder + "/sparse")
    os.makedirs(folder + "/sparse/0", exist_ok=True)
    for file in files:
        if file == "0":
            continue
        source_file = os.path.join(folder, "sparse", file)
        destination_file = os.path.join(folder, "sparse", "0", file)
        shutil.move(source_file, destination_file)

    return
