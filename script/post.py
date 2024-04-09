# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import glob
import os

import cv2
import natsort
import numpy as np

# import sys
# sys.path.append(".")
from thirdparty.colmap.pre_colmap import *
from thirdparty.gaussian_splatting.colmap_loader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
)


def compare_spatial_temporal():
    method_dict = {}
    method_dict["ours"] = ""
    method_dict["ourslite"] = ""
    method_dict["dynamic3DGs"] = ""
    method_dict["hpreal"] = ""
    method_dict["kplane"] = ""
    method_dict["mixvoxel"] = ""
    method_dict["nerfplayer"] = ""

    method_dict["gt"] = method_dict["ours"].replace("/renders/", "/gt/")

    assert method_dict["ours"] != method_dict["gt"]

    top_left = (0, 0)  #
    y, x = top_left[0], top_left[1]

    delta_y = 150
    delta_x_frames = 250

    for k in method_dict.keys():
        total = []
        path = method_dict[k]
        if path != None:
            image_list = glob.glob(path)
            image_list = natsort.natsorted(image_list)
            print(k, len(image_list))
            image_list = image_list[50:]
            print(image_list[0])
            for image_path in image_list[0:delta_x_frames]:
                image = cv2.imread(image_path)
                patch = image[y : y + delta_y, x : x + 1, :]
                total.append(patch)
            final = np.hstack(total)
            cv2.imwrite("output" + str(k) + ".png", final)


def convert_videos():
    save_dir = "/home/output"
    path = "/renders/*.png"

    images = natsort.natsorted(glob.glob(path))

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    file_name = os.path.join(save_dir, "flame_steak.mp4")
    video = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))  # change fps by yourself

    for image_path in images:
        image = cv2.imread(image_path)
        video.write(image)

    cv2.destroyAllWindows()

    video.release()


# code to look for matched image
def look_for_image():
    source_image = "xxx.jpg"
    image = cv2.imread(source_image)
    image_list = glob.glob("/home/gt/*.png")

    image_list = natsort.natsorted(image_list)
    image = cv2.resize(image.astype(np.float32), (1352, 1014), interpolation=cv2.INTER_CUBIC)
    max_psnr = 0
    max_psnr_path = 0
    second_path = 0
    second_psnr = 0
    for image_path in image_list:
        img2 = cv2.imread(image_path).astype(np.float32)
        img2 = cv2.resize(img2.astype(np.float32), (1352, 1014), interpolation=cv2.INTER_CUBIC)

        psnr = cv2.PSNR(image, img2)
        if psnr > max_psnr:
            second_psnr = max_psnr

            max_psnr = psnr
            second_path = max_psnr_path
            max_psnr_path = image_path

    print(max_psnr, max_psnr_path)
    print(second_psnr, second_path)


def remove_nfs():
    # remove not used nfs files

    nfs_list = glob.glob("//.nfs*")

    for f in nfs_list:
        cmd = " lsof -t " + f + " | xargs kill -9 "
        ret = os.system(cmd)
        print(cmd)


def extract_colmap_model_to_db(path, offset=1):

    project_folder = os.path.join(path, "colmap_" + str(offset))
    ref_project_folder = os.path.join(path, "colmap_" + str(0))
    manual_folder = os.path.join(project_folder, "manual")

    cameras_extrinsic_file = os.path.join(ref_project_folder, "sparse/0/", "images.bin")  # from distorted?
    cameras_intrinsic_file = os.path.join(ref_project_folder, "sparse/0/", "cameras.bin")

    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    if not os.path.exists(manual_folder):
        os.makedirs(manual_folder)

    save_txt = os.path.join(manual_folder, "images.txt")
    save_camera = os.path.join(manual_folder, "cameras.txt")
    savepoints = os.path.join(manual_folder, "points3D.txt")
    image_txt_list = []
    camera_txt_list = []
    if os.path.exists(os.path.join(project_folder, "input.db")):
        os.remove(os.path.join(project_folder, "input.db"))

    db = COLMAPDatabase.connect(os.path.join(project_folder, "input.db"))

    db.create_tables()

    helper_dict = {}
    total_cam_name = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        total_cam_name.append(extr.name)
        helper_dict[extr.name] = [extr, intr]

    sorted_total_cam_name_list = natsort.natsorted(total_cam_name)
    sorted_name_dict = {}
    for i in range(len(sorted_total_cam_name_list)):
        sorted_name_dict[sorted_total_cam_name_list[i]] = i  # map each cam with a number

    video_path = glob.glob(ref_project_folder + "/images/*.png")
    for i in range(len(video_path)):
        camera_name = os.path.basename(video_path[i])[:-4]  # "cam" + str(i).zfill(2)
        camera_name_as_key = camera_name + ".png"
        extr, intr = helper_dict[camera_name_as_key]  # extr.name

        width, height, params = intr.width, intr.height, intr.params
        focal_length = intr.params[0]
        fw, fh = intr.params[2], intr.params[3]

        colmapQ = extr.qvec
        T = extr.tvec

        image_id = str(i + 1)
        camera_id = image_id
        png_name = camera_name + ".png"

        line = image_id + " "

        for j in range(4):
            line += str(colmapQ[j]) + " "
        for j in range(3):
            line += str(T[j]) + " "
        line = line + camera_id + " " + png_name + "\n"
        empty_line = "\n"
        image_txt_list.append(line)
        image_txt_list.append(empty_line)

        # model, width, height, params = i, W, H, np.array((focal_length, W//2, H//2, 0.1))

        camera_id = db.add_camera(1, width, height, params)
        camera_line = (
            str(i + 1)
            + " "
            + "PINHOLE "
            + str(width)
            + " "
            + str(height)
            + " "
            + str(focal_length)
            + " "
            + str(focal_length)
            + " "
            + str(fw)
            + " "
            + str(fh)
            + "\n"
        )
        camera_txt_list.append(camera_line)

        image_id = db.add_image(
            png_name,
            camera_id,
            prior_q=np.array((colmapQ[0], colmapQ[1], colmapQ[2], colmapQ[3])),
            prior_t=np.array((T[0], T[1], T[2])),
            image_id=i + 1,
        )
        db.commit()
    db.close()

    with open(save_txt, "w") as f:
        for line in image_txt_list:
            f.write(line)
    with open(save_camera, "w") as f:
        for line in camera_txt_list:
            f.write(line)
    with open(savepoints, "w") as f:
        pass
    print("done")


WEIGHT_DICT = {}
DATA_DICT = {}
n3d = ["flame_salmon_1", "flame_steak", "cook_spinach", "cut_roasted_beef", "coffee_martini", "sear_steak"]
Techni = ["Birthday", "Fabien", "Painter", "Theater", "Train"]
IM_dist = []
CONFIG_DICT = {}


for k in n3d:
    WEIGHT_DICT[k] = "/"
    DATA_DICT[k] = "/"

for k in Techni:
    WEIGHT_DICT[k] = "/techni/"
    DATA_DICT[k] = "/technicolor/"

for k in IM_dist:
    WEIGHT_DICT[k] = "/IM_dist/"
    DATA_DICT[k] = "/IM_dist/"


def get_value_from_args(args_str, key):
    args_list = args_str.split()
    try:
        key_index = args_list.index(key)
        return args_list[key_index + 1]
    except (ValueError, IndexError):
        return None


def generate_script(
    gpu_list,
    gpu_server,
    frame_range=[0, 50],
    step=50,
    scene_list=["flame_salmon_1"],
    option="train",
    test_iter=30000,
    special_name="",
    additional="",
    add="w",
    script="train_v2",
    densify_dict=None,
    config_root=None,
):

    if scene_list[0] not in Techni:
        train_command = " python " + script + ".py -r 2 --quiet --eval --test_iterations -1"
        test_command = " python test.py -r 2" + " --quiet --eval --skip_train"
    else:
        train_command = " python " + script + ".py --quiet --eval --test_iterations -1"
        test_command = " python test.py" + " --quiet --eval --skip_train"
    cmd_list = []

    if option == "g_model":
        for scene in scene_list:
            for i in frame_range:
                colmap_folder = DATA_DICT[scene] + scene + "/colmap_" + str(i)
                model_save_folder = WEIGHT_DICT[scene] + special_name + "/" + scene + "/colmap_" + str(i)
                tmp_command = (
                    train_command
                    + " -s "
                    + colmap_folder
                    + " -m "
                    + model_save_folder
                    + " --config "
                    + config_root
                    + scene
                    + ".json"
                )
                curtest_command = (
                    test_command
                    + " -s "
                    + colmap_folder
                    + " -m "
                    + model_save_folder
                    + " --config "
                    + config_root
                    + scene
                    + ".json"
                )  # how to overwrite config with additional input?

                try:
                    if scene in densify_dict:
                        tmp_command += " --densify " + str(densify_dict[scene])
                except:
                    pass

                curtest_command += " --test_iteration " + str(test_iter)

                cmd_list.append((tmp_command, curtest_command))

    sequences = np.array_split(cmd_list, len(gpu_list))

    if scene in n3d:
        additional_test = " --val_loader colmap_valid"
    elif scene in Techni:
        additional_test = " --val_loader technicolor"
    elif scene in IM_dist:
        additional_test = " --val_loader immersive_valid_ss"
    for idx in range(len(gpu_list)):
        gpu_id = gpu_list[idx]
        script_name = gpu_server + "_gpu" + str(gpu_id).zfill(1) + option + ".sh"  # to across different server
        this_file_commands = sequences[idx]
        print("writing to ", script_name)
        with open(script_name, add) as file_script:
            if len(this_file_commands[0]) == 3:
                for op, test_op, metric_op in this_file_commands:
                    train = (
                        "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + op + " " + additional
                    )  # PYTHONDONTWRITEBYTECODE=1 important !
                    test = (
                        "PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="
                        + str(gpu_id)
                        + " "
                        + test_op
                        + additional_test
                    )
                    if test_iter == 30000:
                        file_script.write("%s ;\n" % train)
                        file_script.write("%s ;\n" % test)  # skip test f
            else:
                for op in this_file_commands:
                    train = "CUDA_VISIBLE_DEVICES=" + str(gpu_id) + " " + op
                    file_script.write("%s ;\n" % train)

            file_script.close()


if __name__ == "__main__":
    # remove_nfs()
    # compare_spatial_temporal()
    # convert_videos()
    pass  #
