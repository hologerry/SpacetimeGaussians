# Copyright (C) 2020, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr


project(fullColmapProcess)

file(GLOB SCRIPTS "*.py" "*.json")

add_custom_target(${PROJECT_NAME} ALL)

include(install_runtime)
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/dataset_tools/preprocess")
ibr_install_rsc(${PROJECT_NAME} TYPE "scripts" FILES ${SCRIPTS})
