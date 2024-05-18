import os
import shutil
import argparse


parser = argparse.ArgumentParser(description="Processing output images")
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
)
args = parser.parse_args()
# 'data/backpack'
current_dir = args.data_path


for type_name in os.listdir('data'):
    type_dir = os.path.join('data', type_name)

    for instance_name in os.listdir(type_dir):
        for i in range(25):
            target_dir = f"{current_dir}-{i+1}"
            src_dir =  f"{current_dir}src"

            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            os.makedirs(target_dir)
            for p in range(4):
                src_path = os.path.join(src_dir,  f"{p+i*4}.png")
                tar_path = os.path.join(target_dir, f"{p+i*4}.png")
                shutil.copyfile(src_path, tar_path)
