
import argparse
import shutil

import numpy as np
import os
from random import shuffle

from PIL import Image

from dataset.exporter import Exporter
from dataset.format import parents, id_to_name
def combine_images(image1_path, image2_path, output_path):
    # 打开两张透明背景的 PNG 图片
    img1 = Image.open(image1_path).convert("RGBA")
    img2 = Image.open(image2_path).convert("RGBA")

    # 将透明背景转为白色背景
    white_bg = Image.new("RGBA", img1.size, (255, 255, 255, 255))
    white_bg2 = Image.new("RGBA", img2.size, (255, 255, 255, 255))
    img1 = Image.alpha_composite(white_bg, img1).convert("RGB")
    img2 = Image.alpha_composite(white_bg2, img2).convert("RGB")

    # 创建新画布（宽度为两张图之和，高度相同）
    new_img = Image.new('RGB', (img1.width + img2.width, img1.height))

    # 拼接图片
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))

    # 保存结果
    new_img.save(output_path, "PNG")
    print(f"图片已保存至: {output_path}")

def main(args):
    '''
    Debug example. Reads one asset, one animation track and binds them and then export as fbx format.
    Users can visualize the results in softwares like blender.
    '''
    render = args.render
    export_fbx = args.export_fbx
    predict_output_dir = args.predict_output_dir

    if export_fbx is True:
        try:
            import bpy
        except:
            print("need bpy==4.2 & python==3.11")
            return

    exporter = Exporter()
    roots = []
    for root, dirs, files in os.walk(predict_output_dir):
        if 'predict_skeleton.npy' in files:
            roots.append(root)
    shuffle(roots)
    print(roots)
    for root in roots:
        print(f"export {os.path.relpath(root, predict_output_dir)}")
        save_path = os.path.join(args.render_output_dir, os.path.relpath(root, predict_output_dir))
        os.makedirs(save_path, exist_ok=True)
        vertices = np.load(os.path.join(root, 'transformed_vertices.npy'))
        # skin = np.load(os.path.join(root, 'predict_skin.npy'))
        joints = np.load(os.path.join(root, 'predict_skeleton.npy'))
        joints_gt = None
        # if os.path.exists(os.path.join(root, 'predict_skeleton_gt.npy')):
        #     joints_gt = np.load(os.path.join(root, 'predict_skeleton_gt.npy'))
        #     j2j = np.load(os.path.join(root, 'J2J_loss.npy'))
        if render:
            exporter._render_skeleton(
                path=os.path.join(save_path, 'skeleton.png'),
                joints=joints,
                parents=parents,
            )
            exporter._render_pc(
                path=os.path.join(save_path, f'ver_trans.png'),
                vertices=vertices,
            )

            if joints_gt is not None:
                exporter._render_skeleton(
                    path=os.path.join(save_path, 'skeleton_gt.png'),
                    joints=joints_gt,
                    parents=parents,
                )


            ske_image = os.path.join(save_path, 'skeleton.png')
            ver_trans_image = os.path.join(save_path, f'ver_trans.png')
            new_image = os.path.join(args.render_output_dir, root.replace('/', '_'))

            if os.path.exists(ske_image):
                combine_images(ske_image, ver_trans_image, new_image+".png")
            else:
                shutil.move(ver_trans_image, new_image+".png")
            # for id in id_to_name:
            #     name = id_to_name[id]
            #     exporter._render_skin(
            #         path=os.path.join(save_path, f'skin_{name}.png'),
            #         skin=skin[:, id],
            #         vertices=vertices,
            #         joint=joints[id],
            #     )
        # if export_fbx:
        #     names = [f'bone_{i}' for i in range(len(parents))]
        #     asset = Asset.load(path=os.path.join("data", root.replace('predict', 'test')+".npz"))
        #     exporter._export_fbx(
        #         path=os.path.join(save_path, f'res.fbx'),
        #         vertices=vertices,
        #         joints=joints,
        #         skin=skin,
        #         parents=parents,
        #         names=names,
        #         faces=asset.faces,
        #     )

def str2bool(val):
    val = val.lower()
    if val == 'false':
        return False
    elif val == 'true':
        return True
    else:
        raise NotImplementedError(f"expect false or true, found {val}")

if __name__ == "__main__":

    output = 'predict'
    tt = 'test'
    # tt = 'val'


    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_output_dir", type=str, default='predict_' + tt + '/' + output)
    parser.add_argument("--render_output_dir", type=str, default='render_' + tt + '/' + output)
    parser.add_argument("--render", type=str2bool, required=False, default=True)
    parser.add_argument("--export_fbx", type=str2bool, required=False, default=False)
    
    args = parser.parse_args()
    main(args)
