import os
import shutil

# MODIFY HERE: Select scenes and hold value
scene_names = ["cozyroom", "deskchair", "nestchair", "pool", "shotgun"]  # You can modify this list
hold = 8                                                                 # You can change the hold interval here


def split_scene_images(scene_name: str, hold: int = 7):
    scene_path = os.path.join("data", scene_name)
    images_path = os.path.join(scene_path, "images")

    if not os.path.exists(images_path):
        print(f"[Skip] {images_path} folder does not exist.")
        return

    # Output directories
    blur_path = os.path.join(scene_path, "blur")
    nv_path = os.path.join(scene_path, "nv")
    hold_file_path = os.path.join(scene_path, f"hold={hold}")

    if os.path.exists(blur_path):
        shutil.rmtree(blur_path)
    if os.path.exists(nv_path):
        shutil.rmtree(nv_path)
    if os.path.exists(hold_file_path):
        os.remove(hold_file_path)

    os.makedirs(blur_path, exist_ok=True)
    os.makedirs(nv_path, exist_ok=True)
    with open(hold_file_path, 'w') as f:
        pass

    # Get sorted list of image files
    image_files = sorted([
        f for f in os.listdir(images_path)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    print(f"[{scene_name}] Detected {len(image_files)} image files.")
    for idx, filename in enumerate(image_files):
        src = os.path.join(images_path, filename)
        if idx % hold == 0:
            dst = os.path.join(nv_path, filename)
        else:
            dst = os.path.join(blur_path, filename)
        shutil.copy2(src, dst)

    print(f"[{scene_name}] Done. {len(image_files)//hold + 1} to nv/, the rest to blur/.")
    print(f"[{scene_name}] Created {hold_file_path} .\n")


if __name__ == "__main__":
    for scene in scene_names:
        split_scene_images(scene, hold)
