import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set the scene name
scene_name = "FloorPlan202"

# Load the pickle files
with open(f'{scene_name}/initial.pkl', 'rb') as file:
    initial_data = pickle.load(file)
with open(f'{scene_name}/shuffled.pkl', 'rb') as file:
    final_data = pickle.load(file)

# Extract the objects data
pfobjects_data = final_data['objects']
pinitial_objects_data =initial_data['objects']

# Define a list of non-moving objects to remove
keywords_to_remove = [
    'table',
    'wall',
    'floor',
    'ceiling',
    'frame',
    'picture',
    'window',
    'bed',
    'mirror',
    'mat',
    'chair',
    'plant',
    'lamp',
    ' desk',
]

def filter_objects_pair(pairs, keywords):
    filtered_pairs = []
    for pair in pairs:
        # Extract the two objects from the pair
        obj_0, obj_1 = pair

        # For each object, find the class name with the highest confidence
        def is_object_valid(obj):
            if 'class_name' in obj and 'conf' in obj:
                max_conf_index = obj['conf'].index(max(obj['conf']))
                max_conf_class_name = obj['class_name'][max_conf_index]
                return not any(keyword in max_conf_class_name for keyword in keywords)
            return False

        # Check both objects in the pair, and if both are valid, add the pair to the filtered list
        if is_object_valid(obj_0) and is_object_valid(obj_1):
            filtered_pairs.append(pair)

    return filtered_pairs


def cloud_length_similarity(cloud1, cloud2):
    len_cloud1 = len(cloud1)
    len_cloud2 = len(cloud2)

    # To avoid division by zero in case one of the clouds is empty
    if len_cloud1 == 0 and len_cloud2 == 0:
        return 1  # Both are empty, hence they are similar
    if len_cloud1 == 0 or len_cloud2 == 0:
        return 0  # One is empty, the other is not, hence no similarity

    # Calculate the absolute difference in lengths
    difference = abs(len_cloud1 - len_cloud2)

    # Normalize the difference by the maximum length
    max_length = max(len_cloud1, len_cloud2)
    normalized_difference = difference / max_length

    # Similarity measure (1 means identical, 0 means no similarity)
    similarity = 1 - normalized_difference

    return similarity

# Compare based on the scene with larger number of objects

if len(pinitial_objects_data) > len(pfobjects_data):
    resersed = True
    fobjects_data = pinitial_objects_data
    initial_objects_data = pfobjects_data
else:
    reversed = False
    fobjects_data = pfobjects_data
    initial_objects_data = pinitial_objects_data

# Finding the best matches
best_matches = []

potential_matches = {}  # To store potential matches with their similarity scores

# extracting objects pairs that have the highest similarity score
for i, fobject in enumerate(fobjects_data):
    for j, iobject in enumerate(initial_objects_data):
        # Calculate the similarity between the two objects
        similarity = np.dot(fobject['clip_ft'], iobject['clip_ft']) / (np.linalg.norm(fobject['clip_ft']) * np.linalg.norm(iobject['clip_ft']))
        if (j, i) not in potential_matches or potential_matches[(j, i)] < similarity:
            potential_matches[(j, i)] = similarity

# Filter the matches to ensure each iobject is only paired with one fobject
best_matches = {}
for (iobject_index, fobject_index), similarity in potential_matches.items():
    if iobject_index not in best_matches or best_matches[iobject_index][1] < similarity:
        best_matches[iobject_index] = (fobject_index, similarity)

# Convert best_matches to a list of pairs (fobject, iobject)
best_matches = [(fobjects_data[fobject_index], initial_objects_data[iobject_index]) for iobject_index, (fobject_index, _) in best_matches.items()]
# filter out the non-moving objects
best_matches = filter_objects_pair(best_matches, keywords_to_remove)
moved_objects_pairs = []

# make a list of  the filterd-displaced best matches
for fobject, iobject in best_matches:
    fcentriod = np.mean(fobject['bbox_np'] , axis=0)
    icentriod = np.mean(iobject['bbox_np'] , axis=0)
    displacement = np.linalg.norm(fcentriod - icentriod)
    point_cloud1 = fobject['pcd_np']
    point_cloud2 = iobject['pcd_np']
    cloud_length_sim = cloud_length_similarity(point_cloud1, point_cloud2)
    if cloud_length_sim > 0.4 and displacement > 0.5:
        moved_objects_pairs.append((fobject, iobject))




# Visualize the filterd-displaced best matches
for fobject, iobject in moved_objects_pairs:
    if reversed:
        fobject, iobject = iobject, fobject
    else:
        fobject, iobject = fobject, iobject
    # Extract data for visualization
    fcentriod = np.mean(fobject['bbox_np'] , axis=0)
    icentriod = np.mean(iobject['bbox_np'] , axis=0)
    displacement = np.linalg.norm(fcentriod - icentriod)

    nclip_ft1 = fobject['clip_ft']
    nclip_ft2 = iobject['clip_ft']
    nsimilarity = np.dot(nclip_ft1, nclip_ft2) / (np.linalg.norm(nclip_ft1) * np.linalg.norm(nclip_ft2))

    point_cloud1 = fobject['pcd_np']
    point_cloud2 = iobject['pcd_np']
    cloud_length_sim = cloud_length_similarity(point_cloud1, point_cloud2)

    final_image_path = fobject['color_path'][0]
    initial_image_path = iobject['color_path'][0]
    imask = iobject['mask'][0]
    fmasks = fobject['mask'][0]
    max_confidence_index_f = fobject["conf"].index(max(fobject["conf"]))
    max_confidence_index_i = iobject["conf"].index(max(iobject["conf"]))
    iclass_name = iobject['class_name'][max_confidence_index_i]
    fclass_name = fobject['class_name'][max_confidence_index_f]
    iconfidence = iobject['conf'][max_confidence_index_i]
    fconfidence = fobject['conf'][max_confidence_index_f]


    # Open and prepare images
    relavent_path_iip= "/".join(initial_image_path.split("/")[-3:])
    relavent_path_fip = "/".join(final_image_path.split("/")[-3:])
    formated_iip= scene_name + "/" + relavent_path_iip
    formated_fip = scene_name + "/" + relavent_path_fip
    iimage = Image.open(formated_iip)
    fimage = Image.open(formated_fip)
    iimage_np = np.array(iimage)
    fimage_np = np.array(fimage)

    # Create RGBA images with masks
    irgba_image = np.zeros((iimage_np.shape[0], iimage_np.shape[1], 4), dtype=np.uint8)
    frgba_image = np.zeros((fimage_np.shape[0], fimage_np.shape[1], 4), dtype=np.uint8)
    irgba_image[..., :3] = iimage_np
    frgba_image[..., :3] = fimage_np
    irgba_image[..., 3] = 255  # Full opacity
    frgba_image[..., 3] = 255
    red_mask = [255, 0, 0, 128]
    for k in range(4):
        irgba_image[..., k] = np.where(imask, np.clip(irgba_image[..., k] * 0.5 + red_mask[k] * 0.5, 0, 255),
                                       irgba_image[..., k])
        frgba_image[..., k] = np.where(fmasks, np.clip(frgba_image[..., k] * 0.5 + red_mask[k] * 0.5, 0, 255),
                                       frgba_image[..., k])

    imasked_image = Image.fromarray(irgba_image)
    fmasked_image = Image.fromarray(frgba_image)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(imasked_image)
    ax[0].set_title( "Initial State", fontsize=14)
    ax[0].text(10, 10, f"{iclass_name} ({iconfidence:.2f})", color='blue', fontsize=12, verticalalignment='top',
               backgroundcolor='white')

    ax[1].imshow(fmasked_image)
    ax[1].set_title( "Shuffled State", fontsize=14)
    ax[1].text(10, 10, f"{fclass_name} ({fconfidence:.2f})", color='blue', fontsize=12, verticalalignment='top',
               backgroundcolor='white')
    fig.text(0.5, 0.95, f"Clip Similarity: {nsimilarity:.3f} | Displacement: {displacement:.3f} | Cloud Similarity: {cloud_length_sim:.3f}", ha='center',
             va='center', fontsize=14)
    plt.show()



