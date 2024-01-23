# Object Rearrangement using ConceptGraphs in AI2-THOR environments 

![Teaser](https://github.com/ibrahimyousri/object-rearrangement/blob/main/teaser.png?raw=true)
## We navigate through AI2-THOR scenes using keymapping collecting  RGP-D trajectories before and after shuffling objects, then we pass these data to ConceptGraphs to create object-based 3D point cloud mappings to detect moved objects in the scene.

## Installation 

Create a conda environment and install AI2-THOR

```bash
  conda install -c conda-forge ai2thor
  pip install ai2thor
```

Install this configurated ConceptGraphs from [here](https://github.com/concept-graphs/concept-graphs/tree/main#setup). Because we have added our  dataloader to adapt the captured RGP-D trajectories and changed  the camera intrinsics  configurations. 

## Capturing RGP-D trajectories from AI-2THOR
#### Scan Initial (arranged) state of the scenes 
Change the scene name you want to capture in the script and run:
```bash
python Scan/scan_initial_state.py
```
In this script, you  navigate through the scene by pressing the following keys,  after pressing each key the  RGP-D trajectory of this event is saved.
* `w` to Move Ahead. 
* `s` to Move Back.
* `a` to Move Left.
* `d` to Move Right.
* `e` to Rotate Right.
* `q` to Rotate Left.

#### Scan shuffled state of the scenes 
```bash
python Scan/scan_shuffled_state.py
```
Same navigation key-mapping  but with  two more keys:
* `r` to pickup object
* `f` to Throw object
* `n` to start Capturing RGP-D trajectories.

Here the mechanism is different than scanning the initial state, as you first have to navigate the scene to pick and throw objects, and after finishing shuffling the room you  press  `n` to start capturing the RGP-D trajectories.


## Building object-based Mappings using ConceptGraphs
Copy the captured scenes to your `$AI2THOR_ROOT` and set the following paths:

```bash
export AI2THOR_ROOT=/path/to/AI2THOR
export CG_FOLDER=/path/to/concept-graphs/
export AI2THOR_CONFIG_PATH=${CG_FOLDER}/conceptgraph/dataset/dataconfigs/replica/ai2thor.yaml
```
Your `AI2THOR_ROOT` folder should have all your scenes folders and follow a directory structure like this :
```


└── AI2THOR_ROOT/
    ├── fl202/
    │   ├── results/
    │   │   ├── depth000000.png
    │   │   ├── depth000001.png
    │   │   ├── depth000002.png
    │   │   ├── .....
    │   │   ├── frame000000.jpg
    │   │   ├── frame000001.jpg
    │   │   ├── frame000002.jpg
    │   │   └── .....
    │   └── traj.txt
    ├── sh202/
    ├── fl206/
    └── sh206/
  ```


Run the following scripts 
```bash
SCENE_NAME=FloorPlan206

# The ConceptGraphs-Detect 
CLASS_SET=ram
python scripts/generate_gsa_results.py \
    --dataset_root $AI2THOR_ROOT \
    --dataset_config $AI2THOR_CONFIG_PATH \
    --scene_id $SCENE_NAME \
    --class_set $CLASS_SET \
    --box_threshold 0.2 \
    --text_threshold 0.2 \
    --stride 5 \
    --add_bg_classes \
    --accumu_classes \
    --exp_suffix withbg_allclasses

# On the ConceptGraphs-Detect 
THRESHOLD=1.2
python slam/cfslam_pipeline_batch.py \
    dataset_root=$AI2THOR_ROOT \
    dataset_config=$AI2THOR_CONFIG_PATH \
    stride=5 \
    scene_id=$SCENE_NAME \
    spatial_sim_type=overlap \
    mask_conf_threshold=0.25 \
    match_method=sim_sum \
    sim_threshold=${THRESHOLD} \
    dbscan_eps=0.1 \
    gsa_variant=ram_withbg_allclasses \
    skip_bg=False \
    max_bbox_area_ratio=0.5 \
    save_suffix=overlap_maskconf0.25_simsum${THRESHOLD}_dbscan.1
    save_objects_all_frames=True

```

## Comparing mapping files
Take the pkl file for each scene before and after shuffle along with the captured RGP-D Trajectories and put them in the following directory structure:
 ```

Compare/
    ├── detect_moved_objects.py
    ├── FloorPlan202/
    │   ├── fl202/
    │   │   ├── results/
    │   │   │   ├── depth000000.png
    │   │   │   ├── depth000001.png
    │   │   │   ├── depth000002.png
    │   │   │   ├── .......
    │   │   │   ├── frame000000.jpg
    │   │   │   ├── frame000001.jpg
    │   │   │   ├── frame000002.jpg
    │   │   │   └── ........
    │   │   └── traj.txt
    │   ├── sh202/
    │   ├── initial.pkl
    │   └── shuffled.pkl
    ├── FloorPlan206/
    ├── FloorPlan207/
    └── FloorPlan301/
```
Change the scene name you want to explore the moved objects in and run
```bash
python Compare/detect_moved_objects.py
```
[Here](https://drive.google.com/file/d/166oxpC5hup6t1On3L6vRo1nhXRgKi4yV/view?usp=sharing) you can download 7 scenes before and after shuffle to explore.
