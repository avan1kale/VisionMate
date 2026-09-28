import os
import json
import shutil
import random

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"E:\DEEP LEARNING\CORNERSTONE"

# COCO balanced dataset
COCO_ROOT = os.path.join(
    BASE_DIR,
    "COCO_Custom_API_Dataset",
    "VisionMate_COCO_Balanced"
)

# SUN RGB-D dataset
SUN_ROOT = os.path.join(
    BASE_DIR,
    "SUNRGBD",
    "SUNRGBD_VisionMate"
)

# Final merged dataset
OUTPUT_ROOT = os.path.join(
    BASE_DIR,
    "VisionMate_Final"
)


# ============================================================
# FINAL 8 CLASSES
# ============================================================

FINAL_CLASSES = {
    "person": 1,
    "car": 2,
    "bicycle": 3,
    "motorcycle": 4,
    "chair": 5,
    "table": 6,
    "door": 7,
    "books": 8
}


# ============================================================
# CLASS NORMALIZATION
# ============================================================

def normalize_class_name(name):

    name = name.lower().strip()

    # COCO
    if name == "dining table":
        return "table"

    # SUN RGB-D / possible naming variation
    if name == "book":
        return "books"

    return name


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val"]:

    os.makedirs(
        os.path.join(
            OUTPUT_ROOT,
            "images",
            split
        ),
        exist_ok=True
    )

os.makedirs(
    os.path.join(
        OUTPUT_ROOT,
        "annotations"
    ),
    exist_ok=True
)


# ============================================================
# CHECK SOURCE DIRECTORIES
# ============================================================

print("\nChecking source datasets...")

print(
    "\nCOCO:",
    COCO_ROOT
)

print(
    "Exists:",
    os.path.exists(COCO_ROOT)
)

print(
    "\nSUN RGB-D:",
    SUN_ROOT
)

print(
    "Exists:",
    os.path.exists(SUN_ROOT)
)


if not os.path.exists(COCO_ROOT):

    raise FileNotFoundError(
        "COCO dataset not found!"
    )


if not os.path.exists(SUN_ROOT):

    raise FileNotFoundError(
        "SUN RGB-D dataset not found!"
    )


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in ["train", "val"]:

    print("\n")
    print("=" * 60)
    print(f"PROCESSING {split.upper()} DATA")
    print("=" * 60)


    # --------------------------------------------------------
    # SOURCE JSON PATHS
    # --------------------------------------------------------

    coco_json_path = os.path.join(
        COCO_ROOT,
        "annotations",
        f"{split}.json"
    )

    sun_json_path = os.path.join(
        SUN_ROOT,
        "annotations",
        f"{split}.json"
    )


    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    with open(
        coco_json_path,
        "r"
    ) as f:

        coco_data = json.load(f)


    with open(
        sun_json_path,
        "r"
    ) as f:

        sun_data = json.load(f)


    # --------------------------------------------------------
    # FINAL COCO FORMAT STRUCTURE
    # --------------------------------------------------------

    final_data = {

        "info": {
            "description":
                "VisionMate Final Dataset - COCO + SUN RGB-D"
        },

        "images": [],

        "annotations": [],

        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "object"
            },
            {
                "id": 2,
                "name": "car",
                "supercategory": "object"
            },
            {
                "id": 3,
                "name": "bicycle",
                "supercategory": "object"
            },
            {
                "id": 4,
                "name": "motorcycle",
                "supercategory": "object"
            },
            {
                "id": 5,
                "name": "chair",
                "supercategory": "object"
            },
            {
                "id": 6,
                "name": "table",
                "supercategory": "object"
            },
            {
                "id": 7,
                "name": "door",
                "supercategory": "object"
            },
            {
                "id": 8,
                "name": "books",
                "supercategory": "object"
            }
        ]
    }


    # ========================================================
    # SOURCE DATASETS
    # ========================================================

    datasets = [
        ("COCO", coco_data),
        ("SUN", sun_data)
    ]


    # ========================================================
    # ID COUNTERS
    # ========================================================

    next_image_id = 1
    next_annotation_id = 1


    # ========================================================
    # CLASS COUNTERS
    # ========================================================

    class_counts = {
        name: 0
        for name in FINAL_CLASSES
    }


    # ========================================================
    # IMAGE COUNTERS
    # ========================================================

    source_image_counts = {
        "COCO": 0,
        "SUN": 0
    }

    copied_image_counts = {
        "COCO": 0,
        "SUN": 0
    }

    skipped_images = []


    # ========================================================
    # PROCESS COCO + SUN
    # ========================================================

    for dataset_name, dataset in datasets:

        print(
            f"\nProcessing {dataset_name}..."
        )


        # ----------------------------------------------------
        # SOURCE ROOT
        # ----------------------------------------------------

        if dataset_name == "COCO":

            source_image_root = os.path.join(
                COCO_ROOT,
                "images",
                split
            )

            prefix = "coco"

        else:

            source_image_root = os.path.join(
                SUN_ROOT,
                "images",
                split
            )

            prefix = "sun"


        # ----------------------------------------------------
        # CATEGORY MAPPING
        # ----------------------------------------------------

        category_map = {}

        for category in dataset["categories"]:

            original_name = category["name"]

            normalized_name = normalize_class_name(
                original_name
            )

            if normalized_name in FINAL_CLASSES:

                category_map[
                    category["id"]
                ] = FINAL_CLASSES[
                    normalized_name
                ]


        print(
            "Category mapping:",
            category_map
        )


        # ----------------------------------------------------
        # OLD IMAGE ID → NEW IMAGE ID
        # ----------------------------------------------------

        image_id_map = {}


        # ====================================================
        # PROCESS IMAGES
        # ====================================================

        for image in dataset["images"]:

            source_image_counts_key = dataset_name

            source_image_counts[
                source_image_counts_key
            ] += 1


            old_image_id = image["id"]

            old_filename = image["file_name"]


            # ------------------------------------------------
            # SOURCE IMAGE
            # ------------------------------------------------

            source_path = os.path.join(
                source_image_root,
                old_filename
            )


            # ------------------------------------------------
            # CHECK SOURCE IMAGE
            # ------------------------------------------------

            if not os.path.exists(source_path):

                skipped_images.append(
                    (
                        dataset_name,
                        split,
                        old_filename
                    )
                )

                continue


            # ------------------------------------------------
            # NEW UNIQUE FILENAME
            # ------------------------------------------------

            extension = os.path.splitext(
                old_filename
            )[1]

            new_filename = (
                f"{prefix}_{old_image_id:07d}"
                f"{extension}"
            )


            destination_path = os.path.join(

                OUTPUT_ROOT,
                "images",
                split,
                new_filename
            )


            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            shutil.copy2(
                source_path,
                destination_path
            )


            # ------------------------------------------------
            # NEW IMAGE ID
            # ------------------------------------------------

            new_image_id = next_image_id

            next_image_id += 1


            image_id_map[
                old_image_id
            ] = new_image_id


            # ------------------------------------------------
            # ADD IMAGE RECORD
            # ------------------------------------------------

            final_data["images"].append({

                "id":
                    new_image_id,

                "file_name":
                    new_filename,

                "width":
                    image["width"],

                "height":
                    image["height"]
            })


            copied_image_counts[
                dataset_name
            ] += 1


        # ====================================================
        # PROCESS ANNOTATIONS
        # ====================================================

        for annotation in dataset["annotations"]:

            old_image_id = annotation[
                "image_id"
            ]


            # ------------------------------------------------
            # IMAGE WAS SKIPPED?
            # ------------------------------------------------

            if old_image_id not in image_id_map:

                continue


            # ------------------------------------------------
            # OLD CATEGORY
            # ------------------------------------------------

            old_category_id = annotation[
                "category_id"
            ]


            # ------------------------------------------------
            # UNKNOWN CATEGORY
            # ------------------------------------------------

            if old_category_id not in category_map:

                continue


            # ------------------------------------------------
            # NEW CATEGORY
            # ------------------------------------------------

            new_category_id = category_map[
                old_category_id
            ]


            # ------------------------------------------------
            # GET BBOX
            # ------------------------------------------------

            bbox = annotation.get(
                "bbox",
                []
            )


            if len(bbox) != 4:

                continue


            x, y, w, h = map(
                float,
                bbox
            )


            # ------------------------------------------------
            # GET IMAGE DIMENSIONS
            # ------------------------------------------------

            new_image_id = image_id_map[
                old_image_id
            ]


            # Find image dimensions
            image_info = next(
                img
                for img in final_data["images"]
                if img["id"] == new_image_id
            )


            width = image_info["width"]
            height = image_info["height"]


            # ------------------------------------------------
            # CLIP BOUNDING BOX
            # ------------------------------------------------

            x = max(
                0,
                min(x, width)
            )

            y = max(
                0,
                min(y, height)
            )

            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )


            # ------------------------------------------------
            # INVALID BOX
            # ------------------------------------------------

            if w <= 0 or h <= 0:

                continue


            # ------------------------------------------------
            # ADD ANNOTATION
            # ------------------------------------------------

            final_data[
                "annotations"
            ].append({

                "id":
                    next_annotation_id,

                "image_id":
                    new_image_id,

                "category_id":
                    new_category_id,

                "bbox": [
                    x,
                    y,
                    w,
                    h
                ],

                "area":
                    w * h,

                "iscrowd":
                    annotation.get(
                        "iscrowd",
                        0
                    )
            })


            # ------------------------------------------------
            # COUNT CLASS
            # ------------------------------------------------

            class_name = next(

                name
                for name, cid
                in FINAL_CLASSES.items()

                if cid == new_category_id
            )


            class_counts[
                class_name
            ] += 1


            next_annotation_id += 1


    # ========================================================
    # SAVE FINAL JSON
    # ========================================================

    output_json_path = os.path.join(

        OUTPUT_ROOT,
        "annotations",
        f"{split}.json"
    )


    with open(
        output_json_path,
        "w"
    ) as f:

        json.dump(
            final_data,
            f,
            indent=2
        )


    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    print("\n" + "-" * 50)

    print(
        f"{split.upper()} SUMMARY"
    )

    print("-" * 50)

    print(
        "COCO source images:",
        source_image_counts["COCO"]
    )

    print(
        "COCO copied images:",
        copied_image_counts["COCO"]
    )

    print(
        "SUN source images:",
        source_image_counts["SUN"]
    )

    print(
        "SUN copied images:",
        copied_image_counts["SUN"]
    )

    print(
        "Final images:",
        len(final_data["images"])
    )

    print(
        "Final annotations:",
        len(final_data["annotations"])
    )

    print("\nClass distribution:")

    for name, count in class_counts.items():

        print(
            f"{name:12s}: {count}"
        )

    print(
        "\nSaved:",
        output_json_path
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL DATASET VALIDATION")
print("=" * 60)


for split in ["train", "val"]:

    json_path = os.path.join(
        OUTPUT_ROOT,
        "annotations",
        f"{split}.json"
    )


    with open(
        json_path,
        "r"
    ) as f:

        data = json.load(f)


    # --------------------------------------------------------
    # IMAGE IDs
    # --------------------------------------------------------

    image_ids = {
        image["id"]
        for image in data["images"]
    }


    # --------------------------------------------------------
    # CHECK ANNOTATIONS
    # --------------------------------------------------------

    broken_image_refs = 0
    invalid_boxes = 0


    for annotation in data["annotations"]:

        if annotation["image_id"] not in image_ids:

            broken_image_refs += 1


        x, y, w, h = annotation["bbox"]


        if w <= 0 or h <= 0:

            invalid_boxes += 1


    # --------------------------------------------------------
    # CATEGORY CHECK
    # --------------------------------------------------------

    category_names = [
        category["name"]
        for category in data["categories"]
    ]


    print(
        f"\n{split.upper()}:"
    )

    print(
        "Images:",
        len(data["images"])
    )

    print(
        "Annotations:",
        len(data["annotations"])
    )

    print(
        "Categories:",
        category_names
    )

    print(
        "Broken image references:",
        broken_image_refs
    )

    print(
        "Invalid bounding boxes:",
        invalid_boxes
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 60)
print("MERGE COMPLETE")
print("=" * 60)

print(
    "\nFinal dataset:"
)

print(
    OUTPUT_ROOT
)