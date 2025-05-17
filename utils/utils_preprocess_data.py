import utils.utils_others as utils_others
import os
import torch



def preprocess_data(data_dir, dataset_names):
    for dataset_name in dataset_names:

        dataset_dir = os.path.join(data_dir, dataset_name)

        add_train_and_test = True
        max_images = 4000

        # common paths for all datasets
        all_img_dir = os.path.join(dataset_dir, "images")
        all_mask_dir = os.path.join(dataset_dir, "masks")

        splitted_images_and_masks_dir = os.path.join(dataset_dir, "splitted_original_images_and_masks")

        train_img_dir = os.path.join(splitted_images_and_masks_dir, "train_images")
        train_mask_dir = os.path.join(splitted_images_and_masks_dir, "train_masks")
        val_img_dir = os.path.join(splitted_images_and_masks_dir, "val_images")
        val_mask_dir = os.path.join(splitted_images_and_masks_dir, "val_masks")
        test_img_dir = os.path.join(splitted_images_and_masks_dir, "test_images")
        test_mask_dir = os.path.join(splitted_images_and_masks_dir, "test_masks")



        # data downloading

        if dataset_name == "isic2018":
            if not os.path.exists(val_img_dir):
                utils_others.download_and_extract("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip", dataset_dir, "ISIC2018_Task1-2_Validation_Input.zip")
            if not os.path.exists(val_mask_dir):
                utils_others.download_and_extract("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip", dataset_dir, "ISIC2018_Task1_Validation_GroundTruth.zip")
            if add_train_and_test:
                if not os.path.exists(all_img_dir):
                    utils_others.download_and_extract("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip", dataset_dir, "ISIC2018_Task1_Training_GroundTruth.zip")
                if not os.path.exists(all_mask_dir):
                    utils_others.download_and_extract("https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip", dataset_dir, "ISIC2018_Task1-2_Training_Input.zip")

            initial_val_img_dir = os.path.join(dataset_dir, "ISIC2018_Task1-2_Validation_Input")
            initial_val_mask_dir = os.path.join(dataset_dir, "ISIC2018_Task1_Validation_GroundTruth")
            initial_all_img_dir = os.path.join(dataset_dir, "ISIC2018_Task1-2_Training_Input")  # except val
            initial_all_mask_dir = os.path.join(dataset_dir, "ISIC2018_Task1_Training_GroundTruth")  # except val

            utils_others.safe_rename(initial_val_img_dir, val_img_dir)
            utils_others.safe_rename(initial_val_mask_dir, val_mask_dir)
            utils_others.safe_rename(initial_all_img_dir, all_img_dir)
            utils_others.safe_rename(initial_all_mask_dir, all_mask_dir)

            extension_train_img = ".jpg"
            extension_train_mask = "_segmentation.png"
            extension_val_img = ".jpg"
            extension_val_mask = "_segmentation.png"
            extension_test_img = ".jpg"
            extension_test_mask = "_segmentation.png"

            # parameters for resized_images_by_folder
            no_train_resized_images=1000  # recommended = 1000
            no_val_resized_images=100  # recommended = 100
            no_test_resized_images=100  # recommended = 100
            train_step=1  # recommended = 1
            val_step=1  # recommended = 1
            test_step=1  # recommended = 1

            if add_train_and_test and not os.path.exists(test_img_dir):
                utils_others.train_test_split_img_and_mask(
                    all_img_dir, all_mask_dir, train_img_dir, test_img_dir,
                    train_mask_dir, test_mask_dir,
                    extension_train_img, extension_train_mask,
                    extension_test_img, extension_test_mask,
                    data_used_ratio=0.5,  # Use 50% of the data
                    split_ratio=0.2       # Split 20% for testing, 80% for training
                    )

        elif dataset_name == "brats":
            file_id = "1zzX792C87wJ-WQtJvQGjGlPQ2UziJBHk"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"
            utils_others.download_and_extract(gdown_url, data_dir, "brats.zip")

        elif dataset_name == "bmshare":
            file_id = "1DTr-m3tlomBd8ypIfsmfj9uQn4JZj83U"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            utils_others.download_and_extract(gdown_url, data_dir, "bmshare.zip")

        elif dataset_name == "isles":
            file_id = "1ec6pxkEQ_gDQWBqjOIbZhXga9ctzX7Du"
            gdown_url = f"https://drive.google.com/uc?id={file_id}"

            utils_others.download_and_extract(gdown_url, data_dir, "isles.zip")

        else:
            raise ValueError("Dataset not supported")


        # common parameters for 3 datasets: "brats", "bmshare", "isles"
        if dataset_name in ("brats", "bmshare", "isles"):
            train_instruction_txt = os.path.join(dataset_dir, "train.txt")
            val_instruction_txt = os.path.join(dataset_dir, "val.txt")
            test_instruction_txt = os.path.join(dataset_dir, "test.txt")

            extension_train_img = ".png"
            extension_train_mask = ".png"
            extension_val_img = ".png"
            extension_val_mask = ".png"
            extension_test_img = ".png"
            extension_test_mask = ".png"

            # parameters for resized_images_by_folder
            no_train_resized_images=1000  # recommended = 1000
            no_val_resized_images=300  # recommended = 300
            no_test_resized_images=100  # recommended = 100
            train_step=2  # recommended = 2
            val_step=5  # recommended = 5
            test_step=1  # recommended = 1

            utils_others.separate_files_respecting_txt(all_img_dir, val_img_dir, val_instruction_txt)
            utils_others.separate_files_respecting_txt(all_mask_dir, val_mask_dir, val_instruction_txt)
            if add_train_and_test:
                utils_others.separate_files_respecting_txt(all_img_dir, train_img_dir, train_instruction_txt)
                utils_others.separate_files_respecting_txt(all_mask_dir, train_mask_dir, train_instruction_txt)
                utils_others.separate_files_respecting_txt(all_img_dir, test_img_dir, test_instruction_txt)
                utils_others.separate_files_respecting_txt(all_mask_dir, test_mask_dir, test_instruction_txt)


        ########################################################

        resized_images_and_masks_dir = os.path.join(dataset_dir, "resized_images_and_masks")
        labels_dir = os.path.join(dataset_dir, "labels")

        os.makedirs(resized_images_and_masks_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        ########################################################

        resized_val_img_dir = os.path.join(resized_images_and_masks_dir, "resized_val_img_dir")
        resized_val_mask_dir = os.path.join(resized_images_and_masks_dir, "resized_val_mask_dir")
        resized_train_img_dir = os.path.join(resized_images_and_masks_dir, "resized_train_images")
        resized_train_mask_dir = os.path.join(resized_images_and_masks_dir, "resized_train_masks")
        resized_test_img_dir = os.path.join(resized_images_and_masks_dir, "resized_test_images")
        resized_test_masks_dir = os.path.join(resized_images_and_masks_dir, "resized_test_masks")

        train_voc_label_dir = os.path.join(labels_dir, "train_voc_labels")
        val_voc_label_dir = os.path.join(labels_dir, "val_voc_labels")
        test_voc_label_dir = os.path.join(labels_dir, "test_voc_labels")

        train_yolo_label_dir = os.path.join(labels_dir, "train_yolo_labels")
        val_yolo_label_dir = os.path.join(labels_dir, "val_yolo_labels")
        test_yolo_label_dir = os.path.join(labels_dir, "test_yolo_labels")

        train_coco_label_dir = os.path.join(labels_dir, "train_coco_labels")
        val_coco_label_dir = os.path.join(labels_dir, "val_coco_labels")
        test_coco_label_dir = os.path.join(labels_dir, "test_coco_labels")

        results_visualisation_dir = os.path.join(dataset_dir, "visualisation")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam_stats_dir = os.path.join(dataset_dir, "statistics_SAM")
        medsam_stats_dir = os.path.join(dataset_dir, "statistics_MedSAM")

        extension_voc_coords = ".txt"
        extension_yolo_coords = ".txt"

        resized_img_dim = (256, 256)
        # extension_val_yolo_coords = extension_val_mask.replace(extension_val_mask.split(".")[-1], "txt")

        ###############################################################

        utils_others.resize_images_by_folder(val_img_dir, resized_val_img_dir, max_images=max_images//10, extension_filter=extension_val_img, target_size=resized_img_dim)
        utils_others.resize_images_by_folder(val_mask_dir, resized_val_mask_dir, max_images=max_images//10, extension_filter=extension_val_mask, target_size=resized_img_dim)
        if add_train_and_test:
            utils_others.resize_images_by_folder(train_img_dir, resized_train_img_dir, max_images=max_images, extension_filter=extension_train_img, target_size=resized_img_dim)
            utils_others.resize_images_by_folder(train_mask_dir, resized_train_mask_dir, max_images=max_images, extension_filter=extension_train_mask, target_size=resized_img_dim)
            utils_others.resize_images_by_folder(test_img_dir, resized_test_img_dir, max_images=max_images//10, extension_filter=extension_test_img, target_size=resized_img_dim)
            utils_others.resize_images_by_folder(test_mask_dir, resized_test_masks_dir, max_images=max_images//10, extension_filter=extension_test_mask, target_size=resized_img_dim)

        ###############################################################

        utils_others.mask_to_voc_labels(resized_val_mask_dir, val_voc_label_dir, mask_extension=extension_val_mask)
        if add_train_and_test:
            utils_others.mask_to_voc_labels(resized_train_mask_dir, train_voc_label_dir, mask_extension=extension_train_mask)
            utils_others.mask_to_voc_labels(resized_test_masks_dir, test_voc_label_dir, mask_extension=extension_test_mask)

        utils_others.voc_to_coco_xml(val_voc_label_dir, val_coco_label_dir, img_size=resized_img_dim)
        if add_train_and_test:
            utils_others.voc_to_coco_xml(train_voc_label_dir, train_coco_label_dir, img_size=resized_img_dim)
            utils_others.voc_to_coco_xml(test_voc_label_dir, test_coco_label_dir, img_size=resized_img_dim)

        utils_others.voc_to_yolo_labels(val_voc_label_dir, val_yolo_label_dir)
        if add_train_and_test:
            utils_others.voc_to_yolo_labels(train_voc_label_dir, train_yolo_label_dir)
            utils_others.voc_to_yolo_labels(test_voc_label_dir, test_yolo_label_dir)


        #############################################################

        import shutil

        def organize_yolo_split(image_dir, label_dir, output_image_dir, output_label_dir):
            """
            Organizes a single dataset split (train, val, or test) into YOLO format.
            Moves images and corresponding labels to the correct directories.
            Creates empty label files for images without annotations.
            """
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            for image_file in os.listdir(image_dir):
                if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(image_dir, image_file)
                    label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

                    # Move image
                    shutil.copy(image_path, os.path.join(output_image_dir, image_file))

                    # Move label if it exists, else create an empty label file
                    if os.path.exists(label_path):
                        shutil.copy(label_path, os.path.join(output_label_dir, os.path.basename(label_path)))
                    else:
                        open(os.path.join(output_label_dir, os.path.basename(label_path)), 'w').close()

            print(f"{output_image_dir} and {output_label_dir} organized successfully!")

        yolo_data_dir = os.path.join(dataset_dir, "yolo_data")
        os.makedirs(yolo_data_dir, exist_ok=True)

        splits = {
            "train": {
                "images": resized_train_img_dir,
                "labels": train_yolo_label_dir
            },
            "val": {
                "images": resized_val_img_dir,
                "labels": val_yolo_label_dir
            },
            "test": {
                "images": resized_test_img_dir,
                "labels": test_yolo_label_dir
            }
        }

        for split, paths in splits.items():
            split_dir = os.path.join(yolo_data_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            organize_yolo_split(paths["images"], paths["labels"],
                                os.path.join(split_dir, "images"),
                                os.path.join(split_dir, "labels"))


        #########################################################################

        yolo_yaml_path = os.path.join(dataset_dir, "training_data.yaml")

        dataset_yaml = f"""
        train: {os.path.join(yolo_data_dir, "train")}
        val: {os.path.join(yolo_data_dir, "val")}

        nc: 1
        names: ['object']
        """
        with open(yolo_yaml_path, "w") as f:
            f.write(dataset_yaml)
