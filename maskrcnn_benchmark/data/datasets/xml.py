import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, cfg, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        self._annopath = os.path.join(self.root, "xmls", "%s.xml")
        self._imgpath = os.path.join(self.root, "images", "%s.png")
        self._imgsetpath = os.path.join(self.root, "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.classnames = cfg.DATASETS.CLASSNAMES
        self.class_to_ind = dict(zip(self.classnames, range(len(self.classnames))))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        masks = SegmentationMask(anno["masks"], (width, height), mode='poly')
        target.add_field("masks", masks)

        target.add_field("labels", anno["labels"])
        # target.add_field("difficult", anno["difficult"])
        return target

    def get_polygon_from_obj(self, objs):
        polygons = []
        for obj in objs:
            points = obj.findall('point')
            poly = []
            for pp in points:
                pos = list(map(int, pp.text.split(',')))
                poly.append(pos[0])
                poly.append(pos[1])
            polygons.append(poly)
        return polygons
    
    def get_bbox_from_polygon(self, polygons):
        x1 = 10000
        y1 = 10000
        x2 = -1
        y2 = -1
        for polygon in polygons:
            xs = [polygon[i] for i in range(len(polygon)) if i % 2 == 0]
            ys = [polygon[i] for i in range(len(polygon)) if i % 2 == 1]
            x1 = min(min(xs), x1)
            y1 = min(min(ys), y1)
            x2 = max(max(xs), x2)
            y2 = max(max(ys), y2)
        return x1, y1, x2, y2

    def _preprocess_annotation(self, target):
        boxes = []
        masks = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1
        
        for obj in target.iter("object"):
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            name = obj.find("name").text.strip()
            if name not in self.classnames:
                print('{} not in dataset {}'.format(name, self.classnames))
                continue
            gt_classes.append(self.class_to_ind[name])
            # difficult_boxes.append(difficult)

            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            if bb is not None:
                box = [
                    bb.find("xmin").text, 
                    bb.find("ymin").text, 
                    bb.find("xmax").text, 
                    bb.find("ymax").text,
                ]
                bndbox = tuple(
                    map(lambda x: x - TO_REMOVE, list(map(int, box)))
                )
                boxes.append(bndbox)
            
            segms = obj.findall('polygon')
            polygons = self.get_polygon_from_obj(segms)
            masks.append(polygons)
            if bb is None:
                x1, y1, x2, y2 = self.get_bbox_from_polygon(polygons)
                bndbox = tuple((x1, y1, x2, y2))
                boxes.append(bndbox)
            
            


        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "masks": masks,
            "labels": torch.tensor(gt_classes),
            # "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return self.classnames[class_id]
