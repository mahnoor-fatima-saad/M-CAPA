import numpy as np 
import os 
import json
from utils import read_npz

class MaterialMapper:
    def __init__(self, configs_dir, config):
        self.config = config
        self.configs_dir = configs_dir
        self.material_map_dict = {}
        self.MP3D_NAME_TO_INDEX_MAPPING = {
        "void": 0,
        "wall": 1,
        "floor": 2, 
        "chair": 3, 
        "door": 4, 
        "table": 5, 
        "picture": 6, 
        "cabinet": 7,
        "cushion": 8,
        "window": 9,
        "sofa": 10,
        "bed": 11,
        "curtain": 12,
        "chest_of_drawers": 13,
        "plant": 14,
        "sink": 15, 
        "stairs": 16,
        "ceiling": 17,
        "toilet": 18,
        "stool": 19,
        "towel": 20,
        "mirror": 21,
        "tv_monitor": 22,
        "shower": 23,
        "column": 24,
        "bathtub": 25,
        "counter": 26,
        "fireplace": 27,
        "lighting": 28,
        "beam": 29,
        "railing": 30,
        "shelving": 31,
        "blinds": 32,
        "gym_equipment": 33,
        "seating": 34,
        "board_panel": 35,
        "furniture": 36,
        "appliances": 37,
        "clothes": 38,
        "objects": 39,
        "misc": 40,
        " ": 41,
        "":41, 
        "unlabelled":41                                  
        }
        # Ade-id : mp3d-id
        
        self.ADE_CONVERSION_DICT = {
        "wall" : [0],
        "ceiling" : [5],
        "floor" : [3, 6, 11, 28],
        "plant" : [4, 9, 13, 17, 29, 66, 72, 91, 94,68],
        "furniture" : [7, 23, 51, 55, 99, 111, 117],
        "window" : [8],
        "chest_of_drawers" : [10, 35, 44],
        "misc" : [2,12, 21, 26, 46, 54,60, 76, 80, 83, 90, 102, 103, 113, 120, 123, 126, 128, 136,147],
        "door" : [14, 58],
        "table" : [15, 16, 33, 56, 64],
        "curtain" : [18],
        "chair" : [19, 30, 31, 75,97],
        "appliances" : [20, 50, 71, 78, 105, 107, 116, 118, 122, 124, 125, 127, 129, 133, 139, 146],
        "counter" : [45, 70, 73],
        "mirror" : [27],
        "objects" : [32, 34, 40, 41, 67,22, 98, 104,108, 112, 119, 132,135, 137, 138, 142,148,100],
        "lighting" : [36, 82, 85, 87, 134],
        "bathtub" : [37, 109],
        "railing" : [38, 95],
        "cushion" : [39],
        "column" : [42,84,48,25,1,101,52],
        "board_panel" : [43, 144,79,14],
        "sink" : [47],
        "fireplace" : [49],
        "stairs" : [53, 59, 96, 121],
        "bed" : [57],
        "beam" : [61, 88, 93],
        "blinds" : [63],
        "toilet" : [65],
        "stool" : [69, 110],
        "tv_monitor" : [74, 89, 130, 141, 143],
        "shelving" : [77],
        "towel" : [81],
        "clothes" : [86, 92, 106, 114, 115, 131, 149],
        "shower" : [145],
        "void": [],
        "cabinet": [62,67,24]

        }


                

        
       
    def create_int_to_string_semantic_mapping(self, semantic_array):
        index_to_name_mapping = np.array([k for k, v in sorted(self.MP3D_NAME_TO_INDEX_MAPPING.items(), key=lambda item: item[1])]) #{v: k for k, v in self.MP3D_NAME_TO_INDEX_MAPPING.items()}
        string_semantic_array = np.take(index_to_name_mapping, semantic_array)
        return string_semantic_array

    def parse_material_json(self, json_data):
        sem_id_to_id = {}
        for material in json_data.get("materials", []):
            material_id = int(material.get('id'))
            labels = material.get('labels', [])
            for label in labels: 
                if self.config["ade_material_mapping"]:
                    #print("calculating material mask", flush=True)
                    if label in self.ADE_CONVERSION_DICT:
                        sem_ids = self.ADE_CONVERSION_DICT[label]
                        for index in sem_ids:
                            sem_id_to_id[index] = material_id
                else:
                    if label in self.MP3D_NAME_TO_INDEX_MAPPING:
                        index = self.MP3D_NAME_TO_INDEX_MAPPING[label]
                        sem_id_to_id[index] = material_id
                    
                    
                
        return sem_id_to_id

    def read_material_json_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    

    
    def create_mapping_dict_for_all_configs(self):
        all_configs = os.listdir(self.configs_dir)
        for config in all_configs:
            material_json = self.read_material_json_file(os.path.join(self.configs_dir, config))
            material_map = self.parse_material_json(material_json)
            self.material_map_dict[int(config.split('.')[0])] = material_map



    def map_material_to_semantic_image(self, material_index, semantic_data, material_image_dir=None):
        sem_id_to_material_id = self.material_map_dict[int(material_index)]
        mapped_image = np.zeros_like(semantic_data)
        unique_ids = np.unique(semantic_data)
        for sem_id in unique_ids:
            mapped_image[semantic_data == sem_id] = sem_id_to_material_id.get(sem_id, 0)

        if material_image_dir is None: 
            return mapped_image
      
        
    def compare_two_mappings(self, array1, array2, material_index, semantic_data):
        if np.array_equal(array1, array2):
            return True
        else: 
            differences = np.where(array1!=array2)
            for diff in zip(*differences):
                print(f"Difference at index {diff}: original value {array1[diff]}, new value {array2[diff]}", flush=True)
            print("Material Dict", self.material_map_dict[int(material_index)])
            print("original semantic image", semantic_data)
            return False
        
    def map_material_to_semantic_images_orientations(self, material_index, semantic_images):
        material_maps = []
        for semantic_data in semantic_images:
            material_maps.append(self.map_material_to_semantic_image(material_index=material_index, semantic_data=semantic_data))
        return material_maps
    

