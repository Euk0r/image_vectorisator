import os
import pathlib
import configparser
import tempfile
import subprocess
import time
from PyQt5.QtWidgets import QProgressBar
#from .workers import BuildingsWorker
from .ImageWorker import ImageWorker
import shutil

#import all files from workers directory
dir_workers = os.path.join(pathlib.Path(__file__).parent.resolve(), "workers")
import importlib
from . import importdir
imported_workers = False
if not imported_workers:
    importdir.do(dir_workers, globals())
    imported_workers = True

all_workers = importdir.__get_module_names_in_dir(dir_workers)


class ModulesManager:
    __dir_modules = ""
    __full_path = ""
    __configs = []
    __amount = 0
    __module_names = []
    __dir_temp_results = ""
    __segmented_tag = "-segmented"
    module_dict = {}
    instance_dict = {}


    @property 
    def amount(self): 
        return self.__amount

    @property 
    def module_names(self): 
        return self.__module_names

    def __init__(self, full_path, dir_modules = "modules/"):
        self.__full_path = full_path + '/'
        self.__dir_modules = dir_modules
        self.__dir_temp_results = self.__full_path + "temp/"
        self.checkModules()
        #init workers
        for id, worker in enumerate(all_workers):
            module = importlib.import_module(worker)
            class_ = getattr(module, worker)
            instance = class_()
            self.module_dict[id] = self.__configs[id]["SYSTEM"]["WORKER_NAME"]
            self.instance_dict[self.__configs[id]["SYSTEM"]["WORKER_NAME"]] = instance

    def checkModules(self):
        directory = self.__full_path + self.__dir_modules
        file = "settings.ini"
        for path in pathlib.Path(directory).rglob(file):
            config = configparser.ConfigParser()
            config.read(path, encoding="utf-8")
            self.__configs.append(config)
            self.__module_names.append(config["COMMON"]["NAME"])
        self.__amount = len(self.__configs)

    def getModuleInfo(self,id,type="COMMON"):
        return self.__configs[id][type]

    def processModule(self, id_module, image_path, progressbar: QProgressBar):
        progressbar.setValue(0)
        directory = os.path.join(self.__dir_temp_results, image_path.split('/')[-1].split('.')[-0] + self.__segmented_tag)

        if True: #not(os.path.exists(directory))
            print(directory + " not exists")

            info = self.getModuleInfo(id_module,"SYSTEM")
            imageW = ImageWorker(image_path)
            python_bin = self.__full_path + self.__dir_modules + info["FOLDER_NAME"] + info["PATH_PYTHON"]
            script_file = self.__full_path + self.__dir_modules + info["FOLDER_NAME"] + info["PATH_SCRIPT"]

            img_ext = image_path.split('.')[-1]
            imageW.preprocessImage(int(info["HEIGHT"]), int(info["WIDTH"]), img_ext, os.path.join(directory, info["FOLDER_TILES"]))

            dir_tiles_segmented = os.path.join(directory, info["FOLDER_TILES_SEGMENTED"])
            if not os.path.exists(dir_tiles_segmented):
                os.makedirs(dir_tiles_segmented)

            p = subprocess.Popen([python_bin, script_file],
                                 cwd=directory,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=True)

            while p.poll() == None:
                percent = imageW.checkProgress(os.path.join(directory, info["FOLDER_TILES_SEGMENTED"]))
                progressbar.setValue(percent)
                if percent == 100:
                    break
                time.sleep(3)

            imgpath_segmented = imageW.processTiles(int(info["HEIGHT"]), int(info["WIDTH"]), os.path.join(directory, info["FOLDER_TILES_SEGMENTED"]), self.__dir_temp_results, self.__segmented_tag)
            print(imgpath_segmented)
        else:
            print(directory + "  exists")
            print("trying to use old data")
            imgpath_segmented = os.path.join(self.__dir_temp_results,image_path.split('/')[-1].replace('.', self.__segmented_tag + '.'))
            print(imgpath_segmented)

        progressbar.setValue(100)
        return imgpath_segmented

    def vectorise(self, img_name_ext, dir_image, id_module):
        temp = self.module_dict[id_module]
        vec = self.instance_dict[temp]
        return vec.vectorise(img_name_ext, dir_image, self.__dir_temp_results)


    def save_vector(self, img_name_ext, path_to_save, id_module, offset=None):
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        temp = self.module_dict[id_module]
        vec = self.instance_dict[temp]
        if os.path.exists(path):
            return vec.save_shp(path, path_to_save, offset)

    def get_working_dir(self, img_name_ext):
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        if (os.path.exists(path)):
            return path
        else:
            return None

    def clean_up(self, img_name_ext):
        """Delete temporal files"""
        path = os.path.join(self.__dir_temp_results, os.path.splitext(img_name_ext)[0])
        path_file = os.path.join(self.__dir_temp_results, img_name_ext)
        if (os.path.exists(path)):
            shutil.rmtree(path, ignore_errors=True)
            os.remove(path_file)