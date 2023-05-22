import glob
import os
from PIL import Image

class ImageWorker:
    amount_tiles=0
    tiles_in_height=0
    tiles_in_width=0
    old_size = None
    
    def __init__(self, image_path):
        self.__image_path = image_path
    
    def preprocessImage(self, height, width, format_img, dir_tiles):
        if not os.path.exists(dir_tiles):
            os.makedirs(dir_tiles)
        old_im = Image.open(self.__image_path)
        self.old_size = old_im.size

        new_im = addLabels(old_im, height, width)
        new_size = new_im.size

        new_im.paste(old_im)
        #new_im.paste(old_im, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2))) #поля по краям

        start_num=0
        self.amount_tiles=0
        self.tiles_in_height, self.tiles_in_width = new_size
        self.tiles_in_height /= height
        self.tiles_in_width /= width
        for k,piece in enumerate(crop(new_im,height,width),start_num):
            img=Image.new('RGB', (height,width), 255)
            img.paste(piece)
            path=os.path.join(dir_tiles,"%(num)s.%(format)s" % {"num": k, "format": format_img})
            img.save(path)
            self.amount_tiles+=1

    def processTiles(self, height, width, dir_tiles, dir_save, segmented_tag='-segmented'):
        tiles = []
        for file_name in glob.glob(dir_tiles+'/*.*'):     
            tiles.append((file_name, int(file_name.split('\\')[-1].split('.')[-2])))
        tiles.sort(key=lambda x: x[1])
        imgheight, imgwidth = self.old_size
        img = Image.new("RGB", (imgheight, imgwidth))
        counter = 0
        range_i = imgheight//height
        range_j = imgwidth//width
        if imgheight%height>0: range_i+=1
        if imgwidth%width>0: range_j+=1
        for i in range(range_j):
            for j in range(range_i):
                tempimg = Image.open(tiles[counter][0])
                img.paste(tempimg, (j*width, i*height))
                counter += 1
        path=os.path.join(dir_save,self.__image_path.split('/')[-1].replace('.',segmented_tag + '.'))
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        img.save(path)
        return path
    
    def checkProgress(self, dir_tiles):
        return int((len(glob.glob(dir_tiles+'/*.*'))*100)/self.amount_tiles)

def crop(im,height,width):
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)

def addLabels(im,height,width):
    imgwidth, imgheight = im.size
    if (imgwidth % width == 0 and imgheight % height == 0): return im
    else:
        imgwidth += width - (imgwidth % width)
        imgheight += height - (imgheight % height)
        img = Image.new("RGB", (imgwidth, imgheight))
        return img