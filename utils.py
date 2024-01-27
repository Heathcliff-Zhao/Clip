from PIL import Image, ImageDraw
import json
import gpustat

def get_available_gpu():
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_free_mem = [gpu.memory_free for gpu in gpu_stats.gpus]
    gpu_id = gpu_free_mem.index(max(gpu_free_mem))
    return gpu_id

def draw_bboxes(image, bboxes, width=3):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        draw.rectangle(bbox, outline='red', width=width)
    return image

def prepare_image_with_bbox(image, bboxes, modify):
    pass