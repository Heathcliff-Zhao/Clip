from PIL import ImageDraw
import gpustat
import random
from copy import deepcopy


def leaf_list(element):
    leaf_node_list = []

    def get_leaf_node(element):
        if 'children' not in element:
            if (element.get('tagName', None) == 'img' or element.get('type', None) == 'text'):
                if element['boxInfo']['height'] > 0 and element['boxInfo']['width'] > 0 and element['boxInfo']['top'] >= 0 and element['boxInfo']['left'] >= 0:
                    leaf_node_list.append(element)
        else:
            
            for child in element['children']:
                get_leaf_node(child)

    get_leaf_node(element)
    return leaf_node_list

def draw_bbox(img, bboxes_info, width=3, color='red'):
    draw = ImageDraw.Draw(img)
    for node in bboxes_info:
        node = deepcopy(node['boxInfo'])
        draw.rectangle([(node['left'], node['top']), (node['left'] + node['width'], node['top'] + node['height'])], outline=color, width=width)
    return img

def get_available_gpu():
    gpu_stats = gpustat.GPUStatCollection.new_query()
    gpu_free_mem = [gpu.memory_free for gpu in gpu_stats.gpus]
    gpu_id = gpu_free_mem.index(max(gpu_free_mem))
    return gpu_id

def disturb_on_bbox(bboxes_info, image_size):
    # move, add
    which_disturb = random.choice(['move', 'add'])
    if which_disturb == 'move':
        for node in bboxes_info:
            move_step = random.randint(50, 100)
            direction = random.choice(['up', 'down', 'left', 'right'])
            if direction == 'up':
                node['boxInfo']['top'] -= move_step
            elif direction == 'down':
                node['boxInfo']['top'] += move_step
            elif direction == 'left':
                node['boxInfo']['left'] -= move_step
            elif direction == 'right':
                node['boxInfo']['left'] += move_step
            else:
                pass
    elif which_disturb == 'add':
        # add random box on image, region limited in horizontal: 0.2 ~ 0.8, vertical: 0.1 ~ 0.3
        columns = random.randint(2, 4)
        rows = random.randint(5, 10)
        width = image_size[0] * (0.8 - 0.2 - 0.1) / columns
        height = image_size[1] * (0.3 - 0.1) / rows
        for i in range(rows):
            for j in range(columns):
                left = int(image_size[0] * 0.2 + image_size[0] * 0.1 / 2 + j * width)
                top = int(image_size[1] * 0.1 + i * height)
                bboxes_info.append({'boxInfo': {'left': left, 'top': top, 'width': width, 'height': height}})
    else:
        pass
    return bboxes_info


def white_cover(image, bboxes):
    for node in bboxes:
        node = node['boxInfo']
        image.paste((255, 255, 255), (node['left'], node['top'], node['left'] + node['width'], node['top'] + node['height']))
    return image

def disturb_on_image(image, bboxes_info):
    image_node = [node for node in bboxes_info if node.get('tagName', None) == 'img']
    # 15% mask
    mask_ratio = 0.15
    mask_bool_list = [True if random.random() < mask_ratio else False for _ in range(len(image_node))]
    # at least one image node is masked
    if not any(mask_bool_list):
        mask_bool_list[random.randint(0, len(image_node) - 1)] = True
    selected_image_node = [node for node, mask_bool in zip(image_node, mask_bool_list) if mask_bool]
    # white cover on image, when mask_bool_list[i] is True
    image = white_cover(image, selected_image_node)
    # TODO: random drop image node ???? much bolder line required
    return image


def prepare_image_with_bbox(image, page_structure, disturb, width=3):
    leaf_node_list = leaf_list(page_structure)
    if disturb:
        which_disturb = random.choice(['bbox', 'image'])
        if which_disturb == 'bbox':
            disturb_on_bbox(leaf_node_list, image.size)
        else:
            disturb_on_image(image, leaf_node_list)
    img_with_bbox = draw_bbox(image, leaf_node_list, width=width)
    return img_with_bbox

def accuracy_score(labels, predictions):
    correct = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            correct += 1
    return correct / len(labels)

def recall_score(labels, predictions):
    # print(labels)
    # print(predictions)
    tp = 0
    fn = 0
    for label, prediction in zip(labels, predictions):
        # print(label, prediction)
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 1 and prediction == 0:
            fn += 1
    return tp / (tp + fn)

def precision_score(labels, predictions):
    tp = 0
    fp = 0
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 1:
            fp += 1
    return tp / (tp + fp)
