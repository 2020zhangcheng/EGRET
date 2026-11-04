# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Annotation parsing: token positive maps, box<->points and scene graphs."""

import numpy as np
import torch

from . import sng_parser


# BRIEF Construct position label(map)
def get_positive_map(tokenized, tokens_positive):
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)  # ([positive], 256])
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)


#########################
# BRIEF Text decoupling #
#########################
def Scene_graph_parse(annos):
    print('Begin text decoupling......')
    for anno in annos:
        caption = ' '.join(anno['utterance'].replace(',', ' , ').split())

        # some error or typo in ScanRefer.
        caption = ' '.join(caption.replace("'m", "am").split())
        caption = ' '.join(caption.replace("'s", "is").split())
        caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
        caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
        caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
        caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
        caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
        caption = ' '.join(caption.replace("2-door", "2 - door").split())
        caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
        caption = ' '.join(caption.replace("computer/", "computer /").split())
        caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
        caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
        caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
        caption = ' '.join(caption.replace("theses", "these").split())
        
        # some error or typo in NR3D.
        if anno['dataset'] == 'nr3d':
            caption = ' '.join(caption.replace('.', ' .').split())
            caption = ' '.join(caption.replace(';', ' ; ').split())
            caption = ' '.join(caption.replace('-', ' ').split())
            caption = ' '.join(caption.replace('"', ' ').split())
            caption = ' '.join(caption.replace('?', ' ').split())
            caption = ' '.join(caption.replace("*", " ").split())
            caption = ' '.join(caption.replace(':', ' ').split())
            caption = ' '.join(caption.replace('$', ' ').split())
            caption = ' '.join(caption.replace("#", " ").split())
            caption = ' '.join(caption.replace("/", " / ").split())
            caption = ' '.join(caption.replace("you're", "you are").split())
            caption = ' '.join(caption.replace("isn't", "is not").split())
            caption = ' '.join(caption.replace("thats", "that is").split())
            caption = ' '.join(caption.replace("doesn't", "does not").split())
            caption = ' '.join(caption.replace("doesnt", "does not").split())
            caption = ' '.join(caption.replace("itis", "it is").split())
            caption = ' '.join(caption.replace("left-hand", "left - hand").split())
            caption = ' '.join(caption.replace("[", " [ ").split())
            caption = ' '.join(caption.replace("]", " ] ").split())
            caption = ' '.join(caption.replace("(", " ( ").split())
            caption = ' '.join(caption.replace(")", " ) ").split())
            caption = ' '.join(caption.replace("wheel-chair", "wheel - chair").split())
            caption = ' '.join(caption.replace(";s", "is").split())
            caption = ' '.join(caption.replace("tha=e", "the").split())
            caption = ' '.join(caption.replace("it’s", "it is").split())
            caption = ' '.join(caption.replace("’s", " is").split())
            caption = ' '.join(caption.replace("isnt", "is not").split())
            caption = ' '.join(caption.replace("Don't", "Do not").split())
            caption = ' '.join(caption.replace("arent", "are not").split())
            caption = ' '.join(caption.replace("cant", "can not").split())
            caption = ' '.join(caption.replace("you’re", "you are").split())
            caption = ' '.join(caption.replace('!', ' !').split())
            caption = ' '.join(caption.replace('id the', ' , the').split())
            caption = ' '.join(caption.replace('youre', 'you are').split())

            caption = ' '.join(caption.replace("'", ' ').split())

            if caption[0] == "'":
                caption = caption[1:]
            if caption[-1] == "'":
                caption = caption[:-1]
        
        anno['utterance'] = caption

        # text parsing
        graph_node, graph_edge = sng_parser.parse(caption)

        # NOTE If no node is parsed, add "this is an object ." at the beginning of the sentence
        if (len(graph_node) < 1) or \
            (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
            caption = "This is an object . " + caption
            anno['utterance'] = caption

            # parse again
            graph_node, graph_edge = sng_parser.parse(caption)

        # node and edge
        anno["graph_node"] = graph_node
        anno["graph_edge"] = graph_edge

        # auxi object
        auxi_entity = None
        for node in graph_node:
            if (node["node_id"] != 0) and (node["node_type"] == "Object"):
                auxi_entity = node
                break
        anno["auxi_entity"] = auxi_entity
    
    print('End text decoupling!')
