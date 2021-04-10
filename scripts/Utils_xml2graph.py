import xml.etree.ElementTree as ET
import numpy as np
import glob
import cv2
import copy
import networkx as nx
import json



def load_floorplan(file):
    tree = ET.parse(file)
    root = tree.getroot()
    name = root.get('FloorName')

    ScaleElem = root.find('Scale')
    P2D = float(ScaleElem.attrib['RealDistance']) / float(ScaleElem.attrib['PixelDistance']) * 2

    boundary = {'min_x': 999999999, 'max_x': 0, 'min_y': 999999999, 'max_y': 0}
    centroids = {'name': [], 'pos': [], 'type': [], 'portals': []}
    cent_origin = {'name': [], 'pos': [], 'type': [], 'portals': []}
    lines = {'points': [], 'type': []}
    for child in root.iter('space'):
        space_attr = child.attrib
        space_type = space_attr['type']

        # children of space
        child_space = child.getchildren()
        contour = child_space[0]
        portals = child_space[1:]

        # name
        centroids['name'].append(child.attrib['name'])
        cent_origin['name'].append(child.attrib['name'])

        # portals
        portals_this = []
        for portal in portals:
            portals_this.append(portal.attrib['target'])
        centroids['portals'].append(portals_this)
        cent_origin['type'].append(portals_this)

        # children of contour
        child_contour = contour.getchildren()

        # centroid
        centroid = child_contour[0]
        cen_x, cen_y = float(centroid.attrib['x']) * P2D, float(centroid.attrib['y']) * P2D
        centroids['pos'].append([cen_x, cen_y])
        centroids['type'].append(space_type)

        cent_origin['pos'].append([centroid.attrib['x'], centroid.attrib['y']])
        cent_origin['type'].append(space_type)

        # linesegments
        linesegments = child_contour[1:]
        N_lines = len(linesegments)
        for i_l in range(0, N_lines):
            line_attr = linesegments[i_l].attrib
            line_type = line_attr['type']

            x1, x2, y1, y2 = float(line_attr['x1']) * P2D, float(line_attr['x2']) * P2D, float(
                line_attr['y1']) * P2D, float(line_attr['y2']) * P2D
            x_values = [x1, x2]
            y_values = [y1, y2]
            lines['points'].append([x_values, y_values])
            lines['type'].append(line_type)

            # map boundary
            boundary['max_x'] = max(boundary['max_x'], x1, x2)
            boundary['min_x'] = min(boundary['min_x'], x1, x2)
            boundary['max_y'] = max(boundary['max_y'], y1, y2)
            boundary['min_y'] = min(boundary['min_y'], y1, y2)
    return lines, centroids, boundary, cent_origin


def floorplan_xml2components(lines, centroids, boundary, res=0.05):
    N_space = len(centroids['type'])
    N_lines = len(lines['type'])
    max_x, min_x, max_y, min_y = boundary['max_x'], boundary['min_x'], boundary['max_y'], boundary['min_y']
    for i_s in range(0, N_space):
        centroids['pos'][i_s][0] = int(np.floor((centroids['pos'][i_s][0] - min_x) / res))
        centroids['pos'][i_s][1] = int(np.floor((centroids['pos'][i_s][1] - min_y) / res))
    for i_l in range(0, N_lines):
        points = lines['points'][i_l]
        x_points = points[0]
        y_points = points[1]
        lines['points'][i_l] = (
        (int(np.floor((x_points[0] - min_x) / res)), int(np.floor((x_points[1] - min_x) / res))),
        (int(np.floor((y_points[0] - min_y) / res)), int(np.floor((y_points[1] - min_y) / res))))
    boundary['max_x'] = int(np.floor((boundary['max_x'] - min_x) / res))
    boundary['min_x'] = int(np.floor((boundary['min_x'] - min_x) / res))
    boundary['max_y'] = int(np.floor((boundary['max_y'] - min_y) / res))
    boundary['min_y'] = int(np.floor((boundary['min_y'] - min_y) / res))
    return lines, centroids, boundary


def floorplan_xml2img(file, type_img=1, draw_portal=False, draw_text=False, draw_center=False, is_show=False,
                      is_save=False, dir_save='', category_substrs=[], cmap_category=[]):
    # type_img : skeleton (1), category_color (2)
    lines, centroids, boundary, cent_origin = load_floorplan(file)
    lines, centroids, boundary = floorplan_xml2components(lines, centroids, boundary, res=1 / 20)

    len_x = boundary['max_x'] - boundary['min_x'] + 1
    len_y = boundary['max_y'] - boundary['min_y'] + 1
    img = 255 * np.ones((len_y, len_x, 3), np.uint8)

    N_lines = len(lines['type'])
    # draw walls
    for i_l in range(0, N_lines):
        line_type = lines['type'][i_l]
        points = lines['points'][i_l]
        x_points = points[0]
        y_points = points[1]
        if line_type == 'Wall' or line_type == 'Window':
            img = cv2.line(img, (x_points[0], y_points[0]), (x_points[1], y_points[1]), color=(0, 0, 0), thickness=2)

    if type_img == 1:
        pass
    elif type_img == 2:
        for i_l in range(0, N_lines):
            line_type = lines['type'][i_l]
            points = lines['points'][i_l]
            x_points = points[0]
            y_points = points[1]
            if line_type == 'Portal':
                img = cv2.line(img, (x_points[0], y_points[0]), (x_points[1], y_points[1]), color=(255, 255, 0))

        rows, cols = img.shape[:2]
        mask = np.zeros((rows + 2, cols + 2), np.uint8)
        newVal = (255, 0, 0)
        loDiff, upDiff = (1, 1, 1), (1, 1, 1)
        N_space = len(centroids['type'])

        log_on = False
        for i_s in range(0, N_space):
            cen_x = centroids['pos'][i_s][0]
            cen_y = centroids['pos'][i_s][1]
            seed = (cen_x, cen_y)
            type_space = centroids['type'][i_s]
            category_types = {key: [] for key in category_substrs.keys()}
            N_category = len(category_substrs)

            flag_found = False
            if log_on:
                print('------')
                print(type_space)
            for key, substrs in category_substrs.items():
                for substr in substrs:
                    if substr in type_space:
                        if log_on:
                            print(substr)
                        category_space = key
                        flag_found = True
                        break
                if flag_found == True:
                    break
            if log_on:
                print(category_space)

            color_space = cmap_category[category_space]
            cv2.floodFill(img, mask, seed, color_space, loDiff, upDiff)

            # remove portal
            for i_l in range(0, N_lines):
                line_type = lines['type'][i_l]
                points = lines['points'][i_l]
                x_points = points[0]
                y_points = points[1]
                if line_type == 'Portal':
                    img = cv2.line(img, (x_points[0], y_points[0]), (x_points[1], y_points[1]), color=(255, 255, 0))

    if draw_center == True:
        for i_s in range(0, N_space):
            cen_x = centroids['pos'][i_s][0]
            cen_y = centroids['pos'][i_s][1]
            cv2.circle(img, (cen_x, cen_y), radius=2, color=(0, 0, 0))

    if draw_text == True:
        # text
        N_space = len(centroids['type'])
        for i_s in range(0, N_space):
            cen_x = centroids['pos'][i_s][0]
            cen_y = centroids['pos'][i_s][1]

            cen_x_origin = cent_origin['pos'][i_s][0]
            cen_y_origin = cent_origin['pos'][i_s][1]
            if centroids['type'][i_s] == '':
                cv2.putText(img, str((cen_x_origin, cen_y_origin)), (cen_x, cen_y), cv2.FONT_HERSHEY_PLAIN, 1.,
                            (0, 0, 255))
            else:
                cv2.putText(img, centroids['type'][i_s], (cen_x, cen_y), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 0))

    if draw_portal == True:
        # draw portal
        for i_l in range(0, N_lines):
            line_type = lines['type'][i_l]
            points = lines['points'][i_l]
            x_points = points[0]
            y_points = points[1]
            if line_type == 'Portal':
                img = cv2.line(img, (x_points[0], y_points[0]), (x_points[1], y_points[1]), color=(255, 255, 0))

    if is_show == True:
        cv2.imshow(img)

    if is_save == True:
        fname = file.split('/')[-1].split('.xml')[0]
        cv2.imwrite(dir_save + fname + '.png', img)


def get_topology_from_xml(file, category_substrs):
    lines, centroids, boundary, centroids_origin = load_floorplan(file)
    lines, centroids, boundary = floorplan_xml2components(lines, centroids, boundary, res=1 / 20)

    G = nx.Graph()
    N_space = len(centroids['name'])
    # add nodes
    for i_s in range(0, N_space):
        G.add_nodes_from([(i_s, {"name": centroids['name'][i_s], "pos": centroids['pos'][i_s]})])
        type_space = centroids['type'][i_s]
        flag_found = False
        for key, substrs in category_substrs.items():
            for substr in substrs:
                if substr in type_space:
                    category_space = key
                    flag_found = True
                    G.nodes[i_s]['type'] = category_space
                    break
            if flag_found == True:
                break

    # add edges
    for i_s in range(0, N_space):
        n_list = [x for x, y in G.nodes(data=True) if y['name'] in centroids['portals'][i_s]]
        e_list = [(i_s, i_f) for i_f in n_list]
        G.add_edges_from(e_list)
    return G
