import numpy as np

def parse_obj_file(file_path):
    vertices = []
    normals = []
    faces = []
    uvs = []
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, start=1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if not parts:
                    continue
                prefix = parts[0]
                if prefix == 'v':
                    if len(parts) < 4:
                        print(f"Warning: Line {line_num} has insufficient vertex data: '{line}'")
                        continue
                    try:
                        x, y, z = map(float, parts[1:4])
                        vertices.append((x, y, z))
                    except ValueError:
                        print(f"Warning: Line {line_num} has invalid vertex coordinates: '{line}'")
                        continue
                elif prefix == 'vn':
                    # Parse vertex normals
                    if len(parts) < 4:
                        print(f"Warning: Line {line_num} has insufficient normal data: '{line}'")
                        continue
                    try:
                        nx, ny, nz = map(float, parts[1:4])
                        normals.append((nx, ny, nz))
                    except ValueError:
                        print(f"Warning: Line {line_num} has invalid normal coordinates: '{line}'")
                        continue
                elif prefix == 'vt':
                    # Parse texture coordinates
                    if len(parts) < 3:
                        print(f"Warning: Line {line_num} has insufficient texture coordinate data: '{line}'")
                        continue
                    try:
                        u, v = map(float, parts[1:3])
                        uvs.append((u, v))
                    except ValueError:
                        print(f"Warning: Line {line_num} has invalid texture coordinates: '{line}'")
                        continue
                elif prefix == 'f':
                    face = []
                    '''
                    Consider the following formats:
                    f v1 v2 v3 ...
                    f v1/vt1 v2/vt2 v3/vt3 ...
                    f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
                    f v1//vn1 v2//vn2 v3//vn3 ...
                    '''
                    for v in parts[1:]:
                        if '/' not in v: # f v1 v2 v3 ...
                            face.append(int(v))
                        elif '//' in v: # f v1//vn1 v2//vn2 ...
                            v_idx_str, vn_idx_str = v.split('//')
                            try:
                                v_idx = int(v_idx_str)
                                vn_idx = int(vn_idx_str)
                                face.append((v_idx, vn_idx))
                            except ValueError:
                                print(f"Warning: Line {line_num} has invalid face indices: '{v}'")
                                continue
                        else:
                            if len(v.split('/')) == 3: # f v1/vt1/vn1 v2/vt2/vn2 ...
                                v_idx_str, vt_idx_str, vn_idx_str = v.split('/')
                                try:
                                    v_idx = int(v_idx_str)
                                    vt_idx = int(vt_idx_str)
                                    vn_idx = int(vn_idx_str)
                                    face.append((v_idx, vt_idx, vn_idx))
                                except ValueError:
                                    print(f"Warning: Line {line_num} has invalid face indices: '{v}'")
                                    continue
                            elif len(v.split('/')) == 2: # f v1/vt1 v2/vt2 ...
                                v_idx_str, vt_idx_str = v.split('/')
                                try:
                                    v_idx = int(v_idx_str)
                                    vt_idx = int(vt_idx_str)
                                    face.append((v_idx, vt_idx))
                                except ValueError:
                                    print(f"Warning: Line {line_num} has invalid face indices: '{v}'")
                                    continue
                            else:
                                print(f"Warning: Line {line_num} has an unsupported face format: '{v}'")
                    if len(face) == 3:
                        faces.append(face)
                    elif len(face) < 3:
                        print(f"Warning: Line {line_num} has a face with fewer than 3 vertices: '{line}'")
                    else:
                        print(f"Warning: Reading polygonal faces, which may lead to undefined behavior.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return None
    return {
        'vertices': np.array(vertices),
        'normals': np.array(normals) if normals != [] else None,
        'uvs': np.array(uvs) if uvs != [] else None,
        'faces': np.array(faces) if faces != [] else None
    }

def normalize_shape(points):
    center = points.mean(dim=0, keepdim=True)
    points_centered = points - center

    max_extent = (points_centered.abs().max(dim=0)[0])  # (3,)
    scale = max_extent.max()
    points_normalized = points_centered / scale

    return points_normalized, center, scale