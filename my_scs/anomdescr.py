from my_scs.characteristics_names_map import characteristics_names_map


def anomdescr(anom):
    assert isinstance(anom, list) and len(anom) > 0, 'The first parameter must be a non-empty list!'

    # params
    n = len(anom)
    descriptions = [None] * n

    # map from names to descriptions
    desc = characteristics_names_map()  # 假设这个函数已经在其他地方定义

    # translate all names
    for i in range(len(anom)):
        a = anom[i]
        prefix = a[:3]
        s = a[3:]

        if prefix == 'rme':
            n = 'Market'
        elif prefix == 're_':
            n = read_desc(desc, s)
        elif prefix == 'r2_':
            n = f"{read_desc(desc, s)}$^2$"
        elif prefix == 'r3_':
            n = f"{read_desc(desc, s)}$^3$"
        elif prefix == 'rX_':
            xsep = '__'
            idx = s.find(xsep)
            if idx == -1:
                xsep = '_'  # OLD convention
                idx = s.find(xsep)
            n = f"{read_desc(desc, s[:idx])}$\\times${read_desc(desc, s[idx + len(xsep):])}"
        else:
            if prefix[:2] == 'r_':
                s = a[2:]
            else:
                s = a
            n = read_desc(desc, s)

        n = n.replace('_', '\\_')

        descriptions[i] = n

    return descriptions


def read_desc(map_dict, s):
    """
    读取描述从映射中，如果没有描述可用则不抛出错误
    """
    if s in map_dict:
        return map_dict[s]
    else:
        print(f"Warning: No description available for [{s}]")
        return s