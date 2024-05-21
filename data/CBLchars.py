CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '港', '澳', '挂', '学', '领', '使', '临',

    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'I', 'O' ]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
LP_CLASS=['黑色车牌', '单层黄牌', '双层黄牌', '普通蓝牌', '拖拉机绿牌', '新能源大型车', '新能源小型车']
LP_CLASS_DICT={LP_C:i for i, LP_C in enumerate(LP_CLASS)}