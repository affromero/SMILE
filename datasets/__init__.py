MASK_LABELS = {
    'background': 0,
    'skin': 1,
    'nose': 2,
    'eye_g': 3,
    'l_eye': 4,
    'r_eye': 5,
    'l_brow': 6,
    'r_brow': 7,
    'l_ear': 8,
    'r_ear': 9,
    'mouth': 10,
    'u_lip': 11,
    'l_lip': 12,
    'hair': 13,
    'hat': 14,
    'ear_r': 15,
    'neck_l': 16,
    'neck': 17,
    'cloth': 18
}

MASK_ATTRS = {
    '5_o_Clock_Shadow': ['skin'],
    'Arched_Eyebrows': ['l_brow', 'r_brow'],
    'Attractive': ['skin'],
    'Bags_Under_Eyes': ['l_eye', 'r_eye'],
    'Bald': ['hair'],
    'Bangs': ['hair'],
    'NOT_Bangs': ['hair'],  # --
    'Big_Lips': ['u_lip', 'l_lip'],
    'Big_Nose': ['nose'],
    'Black_Hair': ['hair'],
    'Blond_Hair': ['hair'],
    'Blurry':
    list(MASK_LABELS.keys()),
    'Brown_Hair': ['hair'],
    'Bushy_Eyebrows': ['l_brow', 'r_brow'],
    'Chubby': ['skin'],
    'Double_Chin': ['skin'],
    # 'Eyeglasses': list(MASK_LABELS.keys()), # ['eye_g'],
    'Eyeglasses': ['eye_g'],
    'NOT_Eyeglasses': ['eye_g'],
    'General_Style': ['mouth', 'u_lip',
                      'l_lip'],  # list(MASK_LABELS.keys()),  # --
    'Goatee': ['skin'],
    'Gray_Hair': ['hair'],
    # 'Hair': list(MASK_LABELS.keys()), # --
    'Hair': ['hair'],  # --
    'NOT_Hair': ['hair'],  # --
    'Few_Hair': ['hair'],  # --
    'Much_Hair': ['hair'],  # --
    # 'NOT_Hair': list(MASK_LABELS.keys()), # --
    'Heavy_Makeup': ['skin'],
    'High_Cheekbones': ['skin'],
    # 'Male': ['skin'],
    # 'Male': list(MASK_LABELS.keys()),  # --
    # 'Female': list(MASK_LABELS.keys()),  # --
    'Short_Hair': ['hair'],  # --
    'Long_Hair': ['hair'],  # --
    'Male': [
        'skin', 'neck', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear',
        'r_ear', 'u_lip', 'l_lip', 'hair'
    ],  # --
    'Female': [
        'skin', 'neck', 'nose', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'l_ear',
        'r_ear', 'u_lip', 'l_lip', 'hair'
    ],  # --
    # 'Male': ['hair'],  # --
    # 'Female': ['hair'], # --
    'Mouth_Slightly_Open': ['mouth'],
    'Mustache': ['skin'],
    'Narrow_Eyes': ['l_eye', 'r_eye'],
    'No_Beard': ['skin'],
    'Oval_Face': ['skin'],
    'Pale_Skin': ['skin'],
    'Pointy_Nose': ['nose'],
    'Receding_Hairline': ['hair'],
    'Rosy_Cheeks': ['skin'],
    'Sideburns': ['hair'],
    'Happiness':
    ['mouth', 'u_lip', 'l_lip', 'l_eye', 'r_eye', 'l_brow', 'r_brow'],
    # 'Smiling': list(MASK_LABELS.keys()), # ['mouth', 'u_lip', 'l_lip'],
    # 'NOT_Smiling': list(MASK_LABELS.keys()), # ['mouth', 'u_lip', 'l_lip'],  # --
    'Smiling': ['mouth', 'u_lip', 'l_lip'],
    'NOT_Smiling': ['mouth', 'u_lip', 'l_lip'],  # --
    'Straight_Hair': ['hair'],
    'Wavy_Hair': ['hair'],
    'Wearing_Earrings': ['ear_r'],
    'Earrings': ['ear_r'],  # --
    'NOT_Earrings': ['ear_r'],  # --
    'Wearing_Hat': ['hat'],
    'Hat': ['hat'],  # --
    'NOT_Hat': ['hat'],  # --
    'Wearing_Lipstick': ['u_lip', 'l_lip'],
    'Wearing_Necklace': ['neck_l'],
    'Wearing_Necktie': ['neck'],
    'Young': ['skin'],
    'Aged': ['skin'],  # --
}
SEMANTIC_ATTRS = {'Background': ['background']}
SEMANTIC_ATTRS_MISSING = {
    'Background': ['background'],
    'Nose': ['nose'],
    'L_eye': ['l_eye'],
    'R_eye': ['r_eye'],
    'L_brow': ['l_brow'],
    'R_brow': ['r_brow'],
    'L_ear': ['l_ear'],
    'R_ear': ['r_ear'],
    'Hat': ['hat'],
    'Ear_r': ['ear_r'],
    'Neck_l': ['neck_l'],
    'Neck': ['neck'],
    'Cloth': ['cloth']
}
SEMANTIC_ATTRS_MISSING_WITH_HAIR = {
    'Background': ['background'],
    'Nose': ['nose'],
    'L_eye': ['l_eye'],
    'R_eye': ['r_eye'],
    'L_brow': ['l_brow'],
    'R_brow': ['r_brow'],
    'L_ear': ['l_ear'],
    'R_ear': ['r_ear'],
    'Hair': ['hair'],
    'Hat': ['hat'],
    'Ear_r': ['ear_r'],
    'Neck_l': ['neck_l'],
    'Neck': ['neck'],
    'Cloth': ['cloth'],
}
SEMANTIC_ATTRS_NO_EYEGLASSES = {
    # 'Background': ['background'],
    'Nose': ['nose'],
    'L_eye': ['l_eye'],
    'R_eye': ['r_eye'],
    'L_brow': ['l_brow'],
    'R_brow': ['r_brow'],
    'L_ear': ['l_ear'],
    'R_ear': ['r_ear'],
    'Mouth': ['mouth'],
    'U_lip': ['u_lip'],
    'L_lip': ['l_lip'],
    'Hair': ['hair'],
    'Hat': ['hat'],
    'Ear_r': ['ear_r'],
    'Neck_l': ['neck_l'],
    'Neck': ['neck'],
    'Cloth': ['cloth'],
}
SEMANTIC_ATTRS_ONLY = {i.capitalize(): [i] for i in MASK_LABELS.keys()}
