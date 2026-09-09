# -*- coding: utf-8 -*-
"""
植物小百科与知识库模块
提供常见植物、花卉、农作物及室内盆栽的详细养护信息、生物学特性与百科介绍。
"""

PLANT_KNOWLEDGE_BASE = {
    "potted plant": {
        "cn_name": "室内观叶盆栽 (通用)",
        "scientific_name": "Potted Plant",
        "category": "观赏绿植",
        "sunlight": "明亮散射光，避免强光暴晒",
        "watering": "见干见湿，盆土干透后再浇透水",
        "description": "常见的室内观赏植物，具备净化空气、调节室内湿度与美化环境的功效。",
        "tips": "注意定期擦拭叶片灰尘，保持室内通风良好，冬季注意防寒保暖。"
    },
    "monstera": {
        "cn_name": "龟背竹",
        "scientific_name": "Monstera deliciosa",
        "category": "天南星科龟背竹属",
        "sunlight": "喜半阴，喜散射光，忌强光直射",
        "watering": "保持盆土湿润但忌积水，生长季多喷水保持湿度",
        "description": "大型常绿藤本植物，叶形奇特呈龟甲孔裂状，常用于现代家居装饰，能有效吸附室内甲醛。",
        "tips": "温度低于10℃时需移入温暖室内，春夏生长旺季可每月施一次稀薄复合肥。"
    },
    "succulent": {
        "cn_name": "多肉植物",
        "scientific_name": "Succulent",
        "category": "景天科/番杏科等",
        "sunlight": "喜充足日照（夏季需适当遮阳通风）",
        "watering": "宁干勿湿，一个月浇水 1~2 次即可",
        "description": "具有肥厚肉质茎叶的多浆植物，形态憨态可掬，耐旱能力极强。",
        "tips": "千万避免频繁浇水导致烂根黑腐，土壤建议颗粒土占比 50% 以上。"
    },
    "pothos": {
        "cn_name": "绿萝",
        "scientific_name": "Epipremnum aureum",
        "category": "天南星科麒麟叶属",
        "sunlight": "极耐阴，明亮散射光环境生长最茂盛",
        "watering": "喜湿润，见表土微干即可浇水，亦可水培",
        "description": "最具人气的室内观叶植物，适应能力极强，生命力顽强，能有效吸收甲醛与苯。",
        "tips": "冬季控水防冻，温度不低于8℃；避免长期处于极度昏暗环境中。"
    },
    "rose": {
        "cn_name": "玫瑰 / 月季",
        "scientific_name": "Rosa hybrida",
        "category": "蔷薇科蔷薇属",
        "sunlight": "喜强光直射，每天至少需 4~6 小时直射日照",
        "watering": "见干见湿，浇水时尽量避免淋在花朵和叶片上",
        "description": "花中皇后，品种繁多，花色艳丽，香气宜人，广泛用于切花与花园绿化。",
        "tips": "注意通风预防白粉病和红蜘蛛，花谢后及时剪去残花以促进新枝萌发。"
    },
    "sunflower": {
        "cn_name": "向日葵",
        "scientific_name": "Helianthus annuus",
        "category": "菊科向日葵属",
        "sunlight": "全日照植物，极喜充足阳光",
        "watering": "苗期适量，开花期需水量较大，需充足水分支持",
        "description": "一年生高大草本，花序随太阳转动，象征阳光、活力与希望。",
        "tips": "需疏松肥沃的砂质土壤，开花期增施磷钾肥有利于花盘增大与种子饱满。"
    },
    "tulip": {
        "cn_name": "郁金香",
        "scientific_name": "Tulipa gesneriana",
        "category": "百合科郁金香属",
        "sunlight": "喜阳光充足但凉爽的环境",
        "watering": "生长期保持湿润，开花后减少浇水",
        "description": "多年生球根花卉，荷兰国花，花形端庄优美，色彩纯正艳丽。",
        "tips": "夏季休眠期需起球冷藏干燥保存，秋冬再行栽种以完成自然春化。"
    },
    "daisy": {
        "cn_name": "雏菊",
        "scientific_name": "Bellis perennis",
        "category": "菊科雏菊属",
        "sunlight": "喜光照，光照充足时株型紧凑、花多色艳",
        "watering": "保持湿润但不积水",
        "description": "早春著名花卉，花小玲珑，活泼娇小，耐寒耐瘠薄。",
        "tips": "喜冷凉气候，怕炎热，夏季注意遮阳降温。"
    },
    "dandelion": {
        "cn_name": "蒲公英",
        "scientific_name": "Taraxacum officinale",
        "category": "菊科蒲公英属",
        "sunlight": "喜光，耐半阴",
        "watering": "耐干旱，自然降雨多能满足生长需求",
        "description": "多年生草本植物，种子带白色冠毛绒球，成熟时随风飘散，全草亦可入药。",
        "tips": "生命力极其顽强，对土壤要求不严。"
    }
}

DEFAULT_PLANT_INFO = {
    "cn_name": "植物样本",
    "scientific_name": "Plantae",
    "category": "植物界",
    "sunlight": "根据具体品种提供适宜光照（大部分植物喜明亮散射光）",
    "watering": "遵循'见干见湿'原则，避免花盆底部长期积水",
    "description": "检测到的植物目标。植物通过光合作用制造养分，释放氧气，美化自然与生活环境。",
    "tips": "建议结合叶片形态、花果特征进一步鉴定具体科属品种。"
}


def get_plant_wiki(class_name: str) -> dict:
    """根据类别英文名称获取植物百科卡片信息"""
    key = class_name.lower().strip()
    if key in PLANT_KNOWLEDGE_BASE:
        return PLANT_KNOWLEDGE_BASE[key]
    
    for k, info in PLANT_KNOWLEDGE_BASE.items():
        if k in key or key in k:
            return info
            
    info = DEFAULT_PLANT_INFO.copy()
    info["cn_name"] = f"{class_name} (检测对象)"
    return info
