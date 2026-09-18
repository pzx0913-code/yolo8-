# -*- coding: utf-8 -*-
"""
智能汽车与车辆知识库模块
提供常见乘用车、商用车、非机动车及交通设施的详细规格、车型分类、动力构型与安全行车规程。
"""

VEHICLE_KNOWLEDGE_BASE = {
    "qiche": {
        "cn_name": "汽车 / 乘用车 (qiche)",
        "en_name": "Qiche / Automobile",
        "category": "乘用车 · M1 类小型客车 (轿车/SUV/新能源汽车)",
        "powertrain": "纯电 (EV) / 混动 (PHEV/HEV) / 燃油 (ICE)",
        "dimensions": "车长 4400~5100 mm · 整备质量 1.3~2.3 吨",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1/C2",
        "scenario": "城市日常通勤、城际高速公路出行，兼具操控性、舒适性与经济能耗",
        "tech_features": "承载式高刚性车身，配置 ABS 防抱死、ESP 电子稳定与 L2 级智能驾驶辅助系统",
        "safety_tips": "行车注意盲区监测，遇行人和大车主动保持安全车距，严禁超速与疲劳驾驶"
    },
    "car": {
        "cn_name": "汽车 / 轿车 (car)",
        "en_name": "Car / Sedan",
        "category": "乘用车 · M1 类小型客车 (紧凑型/中大型)",
        "powertrain": "纯电 (EV) / 插电混动 (PHEV) / 燃油 (ICE)",
        "dimensions": "车长 4500~5000 mm · 整备质量 1.3~2.1 吨",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1/C2",
        "scenario": "城市日常通勤、城际高速公路出行，侧重操控稳定性、乘坐舒适性与经济能耗",
        "tech_features": "承载式车身结构，低重心布局，风阻系数通常在 0.20~0.28 Cd，普遍配备 ABS、ESP 及 L2 级智能驾驶辅助系统",
        "safety_tips": "行车注意盲区监测，跟车保持充足车距；进入弯道前提前减速，高速公路行驶杜绝超速与疲劳驾驶"
    },
    "suv": {
        "cn_name": "运动型多用途汽车 (SUV)",
        "en_name": "Sport Utility Vehicle (SUV)",
        "category": "乘用车 · 城市/越野多功能乘用车",
        "powertrain": "纯电 / 增程式 / 插电混动 / 燃油 (常配四驱)",
        "dimensions": "车长 4600~5200 mm · 整备质量 1.7~2.6 吨",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1/C2",
        "scenario": "全路况出行、家庭长途自驾游及轻度非铺装路面通行，具备开阔视野与高装载拓展能力",
        "tech_features": "底盘离地间隙较高 (通常大于 180mm)，接近角与离去角优良；常配备多模式适时四驱系统与陡坡缓降控制",
        "safety_tips": "车身重心相对轿车偏高，高速变道或紧急避险时避免剧烈打方向以防侧倾；特别关注侧后方大盲区"
    },
    "bus": {
        "cn_name": "大型客车 / 城市公交车",
        "en_name": "Bus / Public Transit Coach",
        "category": "商用客车 · M2/M3 类大型载客汽车",
        "powertrain": "纯电驱动 (城市公交为主) / 高压共轨柴油机 (长途客车)",
        "dimensions": "车长 8000~12000 mm · 核载人数 30~90 人",
        "license_plate": "大型汽车号牌 (黄牌/大型新能源绿牌) · 准驾 A1/A3",
        "scenario": "城市干线公共交通运营、城际旅客运输、大型园区通勤与团体旅游接待",
        "tech_features": "低地板/低入口无障碍设计，采用气压制动与电涡流缓速器，集成主动防碰撞预警与客流统计车载终端",
        "safety_tips": "车身庞大存在显著内轮差与后视镜盲区；非机动车与行人在转弯路口切勿抢行贴近，跟车保持至少 50 米距离"
    },
    "truck": {
        "cn_name": "载货汽车 / 重型卡车",
        "en_name": "Truck / Heavy Freight Carrier",
        "category": "商用货运车 · N2/N3 类载货车辆 / 半挂牵引车",
        "powertrain": "大扭矩重型柴油机 / LNG 液化天然气 / 换电重卡",
        "dimensions": "车长 6000~16000 mm · 总质量 18~49 吨",
        "license_plate": "大型货车号牌 (黄牌) · 准驾车型 B2/A2",
        "scenario": "跨省干线物流货运、集装箱港口周转、矿山砂石骨料运输与工程基建配套",
        "tech_features": "高强度工字形双层大梁车架，并装双联驱动桥，气刹双回路制动与发动机制动，标配北斗双模定位终端",
        "safety_tips": "满载惯性极大且刹车距离长，严禁超载超速；驾驶室右前方与车尾存在视线死角，周围车辆切忌长时间并排或加塞"
    },
    "motorcycle": {
        "cn_name": "摩托车 / 机动二轮车",
        "en_name": "Motorcycle / Two-Wheeler",
        "category": "摩托车 · L3 类两轮机动车辆",
        "powertrain": "单缸/双缸四冲程发动机 (125cc~1000cc) / 高性能电摩",
        "dimensions": "整备质量 110~280 kg · 车长约 1900~2200 mm",
        "license_plate": "摩托车号牌 (黄牌/轻便蓝牌) · 准驾车型 D/E",
        "scenario": "城市轻便快捷通勤、交通警务巡逻执勤、即时配送与公路骑行运动",
        "tech_features": "高功率推重比，车身轻盈灵活；主流车型配置前后双通道 ABS 防抱死及 TCS 牵引力控制系统",
        "safety_tips": "骑行必须按国标规范佩戴安全头盔与护具；严禁在机动车流中穿插抢道，雨天经过道路标线易打滑须平缓减速"
    },
    "van": {
        "cn_name": "轻型厢式客车 / 面包车",
        "en_name": "Light Commercial Van",
        "category": "轻型多用途客货车 · M1/N1 类",
        "powertrain": "高性价比汽油机 / 纯电电驱动",
        "dimensions": "车长 4400~5400 mm · 核载 5~9 人或轻载物资",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1",
        "scenario": "城市短途物流快运、社区商业配送、客货兼顾出行及特种车辆改装",
        "tech_features": "高顶平头方正车身，内部空间容积率极高，配置侧滑移门与对开尾门，底盘调校偏重承载耐久度",
        "safety_tips": "车辆空载与满载时操控差异明显，载货时严禁超载超高；过弯时适当减速防侧滑，严禁人货混装违规载客"
    },
    "mpv": {
        "cn_name": "多用途商务乘用车 (MPV)",
        "en_name": "Multi-Purpose Vehicle (MPV)",
        "category": "乘用车 · 商务接待与家庭旗舰乘用车",
        "powertrain": "插电混动 (PHEV) / 纯电 / 油电混动 (HEV)",
        "dimensions": "车长 4900~5400 mm · 轴距通常大于 3000 mm",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1/C2",
        "scenario": "中高端政企商务接待、机场高铁接驳及多孩多代家庭长途舒适出行",
        "tech_features": "双侧电动侧滑门，2+2+3 座椅布局，第二排配备独立零重力航空座椅，静音工程与底盘滤震性能优异",
        "safety_tips": "第三排座椅距车尾相对较近，行车务必监督后排乘员全程规范系好安全带；车体较长入库建议结合全景环视"
    },
    "sports car": {
        "cn_name": "跑车 / 高性能轿跑",
        "en_name": "Sports Car / Performance Coupe",
        "category": "乘用车 · 高性能双门/四门轿跑",
        "powertrain": "高功率涡轮增压发动机 / 多电机大功率电驱系统",
        "dimensions": "车高普遍低于 1360 mm · 整备质量 1.4~1.9 吨",
        "license_plate": "小型汽车号牌 (蓝牌/绿牌) · 准驾车型 C1/C2",
        "scenario": "赛道封闭场地竞速、高性能驾驶体验与个性化高端出行",
        "tech_features": "极低质心与宽轮距设计，50:50 前后配重比，主动可变空气动力学尾翼，多活塞高性能制动卡钳与通风刹车盘",
        "safety_tips": "动力输出强劲激进，公路行驶须严格遵守限速法规；底盘离地间隙极低，通过减速带或地库陡坡时须低速慢行"
    },
    "pickup": {
        "cn_name": "多用途载货汽车 (皮卡)",
        "en_name": "Pickup Truck",
        "category": "轻型载货汽车 · 客货两用多功能车",
        "powertrain": "高扭矩柴油机 / 涡轮增压汽油机 / 混动四驱",
        "dimensions": "车长 5300~5700 mm · 具备开放式尾箱",
        "license_plate": "轻型货车号牌 (蓝牌/绿牌) · 准驾车型 C1",
        "scenario": "野外工程勘探、农林牧业巡检、户外露营装备运输及专业房车拖挂",
        "tech_features": "非承载式带大梁车架，配备分时四驱系统与后桥差速锁，具备突出的拖拽牵引能力与恶劣路况通过性",
        "safety_tips": "后货箱装载物必须捆绑遮盖扎牢，严禁后箱违规载人；行驶时注意各城市不同区域对皮卡车型的通行管理规定"
    },
    "bicycle": {
        "cn_name": "自行车 / 非机动两轮车",
        "en_name": "Bicycle / Cycle",
        "category": "非机动车 · 绿色微循环代步工具",
        "powertrain": "人力踩踏链条传动 / 智能助力 (E-Bike)",
        "dimensions": "整备质量 10~25 kg · 车长约 1600~1800 mm",
        "license_plate": "非机动车牌照 / 免牌照登记",
        "scenario": "最后一公里接驳、城市慢行交通、绿色骑行健身与休闲代步",
        "tech_features": "轻量化铝合金/碳纤维车架，机械变速套件与线拉/液压碟刹制动装置",
        "safety_tips": "按规定在非机动车道内行驶，严禁逆行或闯红灯；夜间骑行建议配置主动车灯与反光背心，进出路口注意减速"
    },
    "traffic light": {
        "cn_name": "交通信号灯 (路口管制)",
        "en_name": "Traffic Signal Control",
        "category": "道路交通控制与安全管理设施",
        "powertrain": "市电供给 / 太阳能储能式智慧路口控制终端",
        "dimensions": "标准三色灯组 (红、黄、绿)",
        "license_plate": "城市交通基础设施",
        "scenario": "平面交叉十字路口、人行横道斑马线及专用车道动态分流管控",
        "tech_features": "超高亮度 LED 面阵光源，结合微波雷达与地磁线圈实现车流量自适应绿波控制",
        "safety_tips": "严格遵循红灯停、绿灯行、黄灯警示原则；路口转弯车辆应主动礼让直行车辆及行人"
    },
    "stop sign": {
        "cn_name": "停步让行交通标志 (Stop Sign)",
        "en_name": "Stop Traffic Warning Sign",
        "category": "道路交通禁令标志",
        "powertrain": "高强级无源逆反射材料",
        "dimensions": "正八边形红色标志牌",
        "license_plate": "道路交通基础设施",
        "scenario": "支路汇入主干道交叉口、无灯控次要道路交汇处及铁路平交道口",
        "tech_features": "采用高强度微棱镜反光膜，夜间在车灯照射下具备极佳的可视距离与警示反光率",
        "safety_tips": "车辆驶至该标志前必须在停车线前完全刹停，仔细瞭望观察确认安全后方可起步通行"
    },
    "person": {
        "cn_name": "行人 / 交通参与人员 (Pedestrian)",
        "en_name": "Pedestrian / Person",
        "category": "弱势道路交通参与者 (VRU)",
        "powertrain": "人体生理自主动力",
        "dimensions": "成人 / 儿童各体态尺度",
        "license_plate": "无登记牌照",
        "scenario": "城市道路人行横道、非机动车道、路口斑马线及居民生活区",
        "tech_features": "机动性强但防护脆弱，运动轨迹多变且夜间对比度低，属于 ADAS 与自动驾驶主动刹车 (AEB) 重点防护对象",
        "safety_tips": "车辆行经斑马线必须提前减速避让，遇行人过街须停车礼让；转弯时警惕车头 A 柱盲区与内轮差"
    }
}

DEFAULT_VEHICLE_INFO = {
    "cn_name": "机动车辆目标",
    "en_name": "Motor Vehicle",
    "category": "道路交通参与车辆",
    "powertrain": "内燃机 (ICE) / 混合动力 (HEV/PHEV) / 纯电驱动 (EV)",
    "dimensions": "根据实际车型规制测定",
    "license_plate": "按国家机动车登记规定核发号牌",
    "scenario": "城市道路、高架路及公路交通运输通行",
    "tech_features": "配备符合国家机动车安全技术标准的制动、转向、信号与被动碰撞安全防护系统",
    "safety_tips": "遵守道路交通安全法规，严禁超速行驶、疲劳驾驶，与前方车辆保持规定安全制动间距"
}


def get_vehicle_wiki(class_name: str) -> dict:
    """
    根据识别到的类别英文或中文名称，检索车辆知识库元数据
    """
    key = str(class_name).lower().strip()
    
    # 1. 精确匹配
    if key in VEHICLE_KNOWLEDGE_BASE:
        return VEHICLE_KNOWLEDGE_BASE[key]
        
    # 2. 模糊包含匹配
    for k, info in VEHICLE_KNOWLEDGE_BASE.items():
        if k in key or key in k:
            return info
            
    # 3. 常见同义词匹配
    alias_map = {
        "qiche": "qiche",
        "car": "qiche",
        "cars": "qiche",
        "automobile": "qiche",
        "automobiles": "qiche",
        "sedan": "car",
        "passenger car": "qiche",
        "suvs": "suv",
        "motorbikes": "motorcycle",
        "motorbike": "motorcycle",
        "scooter": "motorcycle",
        "buses": "bus",
        "coach": "bus",
        "trucks": "truck",
        "lorry": "truck",
        "semi-truck": "truck",
        "trailer": "truck",
        "bike": "bicycle",
        "minivan": "van",
        "commercial van": "van"
    }
    if key in alias_map:
        return VEHICLE_KNOWLEDGE_BASE[alias_map[key]]

    # 4. 默认安全兜底
    info = DEFAULT_VEHICLE_INFO.copy()
    info["cn_name"] = f"{class_name} (车辆目标)"
    info["en_name"] = class_name
    return info


# 别名导出
get_qiche_wiki = get_vehicle_wiki
get_plant_wiki = get_vehicle_wiki
