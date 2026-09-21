# -*- coding: utf-8 -*-
"""
智能汽车与车辆知识库模块 (v2.0 智能语义解析与多级特征合成引擎)
支持 COCO 基础类别、Stanford Cars 196 类超细分品牌车型，以及主流现代中国车型的全离线毫秒级富特征解析。
"""

import re

# 1. 基础标准交通目标与大类通用知识库 (COCO 兼容)
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
        "tech_features": "承载式车身结构，低重心布局，风阻系数通常在 0.20~0.28 Cd，普遍配备 ABS、ESP 及 L2 级智能辅助系统",
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

# 2. 国际与国内主流汽车品牌多维特征映射字典 (Brand Knowledge Graph)
BRAND_KNOWLEDGE = {
    "AM General": ("AM通用 / 悍马原厂", "美国", "重型军民两用全地形高机动车制造厂", "军用全时四驱、超高离地间隙门式车桥与军工级高强度防护车架"),
    "Acura": ("讴歌", "日本 / 本田技研", "日系前瞻豪华与操控科技品牌", "SH-AWD 超级四轮驱动力自由控制系统与高转速 V6 引擎"),
    "Aston Martin": ("阿斯顿·马丁", "英国", "英伦百年顶级超豪华GT跑车品牌", "手工打造大排量双涡轮增压引擎、全铝空间架构与优雅英伦雪茄车身"),
    "Audi": ("奥迪", "德国 / 大众汽车集团", "德系豪华三强之一 · 科技与性能标杆", "quattro 机械全时四驱系统、轻量化铝合金车身及智能激光大灯"),
    "BMW": ("宝马", "德国", "德系豪华运动与极致驾驶乐趣代表", "50:50 前后轴黄金配重比、精准后轮驱动底盘与 TwinPower Turbo 发动机"),
    "Bentley": ("宾利", "英国 / 大众汽车集团", "世界顶级奢华与手工定制超豪华座驾", "W12 / V8 大马力强劲动力、奢华手工真皮实木座舱与空气自适应悬架"),
    "Bugatti": ("布加迪", "法国 / 大众汽车集团", "世界极限极速量产超级跑车代名词", "8.0L W16 四涡轮增压引擎、极速超 400km/h 与主动升降减速空气尾翼"),
    "Buick": ("别克", "美国 / 通用汽车", "美系舒适品质与商务行政乘用车", "QuietTuning 专利图书馆级静音座舱与厚重舒适滤震底盘"),
    "Cadillac": ("凯迪拉克", "美国 / 通用汽车", "新美式豪华风范与锐利钻石切割美学", "MRC 电磁主动感应悬挂系统、后驱运动构型与超高扭矩输出"),
    "Chevrolet": ("雪佛兰", "美国 / 通用汽车", "全品类美系国民与传奇肌肉车品牌", "经典大排量 Small Block 自然吸气/机械增压引擎与高耐磨承载车架"),
    "Chrysler": ("克莱斯勒", "美国 / 斯特兰蒂斯", "经典美式豪华轿车与多功能商务车", "HEMI 大排量引擎动力总成与宽绰静谧乘坐空间"),
    "Daewoo": ("大宇", "韩国", "韩国实用经济型乘用代步车辆", "高实用性承载式车身与低用车成本日常调校"),
    "Dodge": ("道奇", "美国 / 斯特兰蒂斯", "美系硬派性能车与美式肌肉跑车图腾", "HEMI 机械增压 V8 引擎、狂暴声浪与全尺寸肌肉宽体造型"),
    "Eagle": ("鹰牌", "美国 / 克莱斯勒", "美系经典运动紧凑乘用车", "偏重运动操控的多连杆悬挂与动感轿跑线条"),
    "FIAT": ("菲亚特", "意大利 / 斯特兰蒂斯", "意式浪漫与经典时尚微型车代表", "小巧轻快车身比例、经典复古设计风格与高燃效小排量引擎"),
    "Ferrari": ("法拉利", "意大利", "世界顶级跃马传奇超级跑车", "F1 赛车工程反哺、高转速中置引擎、碳纤维单体壳与 E-Diff 电子限滑差速器"),
    "Fisker": ("菲斯克", "美国", "高端豪华增程纯电轿跑早期开拓者", "低矮宽体太阳能车顶轿跑轮廓与铝合金轻量空间架构"),
    "Ford": ("福特", "美国", "百年汽车工业先驱 · 扎实硬派与皮卡霸主", "EcoBoost 双涡流涡轮增压、高强度非承载式大梁车架与全地形四驱"),
    "GMC": ("通用GMC", "美国 / 通用汽车", "专业级高端硬派皮卡与全尺寸豪华SUV", "高规格拖拽牵引套件、超大排量 V8 引擎与奢华 Denali 旗舰装潢"),
    "Geo": ("吉奥", "美国 / 通用汽车", "美日合资经济型轻巧代步车", "轻盈整备质量、低阻力行驶特性与极低综合油耗"),
    "HUMMER": ("悍马", "美国 / 通用汽车", "重型硬派越野车与全地形霸主图腾", "三把机械差速锁、超强涉水喉深度与接近直角的极限攀爬通过角"),
    "Honda": ("本田", "日本", "发动机专家 · 极致可靠与空间魔术师", "i-VTEC 高转速气门升程技术、MM 人机工程空间最大化理念与轻快驾控"),
    "Hyundai": ("现代", "韩国", "全球前列的大型现代化汽车制造商", "流体雕塑设计美学、自研高效动力总成与丰富前瞻智能辅助系统"),
    "Infiniti": ("英菲尼迪", "日本 / 日产汽车", "日系豪华运动与东方力量美学座驾", "经典 VQ 系列自然吸气自吸之王发动机与全速域线控转向系统"),
    "Isuzu": ("五十铃", "日本", "柴油动力专家与专业级越野皮卡先锋", "高寿命高负荷电控共轨柴油机与纯正非承载式底盘车架"),
    "Jaguar": ("捷豹", "英国 / 塔塔汽车", "英伦动感豪华与优雅轿跑品牌", "全铝合金空间架构车身、前后双叉臂轻量化运动底盘与豹力美学"),
    "Jeep": ("吉普", "美国 / 斯特兰蒂斯", "专业硬派越野与全路况 SUV 鼻祖", "Selec-Terrain 全地形适应自选系统、前后轴电控差速锁与可断开防倾杆"),
    "Lamborghini": ("兰博基尼", "意大利 / 奥迪集团", "蛮牛图腾 · 激进极速前卫超级跑车", "高转速 V10/V12 自然吸气中置引擎、极具科幻感的折线切角与全时四驱"),
    "Land Rover": ("路虎", "英国 / 塔塔汽车", "全地形豪华奢享 SUV 标杆", "全地形反馈适应系统 (Terrain Response) 与自适应全地形动态空气悬架"),
    "Lincoln": ("林肯", "美国 / 福特汽车", "美系百年总统级豪华座驾品牌", "静谧私享座舱空间、自适应悬架与尊崇仪式感迎宾光毯"),
    "MINI": ("迷你", "英国 / 宝马集团", "兼具卡丁车操控乐趣的个性潮流座驾", "四轮四角短悬长轴布局、韧性十足的高刚性底盘与标志性圆形美学"),
    "Maybach": ("迈巴赫", "德国 / 奔驰集团", "世界顶峰超豪华旗舰行政礼宾座驾", "超长轴距头等舱后排、双色车身手工工艺与魔毯主动空气悬挂系统"),
    "Mazda": ("马自达", "日本", "东瀛宝马 · 执着人马一体纯粹驾控", "创驰蓝天高压缩比发动机、GVC 加速度矢量控制与魂动流动曲面设计"),
    "McLaren": ("迈凯伦", "英国", "纯正一级方程式 F1 赛车工程超级跑车", "全碳纤维单体壳座舱 (MonoCell)、双涡轮增压 V8 与主动液压悬挂阻尼"),
    "Mercedes-Benz": ("梅赛德斯-奔驰", "德国", "汽车发明者 · 豪华舒适与安全工业灯塔", "后驱/4MATIC 全时四驱、PRE-SAFE 预防性安全系统与数字化豪华智慧座舱"),
    "Mitsubishi": ("三菱", "日本", "硬派越野与达喀尔拉力拉力赛传奇", "超选四驱系统 (Super Select 4WD) 与高耐磨全气候底盘架构"),
    "Nissan": ("日产", "日本", "技术日产 · 舒适移动大沙发标杆", "Zero Gravity 零重力座椅、CVT 平顺变速系统与高耐用度动力平台"),
    "Plymouth": ("普利茅斯", "美国 / 克莱斯勒", "美系经典宽体巡航与肌肉跑车", "大马力后轮驱动布局、狂野复古美式车体比例"),
    "Porsche": ("保时捷", "德国 / 大众汽车集团", "世界跑车殿堂标杆 · 兼顾赛道与日常", "经典水平对置引擎 (Boxer)、PDK 双离合保时捷变速箱与极致下压力空气动力学"),
    "Ram": ("道奇公羊", "美国 / 斯特兰蒂斯", "美系全尺寸重型豪华多功能皮卡", "后五连杆/四轮空气悬架减震、高强度重载货箱与强悍牵引力"),
    "Rolls-Royce": ("劳斯莱斯", "英国 / 宝马集团", "车中帝王 · 世界超豪华汽车天花板", "帕特农神庙镀铬进气格栅、自杀式对开对坐车门、星空顶与静谧悬浮轮毂车标"),
    "Scion": ("塞恩", "日本 / 丰田汽车", "北美年轻个性化潮流改装座驾", "轻快车重、极高后市场改装宽容度与紧凑前驱/后驱动力"),
    "Spyker": ("世爵", "荷兰", "航空基因传承的百年手工定制超跑", "全手工打造航空铝合金轻量车身、外露式换挡连杆与飞机旋钮座舱"),
    "Suzuki": ("铃木", "日本", "小车之王 · 紧凑精悍与专业四驱专家", "高强度轻量化车体平台、实用四轮驱动系统与极端恶劣路况通过性"),
    "Tesla": ("特斯拉", "美国", "全球智能纯电动汽车革新开创者", "前后双电机超大扭矩电驱、底盘平铺动力电池包与自动辅助驾驶软硬件平台"),
    "Toyota": ("丰田", "日本", "车到山前必有路 · 经久耐用可靠性标杆", "THS 智能电混双擎架构、TNGA 低质心高刚性平台与高保值率品质"),
    "Volkswagen": ("大众", "德国", "全球国民与严谨德系制造标杆", "EA888 / EA211 系列涡轮增压发动机与 DSG 湿式双离合黄金高效动力链"),
    "Volvo": ("沃尔沃", "瑞典 / 吉利汽车", "全球汽车主被动安全标准的制定者", "笼式高强度超高强度硼钢车身、City Safety 城市安全主动刹车系统与健康座舱"),
    "smart": ("精灵smart", "德国 / 奔驰与吉利", "都市潮流轻便出行与微循环精灵", "超小转弯半径、独特 Tridion 高强度安全车架与极致停车便利度"),
    # 国内新势力与自主品牌前瞻覆盖 (前向兼容)
    "BYD": ("比亚迪", "中国", "全球新能源汽车领军自主品牌", "刀片电池高安全结构、DM-i 超级混动系统与 e平台3.0 纯电架构"),
    "Geely": ("吉利", "中国", "自主科技智造与全球化汽车工业集团", "CMA 架构高刚性车身、雷神混动与高性能智能电驱系统"),
    "NIO": ("蔚来", "中国", "高端智能纯电与全场景补能生态领军", "全栈自研高性能智能底盘、换电补能体系与智能数字座舱"),
    "Li Auto": ("理想", "中国", "家庭旗舰智能电动 SUV 标杆", "双电机增程四驱无焦虑长续航、魔毯空气悬挂与多屏家庭娱乐空间"),
    "XPeng": ("小鹏", "中国", "高阶智能辅助驾驶与科技创新先锋", "全场景智能导航辅助驾驶、800V 高压碳化硅超充平台与轻量化车体"),
    "Xiaomi": ("小米", "中国", "人车家全生态高性能智能科技轿跑", "超高转速电机动力平台、CTB 一体化电池技术与澎湃 OS 互联协同"),
    "Zeekr": ("极氪", "中国", "高端智能纯电猎装与操控先锋", "浩瀚 SEA 纯电浩瀚架构、超跑级麋鹿测试表现与百万级空气悬架"),
    "Haval": ("哈弗", "中国 / 长城汽车", "国民 SUV 与硬派全地形普及者", "高刚性安全车架结构、智能电控适时四驱与硬朗通过底盘"),
    "Hongqi": ("红旗", "中国 / 一汽集团", "国车风范 · 新高尚中式豪华座驾", "尚致意东方美学、平顺高扭矩发动机与国宾级极致静谧调校"),
    "Wuling": ("五菱", "中国 / 上汽通用五菱", "人民需要什么就造什么的国民品牌", "极致空间利用率、超皮实耐用传动构件与超低维护使用成本"),
    "Chery": ("奇瑞", "中国", "工程师自研技术与全球出口标杆", "鲲鹏动力自研发动机、火星架构平台与高强度轻量吸能车架"),
    "Changan": ("长安", "中国", "自主创新龙头与高品质智慧乘用车", "蓝鲸动力高压直喷涡轮增压系统与方舟架构智能安全车身")
}

# 3. 车身形态规格库 (Body Type Knowledge Base)
BODY_TYPE_KNOWLEDGE = {
    "Coupe": {
        "name": "双门硬顶跑车 / 轿跑",
        "desc": "双门流线型空气动力学设计，侧重低重心、高刚性与赛道操控体验",
        "dimensions": "车长 4300~4750 mm · 车身高度通常低于 1380 mm · 整备质量 1.3~1.7 吨",
        "scenario": "赛道竞速刷圈、封闭场地性能体验、高端跑车巡航与个性化出行",
        "safety_tips": "底盘离地间隙较低，减速带及地下车库入口须慢行；进入弯道平稳控制油门，避免大马力后驱雨天打滑"
    },
    "Convertible": {
        "name": "敞篷跑车 / 敞篷轿跑",
        "desc": "配备电动机械折叠硬顶或织物软顶机构，具备开扬浪漫的座舱体验",
        "dimensions": "车长 4200~4800 mm · 标配主动防滚架与底盘抗扭加固大梁",
        "scenario": "沿海公路巡航、风景区自驾游览、阳光休闲出行与个性风尚展示",
        "safety_tips": "行车中开合敞篷须严格遵照安全限速要求；露天停放及时关闭顶棚防风雨，注意车内物品防盗"
    },
    "Sedan": {
        "name": "三厢轿车 / 行政座驾",
        "desc": "发动机舱、乘员舱与行李舱独立分立的经典标准四门五座构型",
        "dimensions": "车长 4600~5100 mm · 轴距 2700~3050 mm · 车重 1.4~2.1 吨",
        "scenario": "城市日常通勤、城际高速公路出行、商务政务接待与家庭品质出行",
        "safety_tips": "日常行驶注意盲区变道预警，跟车保持充足安全距离，长途高速行驶严禁疲劳驾驶"
    },
    "SUV": {
        "name": "运动型多用途汽车 (SUV)",
        "desc": "高离地间隙、高坐姿开阔视野，兼顾全路况通过性与大容积储物空间",
        "dimensions": "车身高度通常大于 1700 mm · 最小离地间隙 > 190 mm · 具备优良接近/离去角",
        "scenario": "全气候出行、家庭长途自驾探险、轻度非铺装土路越野与复杂雨雪路况",
        "safety_tips": "车身重心偏高，高速紧急变道或过弯时避免猛打方向以防侧倾；特别留心车头低洼盲区"
    },
    "Hatchback": {
        "name": "两厢掀背车 / 紧凑小钢炮",
        "desc": "乘员舱与后行李舱贯通，大角度一体式后尾门向上掀起，短小精悍",
        "dimensions": "车长 4000~4400 mm · 轴距紧凑灵活 · 便于狭窄街巷通行穿梭与泊车",
        "scenario": "都市青年通勤、高密度拥堵市区通勤穿梭、立体车库快捷泊车",
        "safety_tips": "车身短小灵活但在大车并排行驶时易处于货车盲区，避免长时间与大车并排行驶"
    },
    "Wagon": {
        "name": "旅行车 (Wagon / 瓦罐)",
        "desc": "保留轿车极佳的低重心操控舒适度，同时尾部延展出超长纵深大后备箱",
        "dimensions": "车长 4700~5000 mm · 超大容积行李舱且后排支持全平放倒",
        "scenario": "欧洲风尚公路巡航、户外长途露营装备装载、全家周末远足旅行",
        "safety_tips": "后悬较长倒车入库时注意尾部障碍物距离；后备箱装载重物应尽量贴近靠背并扎紧"
    },
    "Minivan": {
        "name": "多用途厢式车 (MPV / Minivan)",
        "desc": "单厢式高车顶设计，配置双侧侧滑移门，注重多乘员空间舒适度与乘坐平顺性",
        "dimensions": "车长 4800~5400 mm · 内部垂直挑高优越 · 采用 2+2+3 灵活座椅布局",
        "scenario": "多孩多代家庭长途旅行、机场贵宾接待接驳、移动商务会议办公",
        "safety_tips": "车身宽大在狭窄路口转弯时预留足够内轮差空间；后排乘员须全程规范系好安全带"
    },
    "Van": {
        "name": "厢式客货车 / 轻型物流车",
        "desc": "平头/短头厢式结构，侧重内部大容积率与客货运输耐用度",
        "dimensions": "车长 4600~5500 mm · 强化钢板弹簧后桥 · 承载耐久度优越",
        "scenario": "城市短途即时配送、商超百货周转、客货兼用运输与专业特种车改装",
        "safety_tips": "空载与满载制动刹车距离差异明显，载货严禁超重超高；车内人货分离严禁违规载人"
    },
    "Cab": {
        "name": "多用途载货汽车 (皮卡 / Pickup)",
        "desc": "前面为载客驾驶室，后部为开放式大载重货斗，兼顾越野与牵引重载能力",
        "dimensions": "车长 5300~5900 mm · 非承载式带大梁底盘 · 标配大扭矩四驱",
        "scenario": "工程抢险、农林巡检、户外重型越野穿越、房车游艇拖挂与大件物资运输",
        "safety_tips": "后货斗货物必须遮盖雨布并扎紧，严禁货斗载人违章行驶；留意各城市关于货车的禁行路段规定"
    }
}


def parse_vehicle_class_name(class_name: str) -> dict:
    """
    智能解析规整车型名称字符串 (如 'Ferrari 458 Italia Coupe 2012', 'BMW_3_Series_2012', 'Toyota-Camry-2012')
    返回解构字典: brand, brand_cn, model, body_type, body_type_cn, year
    """
    raw = str(class_name).strip()
    if not raw:
        return {"brand": "", "brand_cn": "", "model": "", "body_type": "", "body_type_cn": "", "year": ""}

    # 0. 规范化清洗：将下划线、短横线统一转换为标准空格，合并多余连续空白
    clean_str = re.sub(r"[-_]+", " ", raw)
    clean_str = re.sub(r"\s+", " ", clean_str).strip()

    # 1. 提取年份 (通常为 4 位连续数字 19xx 或 20xx)
    year = ""
    year_match = re.search(r"\b(19\d\d|20\d\d)\b", clean_str)
    if year_match:
        year = year_match.group(1)
        # 从名称中移除年份
        raw_no_year = re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_str).strip()
    else:
        raw_no_year = clean_str

    # 2. 优先匹配多词复合品牌，再匹配单词品牌 (长词优先，双向支持短横线与空格互通)
    brand = ""
    brand_keys = sorted(BRAND_KNOWLEDGE.keys(), key=lambda x: len(re.sub(r"[-_]+", " ", x).split()), reverse=True)
    for b_key in brand_keys:
        b_key_norm = re.sub(r"[-_]+", " ", b_key).strip()
        pattern = r"(?i)\b" + re.escape(b_key_norm) + r"\b"
        if re.search(pattern, raw_no_year):
            brand = b_key
            raw_no_brand = re.sub(pattern, "", raw_no_year, count=1).strip()
            break
    else:
        # 未匹配到已知品牌，取首单词作为品牌兜底
        parts = raw_no_year.split()
        brand = parts[0] if parts else ""
        raw_no_brand = " ".join(parts[1:]) if len(parts) > 1 else ""

    # 3. 匹配车身形态 (Body Type)
    body_type = ""
    body_keys = [
        "Regular Cab", "Crew Cab", "Extended Cab", "SuperCab", "Quad Cab", "Club Cab",
        "Convertible", "Hatchback", "Minivan", "Coupe", "Sedan", "Wagon", "SUV", "Van", "Cab"
    ]
    for b_type in body_keys:
        pattern = r"(?i)\b" + re.escape(b_type) + r"\b"
        if re.search(pattern, raw_no_brand):
            # 统一将各种 Cab 归一到皮卡体系
            if "cab" in b_type.lower():
                body_type = "Cab"
            else:
                body_type = b_type
            raw_no_body = re.sub(pattern, "", raw_no_brand, count=1).strip()
            break
    else:
        raw_no_body = raw_no_brand

    # 4. 剩余的即为具体车系型号 (Model)
    model = re.sub(r"\s+", " ", raw_no_body).strip()
    if not model:
        model = brand

    # 5. 若未显式标注车身形态，进行智能语义推断，杜绝跑车/皮卡/SUV盲目兜底为三厢轿车
    if not body_type:
        supercar_brands = {"Ferrari", "Lamborghini", "Bugatti", "McLaren", "Spyker"}
        coupe_keywords = [
            "corvette", "viper", "gt-r", "gtr", "r8", "sls", "gallardo", "murcielago", "aventador",
            "huracan", "458", "488", "f12", "superleggera", "veyron", "chiron", "carrera", "boxster",
            "cayman", "911", "camaro", "mustang", "challenger", "tt", "tts", "z4", "370z", "350z",
            "brz", "gt86", "miata", "mx-5", "panamera", "gt", "slk", "sl-class", "cl-class"
        ]
        suv_keywords = [
            "wrangler", "cherokee", "land cruiser", "prado", "patrol", "explorer", "suburban", "tahoe",
            "escalade", "range rover", "discovery", "defender", "rav4", "cr-v", "crv", "highlander",
            "touareg", "tiguan", "cayenne", "macan", "x5", "x6", "x3", "x7", "q7", "q5", "q3",
            "gle", "gls", "glc", "g-class", "g-wagon", "outback", "forester", "sequoia", "expedition"
        ]
        pickup_keywords = [
            "f-150", "f-250", "f-350", "f 150", "f 250", "silverado", "sierra", "ram", "tundra",
            "tacoma", "hilux", "ranger", "colorado", "canyon", "d-max"
        ]
        van_keywords = ["savana", "express", "transit", "sprinter", "e-series", "e150", "e250", "caravan", "sienna", "odyssey"]

        lower_check = f"{model} {clean_str}".lower()
        if brand in supercar_brands or any(k in lower_check for k in coupe_keywords):
            body_type = "Coupe"
        elif any(k in lower_check for k in suv_keywords):
            body_type = "SUV"
        elif any(k in lower_check for k in pickup_keywords):
            body_type = "Cab"
        elif any(k in lower_check for k in van_keywords):
            body_type = "Van"
        else:
            body_type = "Sedan"

    brand_info = BRAND_KNOWLEDGE.get(brand)
    if not brand_info:
        for k, v in BRAND_KNOWLEDGE.items():
            if k.lower() == brand.lower():
                brand_info = v
                break
        else:
            brand_info = (brand, "国际主流造车阵营", "知名乘用车与商用车制造品牌", "现代乘用底盘与电控系统")
    brand_cn = brand_info[0]

    body_info = BODY_TYPE_KNOWLEDGE.get(body_type, BODY_TYPE_KNOWLEDGE.get("Sedan", {}))
    body_type_cn = body_info.get("name", "轿车 / 乘用车").split()[0]

    return {
        "brand": brand,
        "brand_cn": brand_cn,
        "model": model,
        "body_type": body_type or "Sedan",
        "body_type_cn": body_type_cn,
        "year": year
    }


def synthesize_vehicle_wiki(class_name: str) -> dict:
    """
    智能语义特征合成核心引擎
    通过对解构元素进行品牌图谱、车身形态与动力特性的多维匹配，0.1ms 内生成结构化中文深度卡片
    """
    parsed = parse_vehicle_class_name(class_name)
    brand_key = parsed["brand"]
    brand_cn = parsed["brand_cn"]
    model_name = parsed["model"]
    body_key = parsed["body_type"]
    body_type_cn = parsed["body_type_cn"]
    year_str = parsed["year"]

    # 查找品牌知识
    brand_info = BRAND_KNOWLEDGE.get(brand_key)
    if not brand_info:
        # 尝试大小写不敏感匹配
        for k, v in BRAND_KNOWLEDGE.items():
            if k.lower() == brand_key.lower():
                brand_info = v
                brand_key = k
                break
        else:
            brand_info = (brand_key, "国际主流造车阵营", "知名乘用车与商用车制造品牌", "现代高标准安全乘用底盘与电控系统")

    brand_cn, brand_origin, brand_desc, brand_feature = brand_info

    # 查找车身形态知识
    body_info = BODY_TYPE_KNOWLEDGE.get(body_key, BODY_TYPE_KNOWLEDGE["Sedan"])
    body_cn = body_info["name"]
    body_desc = body_info["desc"]
    dimensions = body_info["dimensions"]
    scenario = body_info["scenario"]
    safety_tips = body_info["safety_tips"]

    # 推导动力系统与能源构型
    lower_name = str(class_name).lower()
    if any(kw in lower_name for kw in ["hybrid", "phev"]):
        powertrain = "智能双模油电混合动力 (HEV/PHEV) · 阿特金森循环引擎 + 高效驱动电机"
    elif any(kw in lower_name for kw in ["ev", "electric"]) or brand_key in ["Tesla", "NIO", "XPeng", "Xiaomi", "Zeekr"]:
        powertrain = "高性能高压纯电驱动系统 (EV) · 磷酸铁锂/三元锂动力电池包 · 智能动能回收"
    elif brand_key in ["Ferrari", "Lamborghini", "Bugatti", "McLaren", "Aston Martin", "Porsche"]:
        powertrain = "大排量高转速高性能引擎 (V8/V10/W16/水平对置双涡轮) · 赛道级双离合中置/后置后驱"
    elif body_key == "Cab" or brand_key in ["GMC", "Ram"]:
        powertrain = "大排量高扭矩动力总成 (大排量 V8 / EcoBoost 涡轮) · 分时四驱带低速扭矩放大"
    elif any(kw in lower_name for kw in ["amg", "m3", "m5", "m6", "rs 4", "srt", "type-s", "type r"]):
        powertrain = "高性能性能版高功率涡轮增压发动机 · 运动型差速器与强化冷却排气系统"
    else:
        powertrain = "高燃效四冲程汽油/增压发动机 · 多挡手自一体/无级变速箱 · 前置前驱/全时四驱"

    # 推导准驾牌照
    if body_key == "Cab":
        license_plate = "轻型货车号牌 · 准驾车型 C1 (核载 5 人)"
    elif any(kw in lower_name for kw in ["ev", "electric"]) or brand_key in ["Tesla", "NIO", "XPeng", "Xiaomi"]:
        license_plate = "小型新能源汽车号牌 (绿牌) · 准驾车型 C1/C2"
    else:
        license_plate = "小型机动车号牌 (蓝牌) · 准驾车型 C1/C2"

    # 组合中文规范名称
    year_prefix = f"{year_str}款 " if year_str else ""
    cn_title = f"{brand_cn} ({brand_key}) {model_name} {year_prefix}· {body_cn.split()[0]}"

    # 组合车型归属
    category = f"{brand_origin} {brand_desc} · {body_cn}"

    # 组合技术特性
    tech_features = f"采用{body_desc}；搭载{brand_feature}；全车配置 ABS 防抱死、ESP 电子车身稳定与主被动安全防护气囊。"

    return {
        "cn_name": cn_title,
        "en_name": class_name,
        "brand": brand_key,
        "brand_cn": brand_cn,
        "model": model_name,
        "year": year_str if year_str else "-",
        "body_type": body_key,
        "body_type_cn": body_type_cn,
        "category": category,
        "license_plate": license_plate,
        "powertrain": powertrain,
        "dimensions": dimensions,
        "scenario": scenario,
        "tech_features": tech_features,
        "safety_tips": safety_tips
    }


def get_vehicle_wiki(class_name: str) -> dict:
    """
    根据识别到的类别英文或中文名称，检索或智能合成车辆知识库元数据
    """
    key = str(class_name).lower().strip()
    if not key:
        return VEHICLE_KNOWLEDGE_BASE["car"].copy()

    # 1. 通用大类直接精确匹配 (针对 COCO 等原生分类)
    if key in VEHICLE_KNOWLEDGE_BASE:
        res = VEHICLE_KNOWLEDGE_BASE[key].copy()
        res.setdefault("brand", "通用交通")
        res.setdefault("brand_cn", "通用目标")
        res.setdefault("model", res.get("cn_name", key).split()[0])
        res.setdefault("year", "-")
        res.setdefault("body_type", key)
        res.setdefault("body_type_cn", res.get("cn_name", key).split()[0])
        return res

    # 2. 常见中英文通用别名映射
    alias_map = {
        "qiche": "qiche",
        "car": "car",
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
        "bikes": "bicycle",
        "minivan": "van",
        "commercial van": "van",
        "sports_car": "sports car",
        "traffic_light": "traffic light",
        "stop_sign": "stop sign",
        "traffic-light": "traffic light",
        "stop-sign": "stop sign",
        "pedestrian": "person",
        "people": "person",
        "persons": "person",
        # 中文同义词
        "汽车": "qiche",
        "轿车": "car",
        "越野车": "suv",
        "大巴": "bus",
        "大客车": "bus",
        "公交车": "bus",
        "公共汽车": "bus",
        "卡车": "truck",
        "货车": "truck",
        "重卡": "truck",
        "摩托车": "motorcycle",
        "电摩": "motorcycle",
        "面包车": "van",
        "商务车": "mpv",
        "跑车": "sports car",
        "皮卡": "pickup",
        "自行车": "bicycle",
        "红绿灯": "traffic light",
        "停步标志": "stop sign",
        "行人": "person",
    }
    if key in alias_map:
        return VEHICLE_KNOWLEDGE_BASE[alias_map[key]]

    # 3. 检查是否为 Stanford Cars 196 类或细分复合车型名称 (如包含空格、下划线、短横线，或命中品牌库)
    clean_key = re.sub(r"[-_]+", " ", key).strip()
    if (" " in clean_key and len(clean_key) > 3) or any(b.lower() in clean_key for b in BRAND_KNOWLEDGE):
        try:
            return synthesize_vehicle_wiki(class_name)
        except Exception:
            pass

    # 4. 符号规范化匹配
    normalized = clean_key
    if normalized in VEHICLE_KNOWLEDGE_BASE:
        res = VEHICLE_KNOWLEDGE_BASE[normalized].copy()
        res.setdefault("brand", "通用交通")
        res.setdefault("brand_cn", "通用交通")
        res.setdefault("model", res.get("cn_name", normalized).split()[0])
        res.setdefault("year", "-")
        res.setdefault("body_type", normalized)
        res.setdefault("body_type_cn", res.get("cn_name", normalized).split()[0])
        return res
    if normalized in alias_map:
        target_k = alias_map[normalized]
        res = VEHICLE_KNOWLEDGE_BASE[target_k].copy()
        res.setdefault("brand", "通用交通")
        res.setdefault("brand_cn", "通用交通")
        res.setdefault("model", res.get("cn_name", target_k).split()[0])
        res.setdefault("year", "-")
        res.setdefault("body_type", target_k)
        res.setdefault("body_type_cn", res.get("cn_name", target_k).split()[0])
        return res

    # 5. 长词优先的子串包含匹配
    sorted_keys = sorted(VEHICLE_KNOWLEDGE_BASE.keys(), key=len, reverse=True)
    for k in sorted_keys:
        if k in normalized:
            res = VEHICLE_KNOWLEDGE_BASE[k].copy()
            res.setdefault("brand", "通用交通")
            res.setdefault("brand_cn", "通用交通")
            res.setdefault("model", res.get("cn_name", k).split()[0])
            res.setdefault("year", "-")
            res.setdefault("body_type", k)
            res.setdefault("body_type_cn", res.get("cn_name", k).split()[0])
            return res

    # 6. 智能合成兜底（将未知词也作为车型名称尝试提取）
    try:
        return synthesize_vehicle_wiki(class_name)
    except Exception:
        fallback = VEHICLE_KNOWLEDGE_BASE["car"].copy()
        fallback["cn_name"] = f"{class_name} (车辆目标)"
        fallback["en_name"] = class_name
        fallback.setdefault("brand", "未知品牌")
        fallback.setdefault("brand_cn", "未知品牌")
        fallback.setdefault("model", class_name)
        fallback.setdefault("year", "-")
        fallback.setdefault("body_type", "car")
        fallback.setdefault("body_type_cn", "乘用车")
        return fallback


# 别名导出
get_qiche_wiki = get_vehicle_wiki
get_plant_wiki = get_vehicle_wiki
