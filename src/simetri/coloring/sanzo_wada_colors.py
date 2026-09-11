"""Sanzo Wada Dictionary of Color entries as ``Color`` dataclasses."""

from dataclasses import dataclass
from random import choice


@dataclass
class Color:
    """One Sanzo Wada dictionary color with swatch and color-space fields.

    Attributes:
        name: Color name.
        combinations: Related swatch/combination indices.
        swatch: Swatch index.
        cmyk: CMYK components.
        lab: CIELAB components.
        rgb: RGB components.
        hex: Hex color string.
    """

    name: str
    combinations: list
    swatch: int
    cmyk: list
    lab: list
    rgb: list
    hex: str


Hermosa_Pink = Color(
    name="Hermosa_Pink",
    combinations=[176, 227, 273],
    swatch=0,
    cmyk=[0, 30, 6, 0],
    lab=[83.42717631799802, 22.136186770428026, 1.6381322957198563],
    rgb=[249, 193, 206],
    hex="#f9c1ce",
)
Corinthian_Pink = Color(
    name="Corinthian_Pink",
    combinations=[27, 43, 87, 97, 128, 169, 174, 206, 246, 254, 264, 342],
    swatch=0,
    cmyk=[0, 35, 15, 0],
    lab=[80.34637979705501, 25.369649805447466, 7.879377431906619],
    rgb=[248, 182, 186],
    hex="#f8b6ba",
)
Cameo_Pink = Color(
    name="Cameo_Pink",
    combinations=[101, 105, 116, 120, 165, 231],
    swatch=0,
    cmyk=[10, 32, 19, 0],
    lab=[77.21675440604257, 17.198443579766547, 4.949416342412462],
    rgb=[224, 179, 182],
    hex="#e0b3b6",
)
Fawn = Color(
    name="Fawn",
    combinations=[18, 125, 308],
    swatch=0,
    cmyk=[18, 31, 30, 0],
    lab=[74.48996719310291, 11.194552529182886, 9.38521400778211],
    rgb=[209, 176, 167],
    hex="#d1b0a7",
)
Light_Brown_Drab = Color(
    name="Light_Brown_Drab",
    combinations=[35, 68, 185, 191, 223, 239, 244, 268, 285, 321],
    swatch=0,
    cmyk=[8, 30, 20, 25],
    lab=[64.1168841077287, 13.023346303501938, 5.3424124513618665],
    rgb=[181, 147, 146],
    hex="#b59392",
)
Coral_Red = Color(
    name="Coral_Red",
    combinations=[92, 123, 320, 332],
    swatch=0,
    cmyk=[0, 55, 40, 0],
    lab=[70.28305485618371, 39.28793774319067, 23.190661478599225],
    rgb=[245, 142, 132],
    hex="#f58e84",
)
Fresh_Color = Color(
    name="Fresh_Color",
    combinations=[240],
    swatch=0,
    cmyk=[0, 53, 45, 0],
    lab=[70.96208133058671, 37.48249027237355, 27.29571984435799],
    rgb=[246, 145, 126],
    hex="#f6917e",
)
Grenadine_Pink = Color(
    name="Grenadine_Pink",
    combinations=[6, 21, 112, 166, 193, 201, 230, 300, 315, 341],
    swatch=0,
    cmyk=[0, 62, 58, 0],
    lab=[66.89402609292745, 43.82879377431905, 34.70038910505838],
    rgb=[244, 128, 103],
    hex="#f48067",
)
Eosine_Pink = Color(
    name="Eosine_Pink",
    combinations=[34, 59, 90, 108, 134, 153, 197, 242, 248, 276, 287, 314, 327, 336],
    swatch=0,
    cmyk=[0, 63, 23, 0],
    lab=[67.25108720531014, 46.68482490272373, 10.29571984435799],
    rgb=[243, 127, 148],
    hex="#f37f94",
)
Spinel_Red = Color(
    name="Spinel_Red",
    combinations=[14, 147, 165, 184, 195, 224, 277],
    swatch=0,
    cmyk=[0, 70, 21, 0],
    lab=[64.53040360112917, 52.18677042801556, 8.291828793774329],
    rgb=[242, 114, 145],
    hex="#f27291",
)
Old_Rose = Color(
    name="Old_Rose",
    combinations=[55, 162, 260, 265],
    swatch=0,
    cmyk=[15, 70, 40, 0],
    lab=[58.771648737315935, 42.1284046692607, 12.307392996108945],
    rgb=[212, 109, 122],
    hex="#d46d7a",
)
Eugenia_Red_A = Color(
    name="Eugenia_Red_A",
    combinations=[284],
    swatch=0,
    cmyk=[7, 76, 60, 0],
    lab=[58.36575875486381, 50.754863813229576, 28.536964980544752],
    rgb=[226, 98, 94],
    hex="#e2625e",
)
Eugenia_Red_B = Color(
    name="Eugenia_Red_B",
    combinations=[17, 77, 252, 262, 270, 280, 282, 325],
    swatch=0,
    cmyk=[0, 80, 50, 10],
    lab=[54.2748149843595, 54.696498054474716, 23.661478599221795],
    rgb=[218, 82, 93],
    hex="#da525d",
)
Raw_Sienna = Color(
    name="Raw_Sienna",
    combinations=[
        3,
        13,
        33,
        70,
        86,
        130,
        131,
        182,
        243,
        247,
        252,
        255,
        268,
        269,
        279,
        293,
        298,
        319,
        327,
    ],
    swatch=0,
    cmyk=[18, 58, 100, 12],
    lab=[55.063706416418704, 25.548638132295707, 52.17120622568095],
    rgb=[187, 113, 37],
    hex="#bb7125",
)
Vinaceous_Tawny = Color(
    name="Vinaceous_Tawny",
    combinations=[40, 85, 244],
    swatch=0,
    cmyk=[17, 72, 100, 6],
    lab=[53.226520180056454, 38.17509727626458, 50.18677042801556],
    rgb=[197, 97, 39],
    hex="#c56127",
)
Jasper_Red = Color(
    name="Jasper_Red",
    combinations=[155, 194, 216, 219],
    swatch=0,
    cmyk=[2, 83, 100, 0],
    lab=[56.809338521400775, 58.41245136186771, 57.630350194552534],
    rgb=[235, 83, 36],
    hex="#eb5324",
)
Spectrum_Red = Color(
    name="Spectrum_Red",
    combinations=[257, 266, 301, 322],
    swatch=0,
    cmyk=[5, 100, 100, 0],
    lab=[49.60708018616007, 70.9260700389105, 50.077821011673166],
    rgb=[227, 31, 38],
    hex="#e31f26",
)
Red_Orange = Color(
    name="Red_Orange",
    combinations=[31, 164, 179, 241, 264],
    swatch=0,
    cmyk=[9, 90, 100, 0],
    lab=[51.8272678721294, 60.64202334630349, 50.88326848249028],
    rgb=[221, 64, 39],
    hex="#dd4027",
)
Etruscan_Red = Color(
    name="Etruscan_Red",
    combinations=[25, 47, 97, 137, 152, 185, 275],
    swatch=0,
    cmyk=[16, 80, 74, 6],
    lab=[50.74845502403296, 45.88326848249028, 31.0544747081712],
    rgb=[197, 83, 71],
    hex="#c55347",
)
Burnt_Sienna = Color(
    name="Burnt_Sienna",
    combinations=[198, 242, 263, 285, 286, 297, 312, 333, 343],
    swatch=0,
    cmyk=[22, 76, 100, 15],
    lab=[46.43778133821622, 36.23346303501947, 43.7898832684825],
    rgb=[174, 82, 36],
    hex="#ae5224",
)
Ochre_Red = Color(
    name="Ochre_Red",
    combinations=[199, 283],
    swatch=0,
    cmyk=[18, 73, 63, 20],
    lab=[46.81773098344396, 35.73540856031127, 21.848249027237358],
    rgb=[171, 84, 77],
    hex="#ab544d",
)
Scarlet = Color(
    name="Scarlet",
    combinations=[136, 308, 332],
    swatch=0,
    cmyk=[10, 95, 72, 7],
    lab=[46.54612039368276, 61.52529182879377, 28.902723735408557],
    rgb=[203, 47, 67],
    hex="#cb2f43",
)
Carmine = Color(
    name="Carmine",
    combinations=[39, 117, 122, 154, 225, 232, 307, 313],
    swatch=0,
    cmyk=[0, 100, 75, 16],
    lab=[44.29388876173037, 67.18677042801556, 33.71206225680933],
    rgb=[204, 18, 54],
    hex="#cc1236",
)
Indian_Lake = Color(
    name="Indian_Lake",
    combinations=[299, 331],
    swatch=0,
    cmyk=[12, 89, 35, 9],
    lab=[47.54253452353704, 57.21400778210116, 6.677042801556411],
    rgb=[197, 60, 105],
    hex="#c53c69",
)
Rosolanc_Purple = Color(
    name="Rosolanc_Purple",
    combinations=[48, 144, 170, 204, 277, 346],
    swatch=0,
    cmyk=[30, 90, 33, 0],
    lab=[45.93881132219425, 52.365758754863805, -3.159533073929964],
    rgb=[183, 63, 116],
    hex="#b73f74",
)
Pomegranite_Purple = Color(
    name="Pomegranite_Purple",
    combinations=[220, 271],
    swatch=0,
    cmyk=[23, 100, 50, 6],
    lab=[41.42519264515145, 60.96108949416342, 8.400778210116727],
    rgb=[183, 31, 87],
    hex="#b71f57",
)
Hydrangea_Red = Color(
    name="Hydrangea_Red",
    combinations=[142],
    swatch=0,
    cmyk=[38, 90, 70, 0],
    lab=[43.070115205615316, 44.35408560311285, 14.182879377431902],
    rgb=[169, 65, 81],
    hex="#a94151",
)
Brick_Red = Color(
    name="Brick_Red",
    combinations=[37, 108, 246, 322, 328],
    swatch=0,
    cmyk=[22, 84, 100, 18],
    lab=[42.45975432974746, 41.50972762645915, 40.688715953307394],
    rgb=[168, 66, 34],
    hex="#a84222",
)
Carmine_Red = Color(
    name="Carmine_Red",
    combinations=[35, 51, 104, 130, 181, 200, 221, 228, 233, 237, 245, 338],
    swatch=0,
    cmyk=[25, 95, 80, 16],
    lab=[38.71366445410849, 50.22957198443581, 24.688715953307394],
    rgb=[166, 44, 55],
    hex="#a62c37",
)
Pompeian_Red = Color(
    name="Pompeian_Red",
    combinations=[30, 71, 120, 212, 311, 324],
    swatch=0,
    cmyk=[18, 97, 74, 19],
    lab=[38.857099259937435, 54.805447470817114, 23.844357976653697],
    rgb=[171, 36, 57],
    hex="#ab2439",
)
Red = Color(
    name="Red",
    combinations=[251, 261],
    swatch=0,
    cmyk=[30, 100, 70, 10],
    lab=[37.86678873884184, 54.58754863813229, 15.369649805447466],
    rgb=[167, 33, 68],
    hex="#a72144",
)
Brown = Color(
    name="Brown",
    combinations=[110, 121, 145, 161],
    swatch=0,
    cmyk=[35, 74, 90, 35],
    lab=[35.06981002517738, 23.47470817120623, 28.571984435797674],
    rgb=[124, 66, 38],
    hex="#7c4226",
)
Hay_s_Russet = Color(
    name="Hay_s_Russet",
    combinations=[58, 82, 95, 152, 186, 231, 249, 304, 314, 336, 345],
    swatch=0,
    cmyk=[37, 85, 87, 35],
    lab=[31.235217822537575, 30.657587548638134, 23.509727626459153],
    rgb=[121, 51, 39],
    hex="#793327",
)
Vandyke_Red = Color(
    name="Vandyke_Red",
    combinations=[16, 133, 147, 316, 335],
    swatch=0,
    cmyk=[32, 95, 95, 33],
    lab=[30.157930876630807, 40.59922178988327, 27.607003891050596],
    rgb=[130, 36, 31],
    hex="#82241f",
)
Pansy_Purple = Color(
    name="Pansy_Purple",
    combinations=[157, 273],
    swatch=0,
    cmyk=[34, 100, 60, 34],
    lab=[27.605096513313494, 45.922178988326834, 6.089494163424121],
    rgb=[125, 19, 58],
    hex="#7d133a",
)
Pale_Burnt_Lake = Color(
    name="Pale_Burnt_Lake",
    combinations=[124, 171, 177, 205, 217, 258, 269, 283],
    swatch=0,
    cmyk=[25, 90, 80, 40],
    lab=[30.18234531166552, 39.03501945525292, 22.712062256809332],
    rgb=[128, 38, 38],
    hex="#802626",
)
Violet_Red = Color(
    name="Violet_Red",
    combinations=[9],
    swatch=0,
    cmyk=[75, 100, 50, 5],
    lab=[27.77141985198749, 30.38521400778211, -17.743190661478593],
    rgb=[100, 45, 94],
    hex="#642d5e",
)
Vistoris_Lake = Color(
    name="Vistoris_Lake",
    combinations=[63, 91, 165, 226, 290, 337],
    swatch=0,
    cmyk=[40, 71, 55, 40],
    lab=[33.090714885175856, 20.29571984435799, 6.101167315175104],
    rgb=[109, 65, 69],
    hex="#6d4145",
)
Sulpher_Yellow = Color(
    name="Sulpher_Yellow",
    combinations=[
        52,
        72,
        80,
        104,
        132,
        135,
        151,
        208,
        246,
        254,
        270,
        294,
        296,
        310,
        315,
        320,
        321,
        326,
    ],
    swatch=1,
    cmyk=[4, 4, 28, 0],
    lab=[93.22499427786678, -1.7315175097276239, 21.64980544747081],
    rgb=[245, 236, 194],
    hex="#f5ecc2",
)
Pale_Lemon_Yellow = Color(
    name="Pale_Lemon_Yellow",
    combinations=[
        3,
        31,
        60,
        76,
        99,
        109,
        111,
        169,
        185,
        195,
        203,
        228,
        241,
        261,
        272,
        281,
        290,
        292,
        336,
    ],
    swatch=1,
    cmyk=[0, 4, 38, 0],
    lab=[94.8577096208133, -0.14396887159533378, 34.08560311284046],
    rgb=[255, 239, 174],
    hex="#ffefae",
)
Naples_Yellow = Color(
    name="Naples_Yellow",
    combinations=[14, 115, 166, 193, 303, 325],
    swatch=1,
    cmyk=[2, 7, 44, 0],
    lab=[91.7265583276112, 0.31906614785992815, 36.56809338521401],
    rgb=[251, 230, 160],
    hex="#fbe6a0",
)
Ivory_Buff = Color(
    name="Ivory_Buff",
    combinations=[
        11,
        50,
        94,
        102,
        126,
        178,
        184,
        190,
        209,
        214,
        235,
        243,
        262,
        266,
        301,
        343,
    ],
    swatch=1,
    cmyk=[8, 15, 40, 0],
    lab=[85.59700923170824, 3.377431906614788, 27.268482490272362],
    rgb=[235, 211, 162],
    hex="#ebd3a2",
)
Seashell_Pink = Color(
    name="Seashell_Pink",
    combinations=[45, 84, 88, 113, 150, 176, 194, 276, 334],
    swatch=1,
    cmyk=[0, 19, 23, 0],
    lab=[88.12848096437018, 12.08949416342412, 17.268482490272362],
    rgb=[253, 212, 189],
    hex="#fdd4bd",
)
Light_Pinkish_Cinnamon = Color(
    name="Light_Pinkish_Cinnamon",
    combinations=[317],
    swatch=1,
    cmyk=[0, 25, 40, 0],
    lab=[84.2069123369192, 15.630350194552534, 29.809338521400775],
    rgb=[252, 199, 155],
    hex="#fcc79b",
)
Pinkish_Cinnamon = Color(
    name="Pinkish_Cinnamon",
    combinations=[78, 161, 175, 232, 258, 263, 292, 305, 310],
    swatch=1,
    cmyk=[5, 32, 53, 0],
    lab=[77.84695201037613, 17.229571984435808, 34.747081712062254],
    rgb=[238, 180, 128],
    hex="#eeb480",
)
Cinnamon_Buff = Color(
    name="Cinnamon_Buff",
    combinations=[23, 127, 137, 180, 210, 234, 246, 323, 344],
    swatch=1,
    cmyk=[0, 25, 57, 0],
    lab=[83.52941176470588, 14.743190661478593, 43.665369649805456],
    rgb=[253, 197, 126],
    hex="#fdc57e",
)
Cream_Yellow = Color(
    name="Cream_Yellow",
    combinations=[122, 192, 215, 226, 267, 278, 294, 295, 300, 302, 304, 311, 329, 342],
    swatch=1,
    cmyk=[0, 28, 68, 0],
    lab=[81.81277180132753, 16.46692607003891, 51.97665369649806],
    rgb=[253, 191, 104],
    hex="#fdbf68",
)
Golden_Yellow = Color(
    name="Golden_Yellow",
    combinations=[26, 81, 132, 138, 140, 179, 206, 229, 309, 315],
    swatch=1,
    cmyk=[2, 42, 74, 0],
    lab=[73.85671778439001, 25.94163424124514, 51.1828793774319],
    rgb=[243, 162, 87],
    hex="#f3a257",
)
Vinaceous_Cinnamon = Color(
    name="Vinaceous_Cinnamon",
    combinations=[203, 205, 213, 256, 260, 279, 299],
    swatch=1,
    cmyk=[4, 40, 42, 0],
    lab=[74.98741130693523, 24.739299610894932, 24.813229571984436],
    rgb=[238, 167, 140],
    hex="#eea78c",
)
Ochraceous_Salmon = Color(
    name="Ochraceous_Salmon",
    combinations=[32, 71, 121, 186, 217, 220, 223, 238, 296, 339],
    swatch=1,
    cmyk=[15, 38, 55, 0],
    lab=[71.44731822690166, 16.459143968871587, 29.101167315175104],
    rgb=[216, 163, 123],
    hex="#d8a37b",
)
Isabella_Color = Color(
    name="Isabella_Color",
    combinations=[4, 12, 241, 292],
    swatch=1,
    cmyk=[15, 28, 60, 10],
    lab=[69.68337529564354, 7.070038910505843, 33.02334630350194],
    rgb=[197, 165, 110],
    hex="#c5a56e",
)
Maple = Color(
    name="Maple",
    combinations=[282],
    swatch=1,
    cmyk=[5, 26, 56, 20],
    lab=[68.19409475852598, 9.591439688715951, 32.94941634241246],
    rgb=[197, 159, 107],
    hex="#c59f6b",
)
Olive_Buff = Color(
    name="Olive_Buff",
    combinations=[83, 175, 200, 330, 348],
    swatch=1,
    cmyk=[16, 6, 42, 12],
    lab=[78.04837109941252, -7.108949416342412, 23.657587548638134],
    rgb=[193, 196, 148],
    hex="#c1c494",
)
Ecru = Color(
    name="Ecru",
    combinations=[167, 249, 275, 279, 292, 302, 317, 327],
    swatch=1,
    cmyk=[20, 25, 40, 6],
    lab=[72.19043259327077, 4.2295719844358075, 16.731517509727638],
    rgb=[194, 174, 147],
    hex="#c2ae93",
)
Yellow = Color(
    name="Yellow",
    combinations=[22, 62, 68, 154, 240, 251, 295, 313],
    swatch=1,
    cmyk=[0, 0, 100, 0],
    lab=[94.96452277409018, -6.45525291828794, 95.57976653696497],
    rgb=[255, 242, 0],
    hex="#fff200",
)
Lemon_Yellow = Color(
    name="Lemon_Yellow",
    combinations=[
        45,
        123,
        138,
        158,
        168,
        173,
        189,
        210,
        253,
        259,
        289,
        298,
        306,
        317,
        333,
    ],
    swatch=1,
    cmyk=[5, 0, 85, 0],
    lab=[92.65583276112001, -9.389105058365757, 77.7626459143969],
    rgb=[248, 237, 67],
    hex="#f8ed43",
)
Apricot_Yellow = Color(
    name="Apricot_Yellow",
    combinations=[107, 129, 163, 198, 213, 247, 265, 284, 305, 319],
    swatch=1,
    cmyk=[0, 10, 100, 0],
    lab=[89.35072861829556, 1.9066147859922182, 89.45525291828793],
    rgb=[255, 221, 0],
    hex="#ffdd00",
)
Pyrite_Yellow = Color(
    name="Pyrite_Yellow",
    combinations=[239, 250, 255, 287],
    swatch=1,
    cmyk=[23, 25, 80, 0],
    lab=[73.56221866178377, 0.4241245136186649, 49.8171206225681],
    rgb=[202, 179, 86],
    hex="#cab356",
)
Olive_Ocher = Color(
    name="Olive_Ocher",
    combinations=[66, 148, 149, 156, 157, 249, 278],
    swatch=1,
    cmyk=[18, 26, 90, 0],
    lab=[74.87907225146868, 3.568093385214013, 61.805447470817114],
    rgb=[214, 180, 62],
    hex="#d6b43e",
)
Yellow_Ocher = Color(
    name="Yellow_Ocher",
    combinations=[42, 96, 118, 124, 126, 191, 222, 325],
    swatch=1,
    cmyk=[12, 28, 88, 0],
    lab=[76.19134813458457, 8.44357976653697, 62.7898832684825],
    rgb=[226, 181, 64],
    hex="#e2b540",
)
Orange_Yellow = Color(
    name="Orange_Yellow",
    combinations=[114, 148, 153, 164, 170, 257, 286, 338],
    swatch=1,
    cmyk=[0, 33, 100, 0],
    lab=[78.49393453879605, 19.571984435797674, 78.21789883268482],
    rgb=[252, 179, 21],
    hex="#fcb315",
)
Yellow_Orange = Color(
    name="Yellow_Orange",
    combinations=[2, 53, 89, 151, 171, 209, 222, 235, 267, 288, 297, 312, 319, 335],
    swatch=1,
    cmyk=[0, 45, 100, 0],
    lab=[72.95948729686427, 29.404669260700388, 72.94163424124514],
    rgb=[249, 157, 27],
    hex="#f99d1b",
)
Apricot_Orange = Color(
    name="Apricot_Orange",
    combinations=[211, 253, 309, 328],
    swatch=1,
    cmyk=[0, 55, 75, 0],
    lab=[69.3736171511406, 37.688715953307394, 49.64980544747081],
    rgb=[246, 140, 80],
    hex="#f68c50",
)
Orange = Color(
    name="Orange",
    combinations=[7, 46, 141, 144, 149, 256, 272],
    swatch=1,
    cmyk=[0, 68, 100, 0],
    lab=[63.79797055008774, 47.159533073929964, 64.715953307393],
    rgb=[243, 116, 32],
    hex="#f37420",
)
Peach_Red = Color(
    name="Peach_Red",
    combinations=[115, 250, 274, 285, 298, 303, 326, 340],
    swatch=1,
    cmyk=[0, 80, 90, 0],
    lab=[59.09666590371557, 57.54863813229571, 54.35019455252919],
    rgb=[241, 90, 48],
    hex="#f15a30",
)
English_Red = Color(
    name="English_Red",
    combinations=[1, 131, 190, 308, 339],
    swatch=1,
    cmyk=[13, 73, 100, 0],
    lab=[57.2152285038529, 43.29961089494162, 54.361867704280144],
    rgb=[217, 102, 41],
    hex="#d96629",
)
Cinnamon_Rufous = Color(
    name="Cinnamon_Rufous",
    combinations=[8, 10, 103, 158, 172, 204, 206],
    swatch=1,
    cmyk=[20, 60, 82, 5],
    lab=[57.233539330128934, 27.76653696498053, 40.22957198443581],
    rgb=[194, 117, 68],
    hex="#c27544",
)
Orange_Rufous = Color(
    name="Orange_Rufous",
    combinations=[91, 102, 222],
    swatch=1,
    cmyk=[18, 65, 100, 8],
    lab=[54.56168459601739, 31.61089494163423, 51.245136186770424],
    rgb=[193, 107, 39],
    hex="#c16b27",
)
Sulphine_Yellow = Color(
    name="Sulphine_Yellow",
    combinations=[36, 65, 142, 160, 252],
    swatch=1,
    cmyk=[24, 32, 100, 4],
    lab=[67.23887998779279, 4.346303501945528, 60.55252918287937],
    rgb=[193, 159, 44],
    hex="#c19f2c",
)
Khaki = Color(
    name="Khaki",
    combinations=[129, 146, 159, 236, 248],
    swatch=1,
    cmyk=[24, 45, 100, 6],
    lab=[61.01930266269932, 13.859922178988313, 55.05058365758754],
    rgb=[188, 137, 43],
    hex="#bc892b",
)
Citron_Yellow = Color(
    name="Citron_Yellow",
    combinations=[40, 87, 145, 150, 153, 196, 305, 323],
    swatch=1,
    cmyk=[35, 17, 95, 0],
    lab=[72.4650949874113, -13.062256809338521, 58.17509727626458],
    rgb=[178, 183, 62],
    hex="#b2b73e",
)
Citrine = Color(
    name="Citrine",
    combinations=[59, 93, 132, 133, 262],
    swatch=1,
    cmyk=[36, 32, 100, 0],
    lab=[65.5588616769665, -2.579766536964982, 54.486381322957186],
    rgb=[176, 159, 54],
    hex="#b09f36",
)
Buffy_Citrine = Color(
    name="Buffy_Citrine",
    combinations=[100, 177, 233],
    swatch=1,
    cmyk=[42, 40, 82, 8],
    lab=[56.61860074769207, 0.0, 33.29961089494162],
    rgb=[150, 135, 77],
    hex="#96874d",
)
Dark_Citrine = Color(
    name="Dark_Citrine",
    combinations=[10, 41, 274, 304],
    swatch=1,
    cmyk=[38, 34, 67, 20],
    lab=[54.71427481498436, -1.4046692607003877, 23.066147859922182],
    rgb=[139, 131, 91],
    hex="#8b835b",
)
Light_Grayish_Olive = Color(
    name="Light_Grayish_Olive",
    combinations=[107, 184],
    swatch=1,
    cmyk=[43, 36, 62, 19],
    lab=[53.185320820935374, -2.276264591439684, 17.019455252918277],
    rgb=[132, 128, 97],
    hex="#848061",
)
Krongbergs_Green = Color(
    name="Krongbergs_Green",
    combinations=[29],
    swatch=1,
    cmyk=[48, 35, 70, 12],
    lab=[55.12474250400549, -6.1867704280155635, 21.509727626459153],
    rgb=[132, 135, 94],
    hex="#84875e",
)
Olive = Color(
    name="Olive",
    combinations=[96, 201, 254, 258, 277, 310, 334],
    swatch=1,
    cmyk=[48, 38, 100, 15],
    lab=[52.213321126115815, -5.727626459143963, 41.486381322957186],
    rgb=[131, 126, 49],
    hex="#837e31",
)
Orange_Citrine = Color(
    name="Orange_Citrine",
    combinations=[212, 342],
    swatch=1,
    cmyk=[28, 48, 92, 24],
    lab=[50.24185549706264, 11.517509727626447, 42.046692607003905],
    rgb=[152, 111, 45],
    hex="#986f2d",
)
Sudan_Brown = Color(
    name="Sudan_Brown",
    combinations=[207, 214, 273],
    swatch=1,
    cmyk=[25, 60, 65, 19],
    lab=[49.8619058518349, 23.03112840466926, 22.451361867704293],
    rgb=[163, 103, 82],
    hex="#a36752",
)
Olive_Green = Color(
    name="Olive_Green",
    combinations=[66, 243, 270, 297],
    swatch=1,
    cmyk=[56, 40, 85, 22],
    lab=[46.38590066376745, -8.494163424124508, 26.486381322957186],
    rgb=[107, 113, 64],
    hex="#6b7140",
)
Light_Brownish_Olive = Color(
    name="Light_Brownish_Olive",
    combinations=[199, 318],
    swatch=1,
    cmyk=[42, 46, 73, 24],
    lab=[47.50743877317464, 3.747081712062254, 22.194552529182886],
    rgb=[128, 110, 75],
    hex="#806e4b",
)
Deep_Grayish_Olive = Color(
    name="Deep_Grayish_Olive",
    combinations=[146, 343],
    swatch=1,
    cmyk=[50, 48, 78, 37],
    lab=[38.605325398641945, -0.28404669260700643, 19.94163424124514],
    rgb=[99, 90, 58],
    hex="#635a3a",
)
Pale_Raw_Umber = Color(
    name="Pale_Raw_Umber",
    combinations=[26, 73, 160, 234, 296],
    swatch=1,
    cmyk=[46, 63, 87, 32],
    lab=[37.065690089265274, 11.190661478599225, 24.957198443579756],
    rgb=[113, 80, 47],
    hex="#71502f",
)
Sepia = Color(
    name="Sepia",
    combinations=[24, 288],
    swatch=1,
    cmyk=[48, 60, 100, 40],
    lab=[33.82619974059663, 6.867704280155635, 30.23346303501947],
    rgb=[100, 75, 30],
    hex="#644b1e",
)
Madder_Brown = Color(
    name="Madder_Brown",
    combinations=[28, 79, 98, 173, 237, 275, 323],
    swatch=1,
    cmyk=[36, 88, 100, 38],
    lab=[29.08522163729305, 32.0, 29.280155642023345],
    rgb=[118, 44, 25],
    hex="#762c19",
)
Mars_Brown_Tobacco = Color(
    name="Mars_Brown_Tobacco",
    combinations=[19],
    swatch=1,
    cmyk=[39, 76, 100, 47],
    lab=[28.032349126420996, 20.451361867704293, 29.836575875486375],
    rgb=[101, 53, 20],
    hex="#653514",
)
Vandyke_Brown = Color(
    name="Vandyke_Brown",
    combinations=[110, 113, 118, 182, 192, 328],
    swatch=1,
    cmyk=[56, 71, 97, 52],
    lab=[23.811703669794767, 8.49805447470817, 22.101167315175104],
    rgb=[75, 51, 23],
    hex="#4b3317",
)
Turquoise_Green = Color(
    name="Turquoise_Green",
    combinations=[
        36,
        74,
        147,
        163,
        173,
        202,
        223,
        230,
        263,
        272,
        285,
        293,
        300,
        305,
        317,
        346,
    ],
    swatch=2,
    cmyk=[29, 0, 24, 0],
    lab=[85.26283665217059, -16.891050583657588, 4.595330739299612],
    rgb=[181, 222, 204],
    hex="#b5decc",
)
Glaucous_Green = Color(
    name="Glaucous_Green",
    combinations=[7, 150, 171, 207, 239, 260],
    swatch=2,
    cmyk=[30, 9, 24, 0],
    lab=[80.29755092698558, -10.237354085603116, 2.2996108949416225],
    rgb=[180, 205, 194],
    hex="#b4cdc2",
)
Dark_Greenish_Glaucous = Color(
    name="Dark_Greenish_Glaucous",
    combinations=[264, 311],
    swatch=2,
    cmyk=[30, 15, 36, 0],
    lab=[77.09010452429999, -7.5564202334630295, 11.369649805447466],
    rgb=[183, 194, 169],
    hex="#b7c2a9",
)
Yellow_Green = Color(
    name="Yellow_Green",
    combinations=[111, 141, 276, 326, 334],
    swatch=2,
    cmyk=[35, 0, 72, 0],
    lab=[80.60883497367819, -24.599221789883273, 43.91050583657588],
    rgb=[175, 212, 114],
    hex="#afd472",
)
Light_Green_Yellow = Color(
    name="Light_Green_Yellow",
    combinations=[61, 289, 291, 311, 346],
    swatch=2,
    cmyk=[26, 5, 85, 0],
    lab=[81.13527122911421, -15.863813229571988, 60.55642023346303],
    rgb=[199, 209, 79],
    hex="#c7d14f",
)
Night_Green = Color(
    name="Night_Green",
    combinations=[19, 32, 158, 326],
    swatch=2,
    cmyk=[52, 0, 100, 0],
    lab=[73.30739299610894, -36.03112840466926, 57.0272373540856],
    rgb=[135, 197, 64],
    hex="#87c540",
)
Olive_Yellow = Color(
    name="Olive_Yellow",
    combinations=[124, 211, 265, 347],
    swatch=2,
    cmyk=[40, 30, 80, 0],
    lab=[65.48561837186236, -6.054474708171213, 37.836575875486375],
    rgb=[166, 161, 89],
    hex="#a6a159",
)
Artemesia_Green = Color(
    name="Artemesia_Green",
    combinations=[293, 312],
    swatch=2,
    cmyk=[57, 28, 39, 8],
    lab=[58.101777676050965, -12.832684824902728, -2.891050583657588],
    rgb=[112, 147, 144],
    hex="#709390",
)
Andover_Green = Color(
    name="Andover_Green",
    combinations=[244, 346],
    swatch=2,
    cmyk=[60, 40, 50, 10],
    lab=[51.212329289692526, -7.7626459143968845, 1.5369649805447523],
    rgb=[109, 126, 119],
    hex="#6d7e77",
)
Rainette_Green = Color(
    name="Rainette_Green",
    combinations=[73, 162, 188, 266, 301],
    swatch=2,
    cmyk=[42, 20, 62, 10],
    lab=[63.738460364690624, -12.525291828793769, 22.44747081712063],
    rgb=[143, 160, 113],
    hex="#8fa071",
)
Chromium_Green = Color(
    name="Chromium_Green",
    combinations=[105, 200, 219, 283],
    swatch=2,
    cmyk=[50, 16, 58, 20],
    lab=[57.87289234760052, -18.147859922178995, 14.992217898832678],
    rgb=[113, 148, 112],
    hex="#719470",
)
Pistachio_Green = Color(
    name="Pistachio_Green",
    combinations=[127, 137],
    swatch=2,
    cmyk=[64, 29, 56, 6],
    lab=[55.82055390249485, -18.642023346303503, 5.688715953307394],
    rgb=[100, 143, 123],
    hex="#648f7b",
)
Sea_Green = Color(
    name="Sea_Green",
    combinations=[17, 21, 58, 86, 133, 250, 260, 284, 291, 340, 347],
    swatch=2,
    cmyk=[80, 0, 51, 0],
    lab=[65.05226214999618, -48.90272373540856, 0.4591439688715866],
    rgb=[0, 180, 155],
    hex="#00b49b",
)
Benzol_Green = Color(
    name="Benzol_Green",
    combinations=[15, 54, 92, 122, 155, 247, 266, 267, 281, 304, 306],
    swatch=2,
    cmyk=[100, 15, 55, 0],
    lab=[53.56221866178378, -54.62645914396887, -8.307392996108945],
    rgb=[0, 151, 141],
    hex="#00978d",
)
Light_Porcelain_Green = Color(
    name="Light_Porcelain_Green",
    combinations=[44, 193, 328],
    swatch=2,
    cmyk=[86, 22, 50, 3],
    lab=[53.29518577859159, -37.52140077821012, -6.832684824902728],
    rgb=[0, 144, 138],
    hex="#00908a",
)
Green = Color(
    name="Green",
    combinations=[198, 216, 293],
    swatch=2,
    cmyk=[75, 21, 73, 0],
    lab=[57.770656900892654, -34.50972762645914, 15.58754863813229],
    rgb=[72, 155, 110],
    hex="#489b6e",
)
Dull_Viridian_Green = Color(
    name="Dull_Viridian_Green",
    combinations=[136, 256, 306, 316],
    swatch=2,
    cmyk=[90, 20, 80, 0],
    lab=[53.49965667200732, -49.00389105058366, 14.43968871595331],
    rgb=[0, 148, 101],
    hex="#009465",
)
Oil_Green = Color(
    name="Oil_Green",
    combinations=[245, 299, 320],
    swatch=2,
    cmyk=[53, 28, 100, 8],
    lab=[57.639429312581065, -16.01167315175097, 43.76653696498053],
    rgb=[129, 146, 56],
    hex="#819238",
)
Diamine_Green = Color(
    name="Diamine_Green",
    combinations=[38, 146, 217, 242, 251, 313],
    swatch=2,
    cmyk=[87, 32, 91, 18],
    lab=[42.738994430457005, -35.661478599221795, 18.56031128404669],
    rgb=[26, 116, 68],
    hex="#1a7444",
)
Cossack_Green = Color(
    name="Cossack_Green",
    combinations=[5, 135, 262, 270, 278, 294, 319, 341, 348],
    swatch=2,
    cmyk=[76, 32, 91, 18],
    lab=[45.46425574120699, -27.202334630350194, 23.44747081712063],
    rgb=[67, 119, 66],
    hex="#437742",
)
Lincoln_Green = Color(
    name="Lincoln_Green",
    combinations=[70, 121, 203, 210, 280, 290],
    swatch=2,
    cmyk=[60, 48, 86, 37],
    lab=[36.266117341878385, -5.906614785992218, 21.182879377431902],
    rgb=[85, 88, 50],
    hex="#555832",
)
Blackish_Olive = Color(
    name="Blackish_Olive",
    combinations=[109, 318, 336],
    swatch=2,
    cmyk=[56, 32, 63, 55],
    lab=[33.261615930418856, -10.128404669260703, 9.688715953307394],
    rgb=[66, 83, 62],
    hex="#42533e",
)
Deep_Slate_Olive = Color(
    name="Deep_Slate_Olive",
    combinations=[189, 229, 268, 303, 310, 321, 332, 341, 342, 348],
    swatch=2,
    cmyk=[76, 60, 80, 62],
    lab=[18.800640878919662, -7.669260700389103, 7.536964980544752],
    rgb=[37, 49, 34],
    hex="#253122",
)
Nile_Blue = Color(
    name="Nile_Blue",
    combinations=[25, 250, 268, 302, 306, 330, 345],
    swatch=3,
    cmyk=[25, 0, 10, 0],
    lab=[87.6752880140383, -13.252918287937746, -5.031128404669261],
    rgb=[188, 228, 229],
    hex="#bce4e5",
)
Pale_King_s_Blue = Color(
    name="Pale_King_s_Blue",
    combinations=[16, 49, 72, 75, 167, 196, 213, 234, 287],
    swatch=3,
    cmyk=[33, 4, 7, 0],
    lab=[82.21408407721064, -12.735408560311285, -12.883268482490266],
    rgb=[167, 212, 228],
    hex="#a7d4e4",
)
Light_Glaucous_Blue = Color(
    name="Light_Glaucous_Blue",
    combinations=[54, 93, 119, 152, 178, 204, 227, 320, 339, 341],
    swatch=3,
    cmyk=[35, 10, 14, 0],
    lab=[78.20859082932783, -10.568093385214013, -8.723735408560316],
    rgb=[165, 200, 209],
    hex="#a5c8d1",
)
Salvia_Blue = Color(
    name="Salvia_Blue",
    combinations=[29, 129, 135, 139, 142, 188, 209, 212, 237, 272, 294, 321, 330],
    swatch=3,
    cmyk=[41, 25, 10, 0],
    lab=[69.4697489890898, -2.7587548638132233, -16.754863813229576],
    rgb=[151, 172, 200],
    hex="#97acc8",
)
Cobalt_Green = Color(
    name="Cobalt_Green",
    combinations=[156, 188, 201, 202, 230, 271, 281, 282, 290, 291, 308, 333],
    swatch=3,
    cmyk=[42, 0, 42, 0],
    lab=[79.01274128328375, -25.56420233463035, 13.0],
    rgb=[150, 209, 170],
    hex="#96d1aa",
)
Calamine_BLue = Color(
    name="Calamine_BLue",
    combinations=[20, 41, 65, 159, 176, 255, 261, 287, 291, 300],
    swatch=3,
    cmyk=[50, 0, 20, 0],
    lab=[77.09010452429999, -26.241245136186777, -10.373540856031127],
    rgb=[120, 205, 208],
    hex="#78cdd0",
)
Venice_Green = Color(
    name="Venice_Green",
    combinations=[78, 128, 138, 189, 283, 345],
    swatch=3,
    cmyk=[58, 0, 30, 0],
    lab=[73.69954985885404, -31.731517509727624, -6.140077821011673],
    rgb=[98, 198, 191],
    hex="#62c6bf",
)
Cerulian_Blue = Color(
    name="Cerulian_Blue",
    combinations=[1, 63, 99, 125, 148, 227, 240, 264],
    swatch=3,
    cmyk=[84, 26, 32, 0],
    lab=[54.972152285038526, -30.186770428015564, -20.13618677042801],
    rgb=[0, 147, 165],
    hex="#0093a5",
)
Peacock_Blue = Color(
    name="Peacock_Blue",
    combinations=[131, 286],
    swatch=3,
    cmyk=[100, 19, 43, 0],
    lab=[52.78553444724193, -48.91050583657588, -17.51750972762646],
    rgb=[0, 147, 155],
    hex="#00939b",
)
Green_Blue = Color(
    name="Green_Blue",
    combinations=[12, 74, 79, 178, 208, 252, 259, 271, 330],
    swatch=3,
    cmyk=[82, 24, 40, 3],
    lab=[54.134431982909895, -31.268482490272376, -12.883268482490266],
    rgb=[9, 145, 151],
    hex="#099197",
)
Olympic_Blue = Color(
    name="Olympic_Blue",
    combinations=[44, 67, 157, 194, 231, 274, 324],
    swatch=3,
    cmyk=[69, 44, 10, 0],
    lab=[53.090714885175856, -3.634241245136181, -30.739299610894946],
    rgb=[90, 130, 179],
    hex="#5a82b3",
)
Blue = Color(
    name="Blue",
    combinations=[49, 51, 88, 143, 154, 186, 191, 215, 257, 267, 295, 333],
    swatch=3,
    cmyk=[95, 54, 0, 0],
    lab=[43.49431601434348, -8.832684824902728, -48.77042801556421],
    rgb=[0, 110, 184],
    hex="#006eb8",
)
Antwarp_Blue = Color(
    name="Antwarp_Blue",
    combinations=[85, 106, 114, 140, 163, 172, 208, 244, 258, 281, 299, 302, 334],
    swatch=3,
    cmyk=[100, 40, 30, 10],
    lab=[42.03250171663996, -29.124513618677042, -27.727626459143963],
    rgb=[0, 113, 144],
    hex="#007190",
)
Helvetia_Blue = Color(
    name="Helvetia_Blue",
    combinations=[39, 48, 161, 187, 218, 259, 312, 347],
    swatch=3,
    cmyk=[100, 62, 19, 10],
    lab=[35.309376668955515, -12.101167315175104, -36.37354085603113],
    rgb=[0, 91, 141],
    hex="#005b8d",
)
Dark_Medici_Blue = Color(
    name="Dark_Medici_Blue",
    combinations=[160, 224, 241, 249],
    swatch=3,
    cmyk=[70, 45, 45, 15],
    lab=[45.117875944151976, -9.225680933852146, -6.856031128404666],
    rgb=[84, 112, 118],
    hex="#547076",
)
Dusky_Green = Color(
    name="Dusky_Green",
    combinations=[94, 219, 225, 278, 284, 318, 332, 338],
    swatch=3,
    cmyk=[100, 30, 64, 50],
    lab=[27.965209430075532, -36.268482490272376, -3.2801556420233453],
    rgb=[0, 79, 70],
    hex="#004f46",
)
Deep_Lyons_Blue = Color(
    name="Deep_Lyons_Blue",
    combinations=[22, 38, 101, 126, 179, 199, 236, 247, 314, 344],
    swatch=3,
    cmyk=[100, 85, 15, 6],
    lab=[28.532845044632637, 6.560311284046691, -42.68093385214007],
    rgb=[28, 66, 134],
    hex="#1c4286",
)
Violet_Blue = Color(
    name="Violet_Blue",
    combinations=[75, 83, 89, 98, 125, 233, 286, 289, 297, 309, 339],
    swatch=3,
    cmyk=[85, 79, 38, 16],
    lab=[30.14724956130312, 5.435797665369648, -22.377431906614788],
    rgb=[64, 69, 106],
    hex="#40456a",
)
Vandar_Poel_s_Blue = Color(
    name="Vandar_Poel_s_Blue",
    combinations=[5, 77, 151, 167, 168, 309, 343],
    swatch=3,
    cmyk=[100, 73, 43, 10],
    lab=[30.77744716563668, -11.264591439688715, -24.521400778210122],
    rgb=[6, 79, 110],
    hex="#064f6e",
)
Dark_Tyrian_Blue = Color(
    name="Dark_Tyrian_Blue",
    combinations=[2, 60, 67, 119, 141, 245, 279],
    swatch=3,
    cmyk=[90, 66, 36, 50],
    lab=[20.613412680247194, -5.836575875486375, -19.081712062256813],
    rgb=[18, 53, 78],
    hex="#12354e",
)
Dull_Violet_Black = Color(
    name="Dull_Violet_Black",
    combinations=[95, 106, 145, 265, 277, 289, 295, 331],
    swatch=3,
    cmyk=[95, 106, 38, 50],
    lab=[8.1025406271458, 19.431906614785987, -28.813229571984436],
    rgb=[30, 14, 63],
    hex="#1e0e3f",
)
Deep_Indigo = Color(
    name="Deep_Indigo",
    combinations=[6, 28, 139, 155, 182, 211, 232],
    swatch=3,
    cmyk=[100, 92, 52, 60],
    lab=[5.818265049210345, 4.140077821011687, -21.785992217898837],
    rgb=[5, 18, 48],
    hex="#051230",
)
Deep_Slate_Green = Color(
    name="Deep_Slate_Green",
    combinations=[84, 149, 166, 271, 318, 325],
    swatch=3,
    cmyk=[80, 50, 60, 70],
    lab=[16.881055924315252, -12.143968871595334, -2.2295719844357933],
    rgb=[17, 47, 44],
    hex="#112f2c",
)
Grayish_Lavender_A = Color(
    name="Grayish_Lavender_A",
    combinations=[8, 15, 159, 177, 218, 248, 307],
    swatch=4,
    cmyk=[28, 28, 0, 0],
    lab=[73.42336156252384, 7.459143968871587, -19.35019455252919],
    rgb=[181, 177, 216],
    hex="#b5b1d8",
)
Grayish_Lavender_B = Color(
    name="Grayish_Lavender_B",
    combinations=[47, 56, 174, 187, 235, 327, 329, 338],
    swatch=4,
    cmyk=[25, 33, 20, 0],
    lab=[71.34508278019379, 10.046692607003905, -2.210116731517516],
    rgb=[192, 169, 179],
    hex="#c0a9b3",
)
Laelia_Pink = Color(
    name="Laelia_Pink",
    combinations=[20, 254, 280, 337],
    swatch=4,
    cmyk=[20, 48, 18, 0],
    lab=[66.5186541542687, 24.307392996108945, -2.6614785992217946],
    rgb=[202, 146, 168],
    hex="#ca92a8",
)
Lilac = Color(
    name="Lilac",
    combinations=[143, 162, 282, 347],
    swatch=4,
    cmyk=[28, 54, 8, 0],
    lab=[61.530479896238646, 25.922178988326834, -14.252918287937746],
    rgb=[185, 132, 175],
    hex="#b984af",
)
Eupatorium_Purple = Color(
    name="Eupatorium_Purple",
    combinations=[215, 315, 322],
    swatch=4,
    cmyk=[25, 79, 12, 0],
    lab=[52.24536507209888, 46.71206225680933, -11.723735408560316],
    rgb=[191, 88, 146],
    hex="#bf5892",
)
Light_Mauve = Color(
    name="Light_Mauve",
    combinations=[23, 80, 128, 134, 180, 274, 331],
    swatch=4,
    cmyk=[43, 62, 5, 0],
    lab=[53.6415655756466, 23.902723735408557, -23.976653696498047],
    rgb=[154, 114, 170],
    hex="#9a72aa",
)
Aconite_Violet = Color(
    name="Aconite_Violet",
    combinations=[43, 64, 90, 187, 220, 257, 269, 301, 307, 324, 344],
    swatch=4,
    cmyk=[39, 68, 5, 0],
    lab=[52.66956588082704, 30.696498054474716, -22.369649805447466],
    rgb=[163, 106, 165],
    hex="#a36aa5",
)
Dull_Blue_Violet = Color(
    name="Dull_Blue_Violet",
    combinations=[9, 100],
    swatch=4,
    cmyk=[57, 60, 17, 0],
    lab=[50.22201876859693, 13.377431906614788, -22.4591439688716],
    rgb=[128, 113, 158],
    hex="#80719e",
)
Dark_Soft_Violet = Color(
    name="Dark_Soft_Violet",
    combinations=[64, 127, 197],
    swatch=4,
    cmyk=[70, 68, 13, 0],
    lab=[43.99633783474479, 12.346303501945528, -31.171206225680933],
    rgb=[102, 98, 156],
    hex="#66629c",
)
Blue_Violet = Color(
    name="Blue_Violet",
    combinations=[116, 175, 196, 322, 345],
    swatch=4,
    cmyk=[72, 80, 0, 0],
    lab=[39.12260624093995, 23.571984435797674, -41.91050583657588],
    rgb=[100, 80, 161],
    hex="#6450a1",
)
Purple_Drab = Color(
    name="Purple_Drab",
    combinations=[236],
    swatch=4,
    cmyk=[38, 65, 49, 26],
    lab=[41.87228198672465, 19.867704280155635, 5.560311284046691],
    rgb=[132, 86, 91],
    hex="#84565b",
)
Deep_Violet_Plumbeous = Color(
    name="Deep_Violet_Plumbeous",
    combinations=[183, 192, 218],
    swatch=4,
    cmyk=[61, 52, 43, 7],
    lab=[48.02929732204166, 0.7937743190661593, -5.906614785992218],
    rgb=[112, 114, 124],
    hex="#70727c",
)
Veronia_Purple = Color(
    name="Veronia_Purple",
    combinations=[13, 24, 168, 183],
    swatch=4,
    cmyk=[42, 78, 46, 15],
    lab=[40.982681010147246, 29.56031128404669, 0.12840466926070349],
    rgb=[140, 76, 98],
    hex="#8c4c62",
)
Dark_Slate_Purple = Color(
    name="Dark_Slate_Purple",
    combinations=[225, 248],
    swatch=4,
    cmyk=[64, 85, 60, 10],
    lab=[34.306858930342564, 22.217898832684824, -3.5525291828793826],
    rgb=[112, 67, 87],
    hex="#704357",
)
Taupe_Brown = Color(
    name="Taupe_Brown",
    combinations=[57, 123, 174, 224, 275, 280, 288],
    swatch=4,
    cmyk=[30, 70, 35, 40],
    lab=[36.17303730830854, 25.638132295719856, 0.06225680933852118],
    rgb=[122, 68, 86],
    hex="#7a4456",
)
Violet_Carmine = Color(
    name="Violet_Carmine",
    combinations=[337],
    swatch=4,
    cmyk=[64, 90, 70, 10],
    lab=[32.303349355306324, 25.867704280155635, 0.8171206225680976],
    rgb=[113, 59, 76],
    hex="#713b4c",
)
Violet = Color(
    name="Violet",
    combinations=[42, 56, 130, 156, 164, 181, 205, 214, 226, 316, 331, 335],
    swatch=4,
    cmyk=[85, 90, 18, 0],
    lab=[31.493095292591743, 20.665369649805456, -37.92607003891051],
    rgb=[79, 64, 134],
    hex="#4f4086",
)
Red_Violet = Color(
    name="Red_Violet",
    combinations=[4, 37, 134, 136, 170, 172, 183, 316],
    swatch=4,
    cmyk=[76, 100, 25, 15],
    lab=[25.052262149996185, 33.105058365758765, -30.05836575875486],
    rgb=[89, 37, 106],
    hex="#59256a",
)
Cotinga_Purple = Color(
    name="Cotinga_Purple",
    combinations=[61, 181, 238, 253, 307, 329, 348],
    swatch=4,
    cmyk=[66, 100, 42, 40],
    lab=[18.403906309605553, 32.85992217898831, -15.762645914396884],
    rgb=[80, 19, 69],
    hex="#501345",
)
Dusky_Madder_Violet = Color(
    name="Dusky_Madder_Violet",
    combinations=[18, 50, 53, 82, 103, 314],
    swatch=4,
    cmyk=[75, 100, 46, 30],
    lab=[20.103761348897535, 28.64980544747081, -18.256809338521407],
    rgb=[78, 29, 76],
    hex="#4e1d4c",
)
White = Color(
    name="White",
    combinations=[55],
    swatch=5,
    cmyk=[0, 0, 0, 0],
    lab=[100.0, 0.0, 0.0],
    rgb=[255, 255, 255],
    hex="#ffffff",
)
Neutral_Gray = Color(
    name="Neutral_Gray",
    combinations=[34, 139, 180, 195, 197, 221, 228, 229, 273, 303, 324, 340],
    swatch=5,
    cmyk=[29, 18, 20, 0],
    lab=[76.74525062943465, -3.0311284046692606, -2.0311284046692606],
    rgb=[182, 191, 193],
    hex="#b6bfc1",
)
Mineral_Gray = Color(
    name="Mineral_Gray",
    combinations=[11, 30],
    swatch=5,
    cmyk=[33, 18, 25, 7],
    lab=[70.75303273060197, -5.618677042801551, -0.0972762645914429],
    rgb=[162, 176, 173],
    hex="#a2b0ad",
)
Warm_Gray = Color(
    name="Warm_Gray",
    combinations=[69, 76, 81, 143, 169, 238, 259, 261],
    swatch=5,
    cmyk=[37, 28, 36, 3],
    lab=[66.74753948271916, -2.2451361867704236, 4.743190661478593],
    rgb=[161, 163, 154],
    hex="#a1a39a",
)
Slate_Color = Color(
    name="Slate_Color",
    combinations=[27, 33, 57, 140, 202, 243, 245, 251, 253, 263, 296, 329, 335],
    swatch=5,
    cmyk=[85, 70, 62, 30],
    lab=[27.84771496147097, -5.3929961089494185, -6.9727626459144005],
    rgb=[52, 69, 76],
    hex="#34454c",
)
Black = Color(
    name="Black",
    combinations=[
        46,
        52,
        62,
        69,
        112,
        117,
        144,
        190,
        207,
        216,
        221,
        242,
        255,
        256,
        269,
        276,
        288,
        298,
        313,
        323,
        337,
        340,
        344,
    ],
    swatch=5,
    cmyk=[20, 10, 15, 100],
    lab=[5.62752727550164, -0.3968871595330796, -1.1245136186770424],
    rgb=[17, 19, 20],
    hex="#111314",
)

swatches = {}
swatches[1] = [English_Red, Cerulian_Blue]
swatches[2] = [Yellow_Orange, Dark_Tyrian_Blue]
swatches[3] = [Raw_Sienna, Pale_Lemon_Yellow]
swatches[4] = [Isabella_Color, Red_Violet]
swatches[5] = [Cossack_Green, Vandar_Poel_s_Blue]
swatches[6] = [Grenadine_Pink, Deep_Indigo]
swatches[7] = [Orange, Glaucous_Green]
swatches[8] = [Cinnamon_Rufous, Grayish_Lavender_A]
swatches[9] = [Violet_Red, Dull_Blue_Violet]
swatches[10] = [Cinnamon_Rufous, Dark_Citrine]
swatches[11] = [Ivory_Buff, Mineral_Gray]
swatches[12] = [Isabella_Color, Green_Blue]
swatches[13] = [Raw_Sienna, Veronia_Purple]
swatches[14] = [Spinel_Red, Naples_Yellow]
swatches[15] = [Benzol_Green, Grayish_Lavender_A]
swatches[16] = [Vandyke_Red, Pale_King_s_Blue]
swatches[17] = [Eugenia_Red_B, Sea_Green]
swatches[18] = [Fawn, Dusky_Madder_Violet]
swatches[19] = [Mars_Brown_Tobacco, Night_Green]
swatches[20] = [Calamine_BLue, Laelia_Pink]
swatches[21] = [Grenadine_Pink, Sea_Green]
swatches[22] = [Yellow, Deep_Lyons_Blue]
swatches[23] = [Cinnamon_Buff, Light_Mauve]
swatches[24] = [Sepia, Veronia_Purple]
swatches[25] = [Etruscan_Red, Nile_Blue]
swatches[26] = [Golden_Yellow, Pale_Raw_Umber]
swatches[27] = [Corinthian_Pink, Slate_Color]
swatches[28] = [Madder_Brown, Deep_Indigo]
swatches[29] = [Krongbergs_Green, Salvia_Blue]
swatches[30] = [Pompeian_Red, Mineral_Gray]
swatches[31] = [Red_Orange, Pale_Lemon_Yellow]
swatches[32] = [Ochraceous_Salmon, Night_Green]
swatches[33] = [Raw_Sienna, Slate_Color]
swatches[34] = [Eosine_Pink, Neutral_Gray]
swatches[35] = [Light_Brown_Drab, Carmine_Red]
swatches[36] = [Sulphine_Yellow, Turquoise_Green]
swatches[37] = [Brick_Red, Red_Violet]
swatches[38] = [Diamine_Green, Deep_Lyons_Blue]
swatches[39] = [Carmine, Helvetia_Blue]
swatches[40] = [Vinaceous_Tawny, Citron_Yellow]
swatches[41] = [Dark_Citrine, Calamine_BLue]
swatches[42] = [Yellow_Ocher, Violet]
swatches[43] = [Corinthian_Pink, Aconite_Violet]
swatches[44] = [Light_Porcelain_Green, Olympic_Blue]
swatches[45] = [Seashell_Pink, Lemon_Yellow]
swatches[46] = [Orange, Black]
swatches[47] = [Etruscan_Red, Grayish_Lavender_B]
swatches[48] = [Rosolanc_Purple, Helvetia_Blue]
swatches[49] = [Pale_King_s_Blue, Blue]
swatches[50] = [Ivory_Buff, Dusky_Madder_Violet]
swatches[51] = [Carmine_Red, Blue]
swatches[52] = [Sulpher_Yellow, Black]
swatches[53] = [Yellow_Orange, Dusky_Madder_Violet]
swatches[54] = [Benzol_Green, Light_Glaucous_Blue]
swatches[55] = [Old_Rose, White]
swatches[56] = [Grayish_Lavender_B, Violet]
swatches[57] = [Taupe_Brown, Slate_Color]
swatches[58] = [Hay_s_Russet, Sea_Green]
swatches[59] = [Eosine_Pink, Citrine]
swatches[60] = [Pale_Lemon_Yellow, Dark_Tyrian_Blue]
swatches[61] = [Light_Green_Yellow, Cotinga_Purple]
swatches[62] = [Yellow, Black]
swatches[63] = [Vistoris_Lake, Cerulian_Blue]
swatches[64] = [Aconite_Violet, Dark_Soft_Violet]
swatches[65] = [Sulphine_Yellow, Calamine_BLue]
swatches[66] = [Olive_Ocher, Olive_Green]
swatches[67] = [Olympic_Blue, Dark_Tyrian_Blue]
swatches[68] = [Light_Brown_Drab, Yellow]
swatches[69] = [Warm_Gray, Black]
swatches[70] = [Raw_Sienna, Lincoln_Green]
swatches[71] = [Pompeian_Red, Ochraceous_Salmon]
swatches[72] = [Sulpher_Yellow, Pale_King_s_Blue]
swatches[73] = [Pale_Raw_Umber, Rainette_Green]
swatches[74] = [Turquoise_Green, Green_Blue]
swatches[75] = [Pale_King_s_Blue, Violet_Blue]
swatches[76] = [Pale_Lemon_Yellow, Warm_Gray]
swatches[77] = [Eugenia_Red_B, Vandar_Poel_s_Blue]
swatches[78] = [Pinkish_Cinnamon, Venice_Green]
swatches[79] = [Madder_Brown, Green_Blue]
swatches[80] = [Sulpher_Yellow, Light_Mauve]
swatches[81] = [Golden_Yellow, Warm_Gray]
swatches[82] = [Hay_s_Russet, Dusky_Madder_Violet]
swatches[83] = [Olive_Buff, Violet_Blue]
swatches[84] = [Seashell_Pink, Deep_Slate_Green]
swatches[85] = [Vinaceous_Tawny, Antwarp_Blue]
swatches[86] = [Raw_Sienna, Sea_Green]
swatches[87] = [Corinthian_Pink, Citron_Yellow]
swatches[88] = [Seashell_Pink, Blue]
swatches[89] = [Yellow_Orange, Violet_Blue]
swatches[90] = [Eosine_Pink, Aconite_Violet]
swatches[91] = [Vistoris_Lake, Orange_Rufous]
swatches[92] = [Coral_Red, Benzol_Green]
swatches[93] = [Citrine, Light_Glaucous_Blue]
swatches[94] = [Ivory_Buff, Dusky_Green]
swatches[95] = [Hay_s_Russet, Dull_Violet_Black]
swatches[96] = [Yellow_Ocher, Olive]
swatches[97] = [Corinthian_Pink, Etruscan_Red]
swatches[98] = [Madder_Brown, Violet_Blue]
swatches[99] = [Pale_Lemon_Yellow, Cerulian_Blue]
swatches[100] = [Buffy_Citrine, Dull_Blue_Violet]
swatches[101] = [Cameo_Pink, Deep_Lyons_Blue]
swatches[102] = [Ivory_Buff, Orange_Rufous]
swatches[103] = [Cinnamon_Rufous, Dusky_Madder_Violet]
swatches[104] = [Carmine_Red, Sulpher_Yellow]
swatches[105] = [Cameo_Pink, Chromium_Green]
swatches[106] = [Antwarp_Blue, Dull_Violet_Black]
swatches[107] = [Apricot_Yellow, Light_Grayish_Olive]
swatches[108] = [Eosine_Pink, Brick_Red]
swatches[109] = [Pale_Lemon_Yellow, Blackish_Olive]
swatches[110] = [Brown, Vandyke_Brown]
swatches[111] = [Pale_Lemon_Yellow, Yellow_Green]
swatches[112] = [Grenadine_Pink, Black]
swatches[113] = [Seashell_Pink, Vandyke_Brown]
swatches[114] = [Orange_Yellow, Antwarp_Blue]
swatches[115] = [Naples_Yellow, Peach_Red]
swatches[116] = [Cameo_Pink, Blue_Violet]
swatches[117] = [Carmine, Black]
swatches[118] = [Yellow_Ocher, Vandyke_Brown]
swatches[119] = [Light_Glaucous_Blue, Dark_Tyrian_Blue]
swatches[120] = [Cameo_Pink, Pompeian_Red]
swatches[121] = [Brown, Ochraceous_Salmon, Lincoln_Green]
swatches[122] = [Carmine, Cream_Yellow, Benzol_Green]
swatches[123] = [Coral_Red, Lemon_Yellow, Taupe_Brown]
swatches[124] = [Pale_Burnt_Lake, Yellow_Ocher, Olive_Yellow]
swatches[125] = [Fawn, Cerulian_Blue, Violet_Blue]
swatches[126] = [Ivory_Buff, Yellow_Ocher, Deep_Lyons_Blue]
swatches[127] = [Cinnamon_Buff, Pistachio_Green, Dark_Soft_Violet]
swatches[128] = [Corinthian_Pink, Venice_Green, Light_Mauve]
swatches[129] = [Apricot_Yellow, Khaki, Salvia_Blue]
swatches[130] = [Raw_Sienna, Carmine_Red, Violet]
swatches[131] = [Raw_Sienna, English_Red, Peacock_Blue]
swatches[132] = [Sulpher_Yellow, Golden_Yellow, Citrine]
swatches[133] = [Vandyke_Red, Citrine, Sea_Green]
swatches[134] = [Eosine_Pink, Light_Mauve, Red_Violet]
swatches[135] = [Sulpher_Yellow, Cossack_Green, Salvia_Blue]
swatches[136] = [Scarlet, Dull_Viridian_Green, Red_Violet]
swatches[137] = [Etruscan_Red, Cinnamon_Buff, Pistachio_Green]
swatches[138] = [Golden_Yellow, Lemon_Yellow, Venice_Green]
swatches[139] = [Salvia_Blue, Deep_Indigo, Neutral_Gray]
swatches[140] = [Golden_Yellow, Antwarp_Blue, Slate_Color]
swatches[141] = [Orange, Yellow_Green, Dark_Tyrian_Blue]
swatches[142] = [Hydrangea_Red, Sulphine_Yellow, Salvia_Blue]
swatches[143] = [Blue, Lilac, Warm_Gray]
swatches[144] = [Rosolanc_Purple, Orange, Black]
swatches[145] = [Brown, Citron_Yellow, Dull_Violet_Black]
swatches[146] = [Khaki, Deep_Grayish_Olive, Diamine_Green]
swatches[147] = [Spinel_Red, Vandyke_Red, Turquoise_Green]
swatches[148] = [Olive_Ocher, Orange_Yellow, Cerulian_Blue]
swatches[149] = [Olive_Ocher, Orange, Deep_Slate_Green]
swatches[150] = [Seashell_Pink, Citron_Yellow, Glaucous_Green]
swatches[151] = [Sulpher_Yellow, Yellow_Orange, Vandar_Poel_s_Blue]
swatches[152] = [Etruscan_Red, Hay_s_Russet, Light_Glaucous_Blue]
swatches[153] = [Eosine_Pink, Orange_Yellow, Citron_Yellow]
swatches[154] = [Carmine, Yellow, Blue]
swatches[155] = [Jasper_Red, Benzol_Green, Deep_Indigo]
swatches[156] = [Olive_Ocher, Cobalt_Green, Violet]
swatches[157] = [Pansy_Purple, Olive_Ocher, Olympic_Blue]
swatches[158] = [Lemon_Yellow, Cinnamon_Rufous, Night_Green]
swatches[159] = [Khaki, Calamine_BLue, Grayish_Lavender_A]
swatches[160] = [Sulphine_Yellow, Pale_Raw_Umber, Dark_Medici_Blue]
swatches[161] = [Brown, Pinkish_Cinnamon, Helvetia_Blue]
swatches[162] = [Old_Rose, Rainette_Green, Lilac]
swatches[163] = [Apricot_Yellow, Turquoise_Green, Antwarp_Blue]
swatches[164] = [Red_Orange, Orange_Yellow, Violet]
swatches[165] = [Cameo_Pink, Spinel_Red, Vistoris_Lake]
swatches[166] = [Grenadine_Pink, Naples_Yellow, Deep_Slate_Green]
swatches[167] = [Ecru, Pale_King_s_Blue, Vandar_Poel_s_Blue]
swatches[168] = [Lemon_Yellow, Vandar_Poel_s_Blue, Veronia_Purple]
swatches[169] = [Corinthian_Pink, Pale_Lemon_Yellow, Warm_Gray]
swatches[170] = [Rosolanc_Purple, Orange_Yellow, Red_Violet]
swatches[171] = [Pale_Burnt_Lake, Yellow_Orange, Glaucous_Green]
swatches[172] = [Cinnamon_Rufous, Antwarp_Blue, Red_Violet]
swatches[173] = [Lemon_Yellow, Madder_Brown, Turquoise_Green]
swatches[174] = [Corinthian_Pink, Grayish_Lavender_B, Taupe_Brown]
swatches[175] = [Pinkish_Cinnamon, Olive_Buff, Blue_Violet]
swatches[176] = [Hermosa_Pink, Seashell_Pink, Calamine_BLue]
swatches[177] = [Pale_Burnt_Lake, Buffy_Citrine, Grayish_Lavender_A]
swatches[178] = [Ivory_Buff, Light_Glaucous_Blue, Green_Blue]
swatches[179] = [Red_Orange, Golden_Yellow, Deep_Lyons_Blue]
swatches[180] = [Cinnamon_Buff, Light_Mauve, Neutral_Gray]
swatches[181] = [Carmine_Red, Violet, Cotinga_Purple]
swatches[182] = [Raw_Sienna, Vandyke_Brown, Deep_Indigo]
swatches[183] = [Deep_Violet_Plumbeous, Veronia_Purple, Red_Violet]
swatches[184] = [Spinel_Red, Ivory_Buff, Light_Grayish_Olive]
swatches[185] = [Light_Brown_Drab, Etruscan_Red, Pale_Lemon_Yellow]
swatches[186] = [Hay_s_Russet, Ochraceous_Salmon, Blue]
swatches[187] = [Helvetia_Blue, Grayish_Lavender_B, Aconite_Violet]
swatches[188] = [Rainette_Green, Salvia_Blue, Cobalt_Green]
swatches[189] = [Lemon_Yellow, Deep_Slate_Olive, Venice_Green]
swatches[190] = [Ivory_Buff, English_Red, Black]
swatches[191] = [Light_Brown_Drab, Yellow_Ocher, Blue]
swatches[192] = [Cream_Yellow, Vandyke_Brown, Deep_Violet_Plumbeous]
swatches[193] = [Grenadine_Pink, Naples_Yellow, Light_Porcelain_Green]
swatches[194] = [Jasper_Red, Seashell_Pink, Olympic_Blue]
swatches[195] = [Spinel_Red, Pale_Lemon_Yellow, Neutral_Gray]
swatches[196] = [Citron_Yellow, Pale_King_s_Blue, Blue_Violet]
swatches[197] = [Eosine_Pink, Dark_Soft_Violet, Neutral_Gray]
swatches[198] = [Burnt_Sienna, Apricot_Yellow, Green]
swatches[199] = [Ochre_Red, Light_Brownish_Olive, Deep_Lyons_Blue]
swatches[200] = [Carmine_Red, Olive_Buff, Chromium_Green]
swatches[201] = [Grenadine_Pink, Olive, Cobalt_Green]
swatches[202] = [Turquoise_Green, Cobalt_Green, Slate_Color]
swatches[203] = [Pale_Lemon_Yellow, Vinaceous_Cinnamon, Lincoln_Green]
swatches[204] = [Rosolanc_Purple, Cinnamon_Rufous, Light_Glaucous_Blue]
swatches[205] = [Pale_Burnt_Lake, Vinaceous_Cinnamon, Violet]
swatches[206] = [Corinthian_Pink, Golden_Yellow, Cinnamon_Rufous]
swatches[207] = [Sudan_Brown, Glaucous_Green, Black]
swatches[208] = [Sulpher_Yellow, Green_Blue, Antwarp_Blue]
swatches[209] = [Ivory_Buff, Yellow_Orange, Salvia_Blue]
swatches[210] = [Cinnamon_Buff, Lemon_Yellow, Lincoln_Green]
swatches[211] = [Apricot_Orange, Olive_Yellow, Deep_Indigo]
swatches[212] = [Pompeian_Red, Orange_Citrine, Salvia_Blue]
swatches[213] = [Vinaceous_Cinnamon, Apricot_Yellow, Pale_King_s_Blue]
swatches[214] = [Ivory_Buff, Sudan_Brown, Violet]
swatches[215] = [Cream_Yellow, Blue, Eupatorium_Purple]
swatches[216] = [Jasper_Red, Green, Black]
swatches[217] = [Pale_Burnt_Lake, Ochraceous_Salmon, Diamine_Green]
swatches[218] = [Helvetia_Blue, Grayish_Lavender_A, Deep_Violet_Plumbeous]
swatches[219] = [Jasper_Red, Chromium_Green, Dusky_Green]
swatches[220] = [Pomegranite_Purple, Ochraceous_Salmon, Aconite_Violet]
swatches[221] = [Carmine_Red, Neutral_Gray, Black]
swatches[222] = [Yellow_Ocher, Yellow_Orange, Orange_Rufous]
swatches[223] = [Light_Brown_Drab, Ochraceous_Salmon, Turquoise_Green]
swatches[224] = [Spinel_Red, Dark_Medici_Blue, Taupe_Brown]
swatches[225] = [Carmine, Dusky_Green, Dark_Slate_Purple]
swatches[226] = [Vistoris_Lake, Cream_Yellow, Violet]
swatches[227] = [Hermosa_Pink, Light_Glaucous_Blue, Cerulian_Blue]
swatches[228] = [Carmine_Red, Pale_Lemon_Yellow, Neutral_Gray]
swatches[229] = [Golden_Yellow, Deep_Slate_Olive, Neutral_Gray]
swatches[230] = [Grenadine_Pink, Turquoise_Green, Cobalt_Green]
swatches[231] = [Cameo_Pink, Hay_s_Russet, Olympic_Blue]
swatches[232] = [Carmine, Pinkish_Cinnamon, Deep_Indigo]
swatches[233] = [Carmine_Red, Buffy_Citrine, Violet_Blue]
swatches[234] = [Cinnamon_Buff, Pale_Raw_Umber, Pale_King_s_Blue]
swatches[235] = [Ivory_Buff, Yellow_Orange, Grayish_Lavender_B]
swatches[236] = [Khaki, Deep_Lyons_Blue, Purple_Drab]
swatches[237] = [Carmine_Red, Madder_Brown, Salvia_Blue]
swatches[238] = [Ochraceous_Salmon, Cotinga_Purple, Warm_Gray]
swatches[239] = [Light_Brown_Drab, Pyrite_Yellow, Glaucous_Green]
swatches[240] = [Fresh_Color, Yellow, Cerulian_Blue]
swatches[241] = [Red_Orange, Pale_Lemon_Yellow, Isabella_Color, Dark_Medici_Blue]
swatches[242] = [Eosine_Pink, Burnt_Sienna, Diamine_Green, Black]
swatches[243] = [Raw_Sienna, Ivory_Buff, Olive_Green, Slate_Color]
swatches[244] = [Light_Brown_Drab, Vinaceous_Tawny, Andover_Green, Antwarp_Blue]
swatches[245] = [Carmine_Red, Oil_Green, Dark_Tyrian_Blue, Slate_Color]
swatches[246] = [Corinthian_Pink, Brick_Red, Sulpher_Yellow, Cinnamon_Buff]
swatches[247] = [Raw_Sienna, Apricot_Yellow, Benzol_Green, Deep_Lyons_Blue]
swatches[248] = [Eosine_Pink, Khaki, Grayish_Lavender_A, Dark_Slate_Purple]
swatches[249] = [Hay_s_Russet, Ecru, Olive_Ocher, Dark_Medici_Blue]
swatches[250] = [Pyrite_Yellow, Peach_Red, Sea_Green, Nile_Blue]
swatches[251] = [Red, Yellow, Diamine_Green, Slate_Color]
swatches[252] = [Eugenia_Red_B, Raw_Sienna, Sulphine_Yellow, Green_Blue]
swatches[253] = [Lemon_Yellow, Apricot_Orange, Cotinga_Purple, Slate_Color]
swatches[254] = [Corinthian_Pink, Sulpher_Yellow, Olive, Laelia_Pink]
swatches[255] = [Raw_Sienna, Pyrite_Yellow, Calamine_BLue, Black]
swatches[256] = [Vinaceous_Cinnamon, Orange, Dull_Viridian_Green, Black]
swatches[257] = [Spectrum_Red, Orange_Yellow, Blue, Aconite_Violet]
swatches[258] = [Pale_Burnt_Lake, Pinkish_Cinnamon, Olive, Antwarp_Blue]
swatches[259] = [Lemon_Yellow, Green_Blue, Helvetia_Blue, Warm_Gray]
swatches[260] = [Old_Rose, Vinaceous_Cinnamon, Glaucous_Green, Sea_Green]
swatches[261] = [Red, Pale_Lemon_Yellow, Calamine_BLue, Warm_Gray]
swatches[262] = [Eugenia_Red_B, Ivory_Buff, Citrine, Cossack_Green]
swatches[263] = [Burnt_Sienna, Pinkish_Cinnamon, Turquoise_Green, Slate_Color]
swatches[264] = [Corinthian_Pink, Red_Orange, Dark_Greenish_Glaucous, Cerulian_Blue]
swatches[265] = [Old_Rose, Apricot_Yellow, Olive_Yellow, Dull_Violet_Black]
swatches[266] = [Spectrum_Red, Ivory_Buff, Rainette_Green, Benzol_Green]
swatches[267] = [Cream_Yellow, Yellow_Orange, Benzol_Green, Blue]
swatches[268] = [Light_Brown_Drab, Raw_Sienna, Deep_Slate_Olive, Nile_Blue]
swatches[269] = [Raw_Sienna, Pale_Burnt_Lake, Aconite_Violet, Black]
swatches[270] = [Eugenia_Red_B, Sulpher_Yellow, Olive_Green, Cossack_Green]
swatches[271] = [Pomegranite_Purple, Cobalt_Green, Green_Blue, Deep_Slate_Green]
swatches[272] = [Pale_Lemon_Yellow, Orange, Turquoise_Green, Salvia_Blue]
swatches[273] = [Hermosa_Pink, Pansy_Purple, Sudan_Brown, Neutral_Gray]
swatches[274] = [Peach_Red, Dark_Citrine, Olympic_Blue, Light_Mauve]
swatches[275] = [Etruscan_Red, Ecru, Madder_Brown, Taupe_Brown]
swatches[276] = [Eosine_Pink, Seashell_Pink, Yellow_Green, Black]
swatches[277] = [Spinel_Red, Rosolanc_Purple, Olive, Dull_Violet_Black]
swatches[278] = [Cream_Yellow, Olive_Ocher, Cossack_Green, Dusky_Green]
swatches[279] = [Raw_Sienna, Vinaceous_Cinnamon, Ecru, Dark_Tyrian_Blue]
swatches[280] = [Eugenia_Red_B, Lincoln_Green, Laelia_Pink, Taupe_Brown]
swatches[281] = [Pale_Lemon_Yellow, Benzol_Green, Cobalt_Green, Antwarp_Blue]
swatches[282] = [Eugenia_Red_B, Maple, Cobalt_Green, Lilac]
swatches[283] = [Ochre_Red, Pale_Burnt_Lake, Chromium_Green, Venice_Green]
swatches[284] = [Eugenia_Red_A, Apricot_Yellow, Sea_Green, Dusky_Green]
swatches[285] = [Light_Brown_Drab, Burnt_Sienna, Peach_Red, Turquoise_Green]
swatches[286] = [Burnt_Sienna, Orange_Yellow, Peacock_Blue, Violet_Blue]
swatches[287] = [Eosine_Pink, Pyrite_Yellow, Pale_King_s_Blue, Calamine_BLue]
swatches[288] = [Yellow_Orange, Sepia, Taupe_Brown, Black]
swatches[289] = [Lemon_Yellow, Light_Green_Yellow, Violet_Blue, Dull_Violet_Black]
swatches[290] = [Vistoris_Lake, Pale_Lemon_Yellow, Lincoln_Green, Cobalt_Green]
swatches[291] = [Light_Green_Yellow, Sea_Green, Cobalt_Green, Calamine_BLue]
swatches[292] = [Pale_Lemon_Yellow, Pinkish_Cinnamon, Isabella_Color, Ecru]
swatches[293] = [Raw_Sienna, Turquoise_Green, Artemesia_Green, Green]
swatches[294] = [Sulpher_Yellow, Cream_Yellow, Cossack_Green, Salvia_Blue]
swatches[295] = [Cream_Yellow, Yellow, Blue, Dull_Violet_Black]
swatches[296] = [Sulpher_Yellow, Ochraceous_Salmon, Pale_Raw_Umber, Slate_Color]
swatches[297] = [Burnt_Sienna, Yellow_Orange, Olive_Green, Violet_Blue]
swatches[298] = [Raw_Sienna, Lemon_Yellow, Peach_Red, Black]
swatches[299] = [Indian_Lake, Vinaceous_Cinnamon, Oil_Green, Antwarp_Blue]
swatches[300] = [Grenadine_Pink, Cream_Yellow, Turquoise_Green, Calamine_BLue]
swatches[301] = [Spectrum_Red, Ivory_Buff, Rainette_Green, Aconite_Violet]
swatches[302] = [Cream_Yellow, Ecru, Nile_Blue, Antwarp_Blue]
swatches[303] = [Naples_Yellow, Peach_Red, Deep_Slate_Olive, Neutral_Gray]
swatches[304] = [Hay_s_Russet, Cream_Yellow, Dark_Citrine, Benzol_Green]
swatches[305] = [Pinkish_Cinnamon, Apricot_Yellow, Citron_Yellow, Turquoise_Green]
swatches[306] = [Lemon_Yellow, Benzol_Green, Dull_Viridian_Green, Nile_Blue]
swatches[307] = [Carmine, Grayish_Lavender_A, Aconite_Violet, Cotinga_Purple]
swatches[308] = [Fawn, Scarlet, English_Red, Cobalt_Green]
swatches[309] = [Golden_Yellow, Apricot_Orange, Violet_Blue, Vandar_Poel_s_Blue]
swatches[310] = [Sulpher_Yellow, Pinkish_Cinnamon, Olive, Deep_Slate_Olive]
swatches[311] = [Pompeian_Red, Cream_Yellow, Dark_Greenish_Glaucous, Light_Green_Yellow]
swatches[312] = [Burnt_Sienna, Yellow_Orange, Artemesia_Green, Helvetia_Blue]
swatches[313] = [Carmine, Yellow, Diamine_Green, Black]
swatches[314] = [Eosine_Pink, Hay_s_Russet, Deep_Lyons_Blue, Dusky_Madder_Violet]
swatches[315] = [Grenadine_Pink, Sulpher_Yellow, Golden_Yellow, Eupatorium_Purple]
swatches[316] = [Vandyke_Red, Dull_Viridian_Green, Violet, Red_Violet]
swatches[317] = [Light_Pinkish_Cinnamon, Ecru, Lemon_Yellow, Turquoise_Green]
swatches[318] = [Light_Brownish_Olive, Blackish_Olive, Dusky_Green, Deep_Slate_Green]
swatches[319] = [Raw_Sienna, Apricot_Yellow, Yellow_Orange, Cossack_Green]
swatches[320] = [Coral_Red, Sulpher_Yellow, Oil_Green, Light_Glaucous_Blue]
swatches[321] = [Light_Brown_Drab, Sulpher_Yellow, Deep_Slate_Olive, Salvia_Blue]
swatches[322] = [Spectrum_Red, Brick_Red, Eupatorium_Purple, Blue_Violet]
swatches[323] = [Cinnamon_Buff, Citron_Yellow, Madder_Brown, Black]
swatches[324] = [Pompeian_Red, Olympic_Blue, Aconite_Violet, Neutral_Gray]
swatches[325] = [Eugenia_Red_B, Naples_Yellow, Yellow_Ocher, Deep_Slate_Green]
swatches[326] = [Sulpher_Yellow, Peach_Red, Yellow_Green, Night_Green]
swatches[327] = [Eosine_Pink, Raw_Sienna, Ecru, Grayish_Lavender_B]
swatches[328] = [Brick_Red, Apricot_Orange, Vandyke_Brown, Light_Porcelain_Green]
swatches[329] = [Cream_Yellow, Grayish_Lavender_B, Cotinga_Purple, Slate_Color]
swatches[330] = [Olive_Buff, Nile_Blue, Salvia_Blue, Green_Blue]
swatches[331] = [Indian_Lake, Dull_Violet_Black, Light_Mauve, Violet]
swatches[332] = [Coral_Red, Scarlet, Deep_Slate_Olive, Dusky_Green]
swatches[333] = [Burnt_Sienna, Lemon_Yellow, Cobalt_Green, Blue]
swatches[334] = [Seashell_Pink, Olive, Yellow_Green, Antwarp_Blue]
swatches[335] = [Vandyke_Red, Yellow_Orange, Violet, Slate_Color]
swatches[336] = [Eosine_Pink, Hay_s_Russet, Pale_Lemon_Yellow, Blackish_Olive]
swatches[337] = [Vistoris_Lake, Laelia_Pink, Violet_Carmine, Black]
swatches[338] = [Carmine_Red, Orange_Yellow, Dusky_Green, Grayish_Lavender_B]
swatches[339] = [Ochraceous_Salmon, English_Red, Light_Glaucous_Blue, Violet_Blue]
swatches[340] = [Peach_Red, Sea_Green, Neutral_Gray, Black]
swatches[341] = [Grenadine_Pink, Cossack_Green, Deep_Slate_Olive, Light_Glaucous_Blue]
swatches[342] = [Corinthian_Pink, Cream_Yellow, Orange_Citrine, Deep_Slate_Olive]
swatches[343] = [Burnt_Sienna, Ivory_Buff, Deep_Grayish_Olive, Vandar_Poel_s_Blue]
swatches[344] = [Cinnamon_Buff, Deep_Lyons_Blue, Aconite_Violet, Black]
swatches[345] = [Hay_s_Russet, Nile_Blue, Venice_Green, Blue_Violet]
swatches[346] = [Rosolanc_Purple, Turquoise_Green, Light_Green_Yellow, Andover_Green]
swatches[347] = [Olive_Yellow, Sea_Green, Helvetia_Blue, Lilac]
swatches[348] = [Olive_Buff, Cossack_Green, Deep_Slate_Olive, Cotinga_Purple]

swatches2 = {}
swatches3 = {}
swatches4 = {}
colored = [swatches2, swatches3, swatches4]

for i in range(3):
    for k, v in swatches.items():
        if len(v) == i + 2:
            colored[i][k] = [[x / 256 for x in color_.rgb] for color_ in v]


def randomSwatch2(nColors=4):
    """Return a random Sanzo Wada swatch with ``nColors`` colors.

    Args:
        nColors: Number of colors in the swatch (2–4 inclusive).

    Returns:
        Selected swatch entry from the ``colored`` tables.

    Raises:
        ValueError: If ``nColors`` is outside 2–4.
    """
    if 1 < nColors < 5:
        randSwatchIndexes = colored[nColors - 2].keys()
    else:
        raise ValueError("nColors must be between 2 and 4 inclusive")
    keys = list(randSwatchIndexes)
    ind = choice(keys)
    swatch = colored[nColors - 2][ind]

    return ind, swatch
