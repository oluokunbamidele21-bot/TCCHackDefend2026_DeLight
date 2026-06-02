
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "cassava_best_model_int8.tflite"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

# ══════════════════════════════════════════════════════════════
# LOCALIZATION — English, French, Swahili, Yoruba, Hausa
# ══════════════════════════════════════════════════════════════
LANGUAGES = {
    "🇬🇧 English"  : "en",
    "🇫🇷 Français" : "fr",
    "🇰🇪 Swahili"  : "sw",
    "🇳🇬 Yorùbá"   : "yo",
    "🇳🇬 Hausa"    : "ha",
    "🇹🇬 Eʋegbe"   : "ee",   
    "🇹🇬 Kabiyè"   : "kbp",  
}

UI_TEXT = {
    "en": {
        "title"         : "Cassava Disease Detector",
        "subtitle"      : "AI-powered cassava leaf disease detection helping African smallholder farmers protect their crops and livelihoods.",
        "upload_label"  : "📸 Upload or Drop Cassava Leaf Image",
        "detect_btn"    : "🔍  Analyse Leaf",
        "tips_title"    : "📷 Photo Tips",
        "tips"          : ["Use natural daylight", "Focus on a single leaf", "Fill the frame with the leaf", "Keep the camera steady"],
        "result_title"  : "Analysis Result",
        "confidence"    : "Confidence",
        "severity"      : "Severity",
        "symptoms"      : "Symptoms",
        "cause"         : "Cause",
        "spreads"       : "How It Spreads",
        "treatment"     : "Treatment Steps",
        "prevention"    : "Prevention",
        "breakdown"     : "Confidence Breakdown",
        "waiting"       : "Upload a cassava leaf image and click Analyse Leaf",
        "how_title"     : "How It Works",
        "step1"         : "Upload",
        "step1_desc"    : "Take or upload a clear cassava leaf photo",
        "step2"         : "Analyse",
        "step2_desc"    : "MobileNetV2 AI scans the leaf pattern",
        "step3"         : "Detect",
        "step3_desc"    : "Disease identified across 5 classes",
        "step4"         : "Treat",
        "step4_desc"    : "Get immediate actionable advice",
        "stats_scans"   : "Leaves Analysed",
        "stats_acc"     : "Model Accuracy",
        "stats_classes" : "Disease Classes",
        "stats_size"    : "Model Size",
        "footer"        : "Built by Delight Entreprises · Empowering African Agriculture with AI 🌍",
        "retake"        : "Please Retake Your Photo",
        "select_lang"   : "Language / Langue",
        "severity_none" : "HEALTHY",
        "severity_mod"  : "MODERATE",
        "severity_high" : "HIGH",
        "severity_crit" : "CRITICAL",
    },
    "fr": {
        "title"         : "Détecteur de Maladies du Manioc",
        "subtitle"      : "Détection des maladies des feuilles de manioc par IA pour aider les petits agriculteurs africains à protéger leurs récoltes.",
        "upload_label"  : "📸 Télécharger une image de feuille de manioc",
        "detect_btn"    : "🔍  Analyser la feuille",
        "tips_title"    : "📷 Conseils photo",
        "tips"          : ["Utilisez la lumière naturelle", "Focalisez sur une seule feuille", "Remplissez le cadre avec la feuille", "Gardez l'appareil stable"],
        "result_title"  : "Résultat d'Analyse",
        "confidence"    : "Confiance",
        "severity"      : "Gravité",
        "symptoms"      : "Symptômes",
        "cause"         : "Cause",
        "spreads"       : "Mode de Propagation",
        "treatment"     : "Étapes de Traitement",
        "prevention"    : "Prévention",
        "breakdown"     : "Répartition des Scores",
        "waiting"       : "Téléchargez une image et cliquez sur Analyser la feuille",
        "how_title"     : "Comment ça marche",
        "step1"         : "Télécharger",
        "step1_desc"    : "Prenez ou uploadez une photo claire",
        "step2"         : "Analyser",
        "step2_desc"    : "L'IA MobileNetV2 scanne la feuille",
        "step3"         : "Détecter",
        "step3_desc"    : "Maladie identifiée parmi 5 classes",
        "step4"         : "Traiter",
        "step4_desc"    : "Obtenez des conseils immédiats",
        "stats_scans"   : "Feuilles Analysées",
        "stats_acc"     : "Précision du Modèle",
        "stats_classes" : "Classes de Maladies",
        "stats_size"    : "Taille du Modèle",
        "footer"        : "Créé par Delight Entreprises · L'IA au service de l'Agriculture Africaine 🌍",
        "retake"        : "Veuillez Reprendre la Photo",
        "select_lang"   : "Language / Langue",
        "severity_none" : "SAIN",
        "severity_mod"  : "MODÉRÉ",
        "severity_high" : "ÉLEVÉ",
        "severity_crit" : "CRITIQUE",
    },
    "sw": {
        "title"         : "Kigundua Magonjwa ya Muhogo",
        "subtitle"      : "Kugundua magonjwa ya muhogo kwa AI kusaidia wakulima wadogo wa Afrika kulinda mazao yao.",
        "upload_label"  : "📸 Pakia picha ya jani la muhogo",
        "detect_btn"    : "🔍  Chunguza Jani",
        "tips_title"    : "📷 Vidokezo vya Picha",
        "tips"          : ["Tumia mwanga wa asili", "Lenga jani moja", "Jaza fremu na jani", "Shika kamera imara"],
        "result_title"  : "Matokeo ya Uchambuzi",
        "confidence"    : "Uhakika",
        "severity"      : "Ukali",
        "symptoms"      : "Dalili",
        "cause"         : "Sababu",
        "spreads"       : "Jinsi Inavyoenea",
        "treatment"     : "Hatua za Matibabu",
        "prevention"    : "Kinga",
        "breakdown"     : "Mgawanyo wa Alama",
        "waiting"       : "Pakia picha ya jani la muhogo kisha bonyeza Chunguza Jani",
        "how_title"     : "Jinsi Inavyofanya Kazi",
        "step1"         : "Pakia",
        "step1_desc"    : "Piga au pakia picha wazi ya jani",
        "step2"         : "Chunguza",
        "step2_desc"    : "AI ya MobileNetV2 huchunguza jani",
        "step3"         : "Gundua",
        "step3_desc"    : "Ugonjwa kutambuliwa kati ya madarasa 5",
        "step4"         : "Tibu",
        "step4_desc"    : "Pata ushauri wa matibabu",
        "stats_scans"   : "Majani Yaliyochunguzwa",
        "stats_acc"     : "Usahihi wa Modeli",
        "stats_classes" : "Madarasa ya Magonjwa",
        "stats_size"    : "Ukubwa wa Modeli",
        "footer"        : "Imeundwa na Delight Entreprises · Kuwezesha Kilimo cha Afrika kwa AI 🌍",
        "retake"        : "Tafadhali Piga Picha Tena",
        "select_lang"   : "Language / Lugha",
        "severity_none" : "AFYA NZURI",
        "severity_mod"  : "WASTANI",
        "severity_high" : "JUU",
        "severity_crit" : "HATARI",
    },
    "yo": {
        "title"         : "Ẹ̀rọ Ìdámọ̀ Àrùn Igi Pàkì",
        "subtitle"      : "Ìdámọ̀ àrùn ewé igi pàkì nípa AI láti ran àwọn àgbẹ̀ kéékèèké ní Áfríkà lọ́wọ́.",
        "upload_label"  : "📸 Gbé fọ́tò ewé igi pàkì sí ẹ̀rọ",
        "detect_btn"    : "🔍  Ṣàyẹ̀wò Ewé",
        "tips_title"    : "📷 Àwọn Ìmọ̀ràn Fọ́tò",
        "tips"          : ["Lo ìmọ́lẹ̀ adáyeba", "Fojúsí ewé kan ṣoṣo", "Jẹ́ kí ewé kún àpótí", "Di kámẹ̀rà dúró"],
        "result_title"  : "Ìjábọ̀ Ìwádìí",
        "confidence"    : "Ìgbẹ́kẹ̀lé",
        "severity"      : "Ìpele Àrùn",
        "symptoms"      : "Àmì Àrùn",
        "cause"         : "Ohun tó fà á",
        "spreads"       : "Bí ó Ṣe Ń Tàn",
        "treatment"     : "Àwọn Ìgbésẹ̀ Ìtọ́jú",
        "prevention"    : "Àbójútó",
        "breakdown"     : "Ìpínpín Àwọn Ìgbẹ́kẹ̀lé",
        "waiting"       : "Gbé fọ́tò ewé sí ẹ̀rọ kí o sì tẹ Ṣàyẹ̀wò Ewé",
        "how_title"     : "Bí Ó Ṣe Ń Ṣiṣẹ́",
        "step1"         : "Gbé Sí",
        "step1_desc"    : "Ya tàbí gbé fọ́tò ewé tó gbọ́",
        "step2"         : "Ṣàyẹ̀wò",
        "step2_desc"    : "AI MobileNetV2 ń ṣàyẹ̀wò ewé",
        "step3"         : "Dámọ̀",
        "step3_desc"    : "Àrùn tí a dámọ̀ lára àwọn 5",
        "step4"         : "Tọ́jú",
        "step4_desc"    : "Gba ìmọ̀ràn ìtọ́jú",
        "stats_scans"   : "Àwọn Ewé tí a Ṣàyẹ̀wò",
        "stats_acc"     : "Déédéé Àwòkọ́ṣe",
        "stats_classes" : "Àwọn Ìpele Àrùn",
        "stats_size"    : "Ìwọ̀n Àwòkọ́ṣe",
        "footer"        : "Dá sílẹ̀ láti ọwọ́ Delight Entreprises · AI fún Iṣẹ́ Àgbẹ̀ Áfríkà 🌍",
        "retake"        : "Jọ̀wọ́ Ya Fọ́tò Mìíràn",
        "select_lang"   : "Language / Èdè",
        "severity_none" : "ILERA",
        "severity_mod"  : "ÌGBÀÁRỌ̀",
        "severity_high" : "GÍGA",
        "severity_crit" : "EWÚU",
    },
    "ha": {
        "title"         : "Mai Gano Cututtukan Rogo",
        "subtitle"      : "Gano cututtukan ganyen rogo ta hanyar AI don taimaka wa manoman Afrika karami su kare gonakinsu.",
        "upload_label"  : "📸 Loda hoto na ganyen rogo",
        "detect_btn"    : "🔍  Bincika Ganye",
        "tips_title"    : "📷 Shawarwarin Hoto",
        "tips"          : ["Yi amfani da haske na yanayi", "Mayar da hankali ga ganye ɗaya", "Cika firam da ganye", "Riƙe kyamarar da kwanciyar hankali"],
        "result_title"  : "Sakamakon Bincike",
        "confidence"    : "Tabbaci",
        "severity"      : "Tsanani",
        "symptoms"      : "Alamomi",
        "cause"         : "Dalilin",
        "spreads"       : "Yadda Yake Yaɗuwa",
        "treatment"     : "Matakai na Magani",
        "prevention"    : "Kariya",
        "breakdown"     : "Rabe-raben Maki",
        "waiting"       : "Loda hoto na ganyen rogo sannan danna Bincika Ganye",
        "how_title"     : "Yadda Yake Aiki",
        "step1"         : "Loda",
        "step1_desc"    : "Ɗauki ko loda hoto mai kyau",
        "step2"         : "Bincika",
        "step2_desc"    : "AI ta MobileNetV2 na bincika ganye",
        "step3"         : "Gano",
        "step3_desc"    : "An gano cuta daga cikin nau'i 5",
        "step4"         : "Magani",
        "step4_desc"    : "Sami nasihar magani nan take",
        "stats_scans"   : "Ganyen da aka Bincika",
        "stats_acc"     : "Daidaiton Ƙirar",
        "stats_classes" : "Nau'in Cututtuka",
        "stats_size"    : "Girman Ƙirar",
        "footer"        : "Ƙirƙirawa ta Delight Entreprises · AI don Noma na Afirka 🌍",
        "retake"        : "Da Fatan Sake Ɗaukar Hoto",
        "select_lang"   : "Language / Harshe",
        "severity_none" : "LAFIYA",
        "severity_mod"  : "MATSAKAICI",
        "severity_high" : "TSANANI",
        "severity_crit" : "HAƊARI",
    },
    "ee": {
        "title"         : "Agbeli Dɔ Kpɔla",
        "subtitle"      : "Nufiafi si wòzã ɖe agbeli glewo dɔ kpɔm, na adzeviwo le Afrika ƒe agble ŋu.",
        "upload_label"  : "📸 Tso agbeli gle ƒe foto le afisia",
        "detect_btn"    : "🔍  Kpɔ gle dɔ",
        "tips_title"    : "📷 Foto ƒe Kpekpeɖeŋu",
        "tips"          : [
            "Zã ɣeyiɣi ƒe dzeƒe",
            "Kpɔ gle ɖeka kaba",
            "Hu gle na foto",
            "Dzidzi du na nufiafi"
        ],
        "result_title"  : "Kpɔla ƒe Dɔwɔwɔ",
        "confidence"    : "Teƒe",
        "severity"      : "Dɔ ƒe Tsitsri",
        "symptoms"      : "Dɔ ƒe Adesim",
        "cause"         : "Dɔ ƒe Gbegbɔgblɔ",
        "spreads"       : "Alesi Dɔ Ɖoa",
        "treatment"     : "Ɖoɖo Ƒe Ʋɔwɔwɔwo",
        "prevention"    : "Dɔ Ƒe Siasiam",
        "breakdown"     : "Teƒe Ƒe Nuteƒeɖeɖe",
        "waiting"       : "Tso agbeli gle foto eye i tia Kpɔ gle dɔ",
        "how_title"     : "Alesi Nufiafi Wɔa Dɔ",
        "step1"         : "Tso Foto",
        "step1_desc"    : "Tso ehehe agbeli gle foto",
        "step2"         : "Kpɔkpɔ",
        "step2_desc"    : "Nufiafi ƒe AI le gle kpɔm",
        "step3"         : "Hiã Dɔ",
        "step3_desc"    : "Dɔ wòkpɔ le klasse atɔ̃ ŋu",
        "step4"         : "Ɖoɖo",
        "step4_desc"    : "Xɔ ɖoɖo ƒe kpekpeɖeŋu",
        "stats_scans"   : "Glewo Wòkpɔ",
        "stats_acc"     : "Nufiafi ƒe Teƒe",
        "stats_classes" : "Dɔwo ƒe Klasse",
        "stats_size"    : "Nufiafi ƒe Didi",
        "footer"        : "Woɖo na Delight Entreprises · AI na Afrika ƒe Agble Dɔwɔwɔ 🌍",
        "retake"        : "Tso Foto Ɖe Gbɔ",
        "select_lang"   : "Language / Gbe",
        "severity_none" : "LƆLƆ̃",
        "severity_mod"  : "KATÃ",
        "severity_high" : "TSITSRI",
        "severity_crit" : "VEVIE HAHÃ",
    },
    "kbp": {
        "title"         : "Kasaafa Kulusi Yɔd̀ʊ Tɛɛwʊ",
        "subtitle"      : "Teŋ tʊ lɛ AI kɛ kasaafa legʊ kulusi yɔd̀ʊ tɔ, taŋfalʊ Afrik taa kɛ.",
        "upload_label"  : "📸 Sʊ kasaafa legʊ foto cɛ wɛ",
        "detect_btn"    : "🔍  Yɔd̀ʊ legʊ kulusi",
        "tips_title"    : "📷 Foto Kɛ Tɔ̃ŋ",
        "tips"          : [
            "Zã wɛld̀ʊ taa ɣeyiɣi",
            "Kɔlɔ legʊ kʊŋ ɖeka",
            "Hu legʊ foto taa",
            "Tɛɛwʊ kɛ dzidzi"
        ],
        "result_title"  : "Yɔd̀ʊ Tɛɛ Lɛ",
        "confidence"    : "Pɛlɛ",
        "severity"      : "Kulusi Lɛ Tsɩtsɩ",
        "symptoms"      : "Kulusi Lɛ Ñɩmɩ",
        "cause"         : "Kulusi Lɛ Tɔ̃ŋ",
        "spreads"       : "Kulusi Lɛ Ɖɔ Tɔ̃ŋ",
        "treatment"     : "Laafɩ Ŋmaŋ Lɛ",
        "prevention"    : "Kulusi Siasiam",
        "breakdown"     : "Pɛlɛ Lɛ Ñɩŋ",
        "waiting"       : "Sʊ kasaafa legʊ foto na tia Yɔd̀ʊ legʊ kulusi",
        "how_title"     : "Ɩyɔ Tɛɛwʊ Lɛ Tɔŋ",
        "step1"         : "Sʊ Foto",
        "step1_desc"    : "Sʊ kasaafa legʊ foto",
        "step2"         : "Kpakpaañɩ",
        "step2_desc"    : "AI lɛ legʊ kpakpaañɩ",
        "step3"         : "Yɔd̀ʊ",
        "step3_desc"    : "Kulusi lɛ yɔd̀ʊ klasi 5 taa",
        "step4"         : "Laafɩ",
        "step4_desc"    : "Xɔ laafɩ kɛ tɔ̃ŋ",
        "stats_scans"   : "Legʊ Wòkpɔ",
        "stats_acc"     : "Tɛɛwʊ Lɛ Pɛlɛ",
        "stats_classes" : "Kulusi Klasi",
        "stats_size"    : "Tɛɛwʊ Lɛ Didi",
        "footer"        : "Delight Entreprises kɛ tɔ · AI Afrik taŋfalʊ kɛ 🌍",
        "retake"        : "Sʊ Foto Ɖe Gbɔ",
        "select_lang"   : "Language / Kɩtɩkpɩyɛ",
        "severity_none" : "ƖLƐ KƱLƱ",
        "severity_mod"  : "KATÃ",
        "severity_high" : "TSƖTSƖ",
        "severity_crit" : "VƐVƐ HAHÃ",
    },
}
# ══════════════════════════════════════════════════════════════
# DISEASE INFO — all 5 languages
# ══════════════════════════════════════════════════════════════
DISEASE_INFO = {
    "CBB": {
        "emoji" : "🦠",
        "color" : "#EF4444",
        "severity": "HIGH",
        "en": {
            "full_name" : "Cassava Bacterial Blight",
            "symptoms"  : "Angular water-soaked leaf spots, wilting, stem dieback, gummy exudate on stems.",
            "cause"     : "Bacterium: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Infected cuttings, tools, rain splash, and wind",
            "treatment" : ["Remove and burn all infected plant parts immediately","Avoid working in fields when plants are wet","Use clean disease-free cuttings for planting","Apply copper-based bactericides (Kocide, Copper Oxychloride)","Rotate crops — avoid planting cassava in same field for 1 season"],
            "prevention": "Use certified disease-free stem cuttings. Plant resistant varieties like TME 419.",
        },
        "fr": {
            "full_name" : "Brûlure Bactérienne du Manioc",
            "symptoms"  : "Taches foliaires angulaires gorgées d'eau, flétrissement, dessèchement des tiges.",
            "cause"     : "Bactérie : Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Boutures infectées, outils, éclaboussures de pluie et vent",
            "treatment" : ["Enlever et brûler immédiatement toutes les parties infectées","Éviter de travailler dans les champs quand les plantes sont mouillées","Utiliser des boutures saines et certifiées","Appliquer des bactéricides à base de cuivre","Pratiquer la rotation des cultures pendant au moins 1 saison"],
            "prevention": "Utiliser des boutures certifiées sans maladie. Planter des variétés résistantes comme TME 419.",
        },
        "sw": {
            "full_name" : "Ugonjwa wa Bakteria wa Muhogo",
            "symptoms"  : "Madoa ya majani yenye maji, kunyauka, kifo cha shina.",
            "cause"     : "Bakteria: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Vipande vilivyoathirika, zana, maji ya mvua, na upepo",
            "treatment" : ["Ondoa na choma sehemu zote zilizoathirika mara moja","Epuka kufanya kazi shambani wakati mimea ina unyevu","Tumia vipande visivyo na magonjwa","Tumia dawa za bakteria zenye shaba","Zungusha mazao kwa angalau msimu 1"],
            "prevention": "Tumia vipande vilivyothibitishwa bila magonjwa. Panda aina zinazostahimili kama TME 419.",
        },
        "yo": {
            "full_name" : "Àrùn Bakitéríà Igi Pàkì",
            "symptoms"  : "Àwọn abawọn omi lórí ewé, ìrẹwẹ̀sì, ikú ẹsẹ igi.",
            "cause"     : "Bakitéríà: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Àwọn ẹ̀kúnrẹ́rẹ́ tó ní àrùn, irinṣẹ́, òjò àti afẹ́fẹ́",
            "treatment" : ["Yọ kí o sì sun gbogbo apá tó ní àrùn lẹ́sẹ̀kẹsẹ̀","Má ṣiṣẹ́ ní pápá nígbà tí ewé wọn nínú omi","Lo àwọn ẹ̀kúnrẹ́rẹ́ tó mọ́","Lo àwọn oogun bakitéríà tó ní bàbà","Yí àgbàdo padà fún ó kéré jù ìgbà kan"],
            "prevention": "Lo àwọn ẹ̀kúnrẹ́rẹ́ tó jẹ́rìí sí. Gbin àwọn irúgbìn tó léwájú bí TME 419.",
        },
        "ha": {
            "full_name" : "Cutar Ƙwayoyin Cuta ta Rogo",
            "symptoms"  : "Tabo na ruwa a ganye, bushewar shuka, mutuwar reshe.",
            "cause"     : "Ƙwayoyin cuta: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Yankan da aka kamu, kayan aiki, ruwan sama, da iska",
            "treatment" : ["Cire da ƙone dukkan sassa masu cuta nan take","Guje wa aiki a gonaki lokacin da shuke-shuke suna jike","Yi amfani da yankan da babu cuta","Yi amfani da maganin ƙwayoyin cuta na jan ƙarfe","Juya amfanin gona na tsawon lokaci 1 aƙalla"],
            "prevention": "Yi amfani da yankan da aka tabbatar babu cuta. Shuka ire-iren da ke jurewa kamar TME 419.",
        },
        "ee": {
            "full_name" : "Agbeli Baketeria Dɔ",
            "symptoms"  : "Gle dzi agbledede, gle zuzu, ati ƒe ku, mìmì le atia dzi.",
            "cause"     : "Baketeria: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Dɔ ati, nudzraɖiwo, zãzã kple fafa",
            "treatment" : [
                "Tsɔ eye ku dɔ ati katã akpakpa",
                "Mègaɖo agble me o gle siwo ƒe tsi le wo dzi",
                "Zã ati siwo mele dɔ o",
                "Zã baketeria ɖoɖo (Kocide, Copper Oxychloride)",
                "Tafi ati siwo wole agble ŋu yeye aɖe me ƒo ƒe ŋu",
            ],
            "prevention": "Zã ati siwo wòkpɔ be mele dɔ o. Ɖo TME 419.",
        },
        "kbp": {
            "full_name" : "Kasaafa Baketeria Kulusi",
            "symptoms"  : "Legʊ taa ñɩmɩ, ɩwɛtʊ, ateŋ ku, mìmì ateŋ taa.",
            "cause"     : "Baketeria: Xanthomonas axonopodis pv. manihotis",
            "spread"    : "Kulusi ateŋ, tɛɛwʊ, lɩm kɛ helim",
            "treatment" : [
                "Tɔkɩ na sʊ kulusi ateŋ katã na yaa",
                "Taa taŋ kɛ agblɛ taa legʊ ñɩm taa",
                "Zã ateŋ kulusi wɛlɛ",
                "Zã baketeria laafɩ (Kocide, Copper Oxychloride)",
                "Kɔlɔ kasaafa kʊ agblɛ hʊlʊ taa ɖeka ɖuɣu",
            ],
            "prevention": "Zã ateŋ wòkpɔ be kulusi wɛlɛ. Sʊ TME 419.",
        },
    },
    "CBSD": {
        "emoji" : "🟤",
        "color" : "#DC2626",
        "severity": "CRITICAL",
        "en": {
            "full_name" : "Cassava Brown Streak Disease",
            "symptoms"  : "Yellow/brown streaks on leaves, brown necrotic patches in storage roots.",
            "cause"     : "Virus: CBSV and UCBSV transmitted by whiteflies",
            "spread"    : "Whiteflies (Bemisia tabaci) and infected cuttings",
            "treatment" : ["Immediately uproot and burn infected plants","Control whitefly populations with insecticide sprays","Use CBSD-tolerant varieties: Narocass 1, Kiroba, Namikonga","Source planting material only from certified clean nurseries","Never replant cuttings from infected fields"],
            "prevention": "Plant only CBSD-resistant varieties. Control whiteflies early.",
        },
        "fr": {
            "full_name" : "Maladie des Stries Brunes du Manioc",
            "symptoms"  : "Stries jaunes/brunes sur les feuilles, taches nécrotiques brunes dans les racines.",
            "cause"     : "Virus: CBSV et UCBSV transmis par les mouches blanches",
            "spread"    : "Mouches blanches (Bemisia tabaci) et boutures infectées",
            "treatment" : ["Déraciner et brûler immédiatement les plantes infectées","Contrôler les mouches blanches avec des insecticides","Utiliser des variétés tolérantes: Narocass 1, Kiroba, Namikonga","S'approvisionner uniquement dans des pépinières certifiées","Ne jamais replanter des boutures de champs infectés"],
            "prevention": "Planter uniquement des variétés résistantes au CBSD. Contrôler les mouches blanches tôt.",
        },
        "sw": {
            "full_name" : "Ugonjwa wa Kupiga Mstari wa Kahawia",
            "symptoms"  : "Mistari ya njano/kahawia kwenye majani, madoa ya necrotic kwenye mizizi.",
            "cause"     : "Virusi: CBSV na UCBSV kupitia nzi weupe",
            "spread"    : "Nzi weupe (Bemisia tabaci) na vipande vilivyoathirika",
            "treatment" : ["Ng'oa na choma mimea iliyoathirika mara moja","Dhibiti nzi weupe kwa dawa za kuua wadudu","Tumia aina zinazostahimili: Narocass 1, Kiroba","Pata mbegu tu kutoka bustani zilizothibitishwa","Usipande vipande kutoka mashamba yaliyoathirika"],
            "prevention": "Panda aina zinazostahimili CBSD peke yake. Dhibiti nzi weupe mapema.",
        },
        "yo": {
            "full_name" : "Àrùn Ìlà Àwọ̀ Búrú Igi Pàkì",
            "symptoms"  : "Àwọn ìlà ofeèfee/àwọ̀ búrú lórí ewé, àwọn abawọn dúdú nínú gbòǹgbò.",
            "cause"     : "Fáírọ̀sì: CBSV àti UCBSV nípa àwọn eṣinṣin funfun",
            "spread"    : "Eṣinṣin funfun (Bemisia tabaci) àti àwọn ẹ̀kúnrẹ́rẹ́ tó ní àrùn",
            "treatment" : ["Wà kí o sì sun àwọn irúgbìn tó ní àrùn lẹ́sẹ̀kẹsẹ̀","Ṣàkóso àwọn eṣinṣin funfun pẹ̀lú ìfọ́ oogun","Lo àwọn irúgbìn tó léwájú: Narocass 1, Kiroba","Gba ohun èlò gbìn láti ọ̀dọ̀ àwọn ọgbà ẹ̀kọ́ tó jẹ́rìí sí","Má gbìn àwọn ẹ̀kúnrẹ́rẹ́ láti pápá tó ní àrùn"],
            "prevention": "Gbin àwọn irúgbìn tó léwájú CBSD nìkan. Ṣàkóso eṣinṣin funfun ní àkọ́kọ́.",
        },
        "ha": {
            "full_name" : "Cutar Ratsin Launin Ruwan Kasa",
            "symptoms"  : "Ratsin rawaya/ruwan kasa a ganye, tabo mara lafiya a cikin tushe.",
            "cause"     : "Ƙwayar cuta: CBSV da UCBSV ta hanyar farin tashi",
            "spread"    : "Farin tashi (Bemisia tabaci) da yankan da aka kamu",
            "treatment" : ["Tumke da ƙone shuke-shuke masu cuta nan take","Sarrafa farar tashi da magungunan kashe kwari","Yi amfani da ire-iren da ke jurewa: Narocass 1, Kiroba","Sami kayan shuka daga wuraren da aka tabbatar kawai","Kada a sake shuka yankan daga gonaki masu cuta"],
            "prevention": "Shuka ire-iren da ke jurewa CBSD kawai. Sarrafa farar tashi da wuri.",
        },
        "ee": {
            "full_name" : "Agbeli Kɔkɔe Ƒe Dɔ",
            "symptoms"  : "Gle dzi srɔ̃srɔ̃ dzidzi kple kɔkɔe, agbeli ƒe nuku le wua.",
            "cause"     : "Fáírɔsì: CBSV kple UCBSV le zãzã tefe",
            "spread"    : "Zãzã (Bemisia tabaci) kple dɔ ati",
            "treatment" : [
                "Tsɔ eye ku dɔ ati katã akpakpa",
                "Zã ɖoɖo na zãzã kpɔtɔwo",
                "Ɖo Narocass 1, Kiroba ame ati",
                "Xɔ ati kpakpa le dzɔdzɔe ƒe afɔɖeƒewo",
                "Megaɖo dɔ atisiawo le agbleme o",
            ],
            "prevention": "Ɖo ati siwo le CBSD ta. Kpe zãzã ŋu ɣeyiɣi.",
        },
        "kbp": {
            "full_name" : "Kasaafa Kɔkɔe Kulusi",
            "symptoms"  : "Legʊ taa srɔ̃srɔ̃ kɔkɔe, kasaafa nuku taa kulusi.",
            "cause"     : "Fáírɔsì: CBSV kɛ UCBSV zãzã tɔ̃ŋ",
            "spread"    : "Zãzã (Bemisia tabaci) kɛ kulusi ateŋ",
            "treatment" : [
                "Tɔkɩ sʊ kulusi ateŋ katã na yaa",
                "Zã laafɩ zãzã kɔlɔ tɔ̃ŋ",
                "Sʊ Narocass 1, Kiroba ateŋ",
                "Xɔ ateŋ kpakpa dzɔdzɔe tɔ̃ŋ",
                "Taa sʊ kulusi ateŋ agblɛ taa",
            ],
            "prevention": "Sʊ CBSD kulusi wɛlɛ ateŋ. Kɔlɔ zãzã ɣeyiɣi.",
        },
    },
    "CGM": {
        "emoji" : "🐛",
        "color" : "#F59E0B",
        "severity": "MODERATE",
        "en": {
            "full_name" : "Cassava Green Mite",
            "symptoms"  : "Leaf chlorosis, distortion, stunted growth, bronze discoloration.",
            "cause"     : "Mite: Mononychellus tanajoa",
            "spread"    : "Wind, infected planting material, and farm tools",
            "treatment" : ["Spray with acaricides: Abamectin, Dicofol, or Spiromesifen","Introduce natural predators: Typhlodromalus aripo","Remove heavily infested leaves and burn them","Avoid drought stress — mites thrive in dry conditions","Use neem-based sprays as organic alternative"],
            "prevention": "Monitor fields regularly. Biological control is most sustainable.",
        },
        "fr": {
            "full_name" : "Acarien Vert du Manioc",
            "symptoms"  : "Chlorose foliaire, déformation, croissance rabougrie, décoloration bronzée.",
            "cause"     : "Acarien: Mononychellus tanajoa",
            "spread"    : "Vent, matériel végétal infecté et outils agricoles",
            "treatment" : ["Pulvériser des acaricides: Abamectin, Dicofol ou Spiromesifen","Introduire des prédateurs naturels: Typhlodromalus aripo","Retirer les feuilles très infestées et les brûler","Éviter le stress hydrique — les acariens prolifèrent dans la sécheresse","Utiliser des sprays à base de neem"],
            "prevention": "Surveiller régulièrement. La lutte biologique est la plus durable.",
        },
        "sw": {
            "full_name" : "Utitiri wa Kijani wa Muhogo",
            "symptoms"  : "Manjano ya majani, upotoshaji, ukuaji mdogo, rangi ya shaba.",
            "cause"     : "Utitiri: Mononychellus tanajoa",
            "spread"    : "Upepo, vifaa vya kupanda vilivyoathirika, na zana za kilimo",
            "treatment" : ["Nyunyizia acaricides: Abamectin, Dicofol au Spiromesifen","Ingiza wadudu wa asili wanaowinda: Typhlodromalus aripo","Ondoa majani yaliyoathirika sana na kuyachoma","Epuka msongo wa ukame — utitiri hustawi katika ukame","Tumia dawa za neem kama mbadala wa kikaboni"],
            "prevention": "Fuatilia mashamba mara kwa mara. Udhibiti wa kibiolojia ni endelevu zaidi.",
        },
        "yo": {
            "full_name" : "Màìtì Àwọ̀ Ewé Igi Pàkì",
            "symptoms"  : "Ìdàrú àwọ̀ ewé, ìrọ̀kẹ̀, ìdẹ̀rọ̀ ìdàgbàsókè, àwọ̀ bàbà.",
            "cause"     : "Màìtì: Mononychellus tanajoa",
            "spread"    : "Afẹ́fẹ́, ohun èlò gbìn tó ní àrùn, àti irinṣẹ́ oko",
            "treatment" : ["Fọ pẹ̀lú acaricides: Abamectin, Dicofol tàbí Spiromesifen","Mú àwọn apanirun àdánidá wọlé: Typhlodromalus aripo","Yọ àwọn ewé tó kún fún màìtì kí o sì sun wọn","Yẹra fún ìpọnjú omi gbígbẹ — màìtì ń gbépo ní gbígbẹ","Lo àwọn ìfọ́ neem gẹ́gẹ́ bí àṣàyàn àdánidá"],
            "prevention": "Ṣàbójútó pápá rẹ leralera. Ìṣàkóso ẹ̀dá jẹ́ tó pẹ́ jù.",
        },
        "ha": {
            "full_name" : "Kudan Kore na Rogo",
            "symptoms"  : "Rawaya a ganye, karkacewar ganye, rashin girma, launin tagulla.",
            "cause"     : "Kuda: Mononychellus tanajoa",
            "spread"    : "Iska, kayan shuka masu cuta, da kayan aiki na noma",
            "treatment" : ["Yi fesa da acaricides: Abamectin, Dicofol ko Spiromesifen","Gabatar da mafarauta na halitta: Typhlodromalus aripo","Cire ganye masu kuda sosai ka ƙone su","Guje wa damuwar fari — kuda na bunƙasa a yanayin bushewa","Yi amfani da feshin neem a matsayin madadin halitta"],
            "prevention": "Kula da gonaki akai-akai. Sarrafa ta hanyar halittu shine mafi dorewa.",
        },
        "ee": {
            "full_name" : "Agbeli Gbɔ̃ Ƒe Dɔ",
            "symptoms"  : "Gle ƒe dzidzi yeyiyi, gle gbɔgblɔ, ati ƒe dɔwɔwɔ dada.",
            "cause"     : "Gbɔ̃: Mononychellus tanajoa",
            "spread"    : "Fafa, dɔ ati kple agble ƒe nudzraɖiwo",
            "treatment" : [
                "Zã acaricides: Abamectin, Dicofol alo Spiromesifen",
                "Tsɔ nubablawo vá: Typhlodromalus aripo",
                "Tsɔ eye ku dɔ glewo katã",
                "Gblɔ xɔxɔ nusẽsẽ — gbɔ̃wo tona le ɣeyiɣi sẽwo me",
                "Zã neem ƒe ɖoɖo",
            ],
            "prevention": "Kpɔ agble le ŋkeke siwo katã. Nubabla ƒe siasiam dina gbãtɔ.",
        },
        "kbp": {
            "full_name" : "Kasaafa Gbɔ̃ Kulusi",
            "symptoms"  : "Legʊ dzidzi yeyiyi, legʊ gbɔgblɔ, ateŋ dɔwɔ dada.",
            "cause"     : "Gbɔ̃: Mononychellus tanajoa",
            "spread"    : "Helim, kulusi ateŋ, tɛɛwʊ tɔ̃ŋ",
            "treatment" : [
                "Zã acaricides: Abamectin, Dicofol kɛ Spiromesifen",
                "Sʊ nubabla: Typhlodromalus aripo",
                "Tɔkɩ sʊ kulusi legʊ katã na yaa",
                "Taa ɖe lɩm tɔ̃ŋ — gbɔ̃ tona ɣeyiɣi taa",
                "Zã neem laafɩ",
            ],
            "prevention": "Kɔlɔ agblɛ kɛ wɛld̀ʊ. Nubabla siasiam dina.",
        },
    },
    "CMD": {
        "emoji" : "🍂",
        "color" : "#EA580C",
        "severity": "HIGH",
        "en": {
            "full_name" : "Cassava Mosaic Disease",
            "symptoms"  : "Mosaic yellowing, leaf distortion, twisted leaflets, stunted plants.",
            "cause"     : "Virus: African Cassava Mosaic Virus (ACMV) via whiteflies",
            "spread"    : "Whiteflies and infected stem cuttings",
            "treatment" : ["Uproot and destroy severely infected plants","Control whitefly vectors using imidacloprid or thiamethoxam","Replace with CMD-resistant varieties: TME 419","Practice roguing — remove infected plants early","Never use cuttings from mosaic-infected plants"],
            "prevention": "CMD-resistant varieties reduce yield loss by up to 90%.",
        },
        "fr": {
            "full_name" : "Maladie de la Mosaïque du Manioc",
            "symptoms"  : "Jaunissement en mosaïque, déformation des feuilles, torsion, nanisme.",
            "cause"     : "Virus: Virus Africain de la Mosaïque du Manioc (ACMV) via mouches blanches",
            "spread"    : "Mouches blanches et boutures de tiges infectées",
            "treatment" : ["Déraciner et détruire les plantes très infectées","Contrôler les mouches blanches avec imidacloprid ou thiamethoxam","Remplacer par des variétés résistantes: TME 419","Pratiquer le roguing — enlever les plantes infectées tôt","Ne jamais utiliser des boutures de plantes mosaïquées"],
            "prevention": "Les variétés résistantes réduisent les pertes jusqu'à 90%.",
        },
        "sw": {
            "full_name" : "Ugonjwa wa Moseki wa Muhogo",
            "symptoms"  : "Njano ya moseki, upotoshaji wa majani, majani yaliopinda, mimea midogo.",
            "cause"     : "Virusi: African Cassava Mosaic Virus (ACMV) kupitia nzi weupe",
            "spread"    : "Nzi weupe na vipande vya shina vilivyoathirika",
            "treatment" : ["Ng'oa na uharibu mimea iliyoathirika sana","Dhibiti nzi weupe kwa kutumia imidacloprid au thiamethoxam","Badilisha na aina zinazostahimili: TME 419","Fanya roguing — ondoa mimea iliyoathirika mapema","Usitumie vipande kutoka kwa mimea yenye moseki"],
            "prevention": "Aina zinazostahimili hupunguza hasara ya mazao hadi 90%.",
        },
        "yo": {
            "full_name" : "Àrùn Mosaic Igi Pàkì",
            "symptoms"  : "Ojúlówó ìdàrú àwọ̀ ewé, ìrọ̀kẹ̀, àwọn ẹ̀ka tó yí padà, àwọn ìrúgbìn kéré.",
            "cause"     : "Fáírọ̀sì: African Cassava Mosaic Virus (ACMV) nípa eṣinṣin funfun",
            "spread"    : "Eṣinṣin funfun àti àwọn ẹ̀kúnrẹ́rẹ́ ẹsẹ tó ní àrùn",
            "treatment" : ["Wà kí o pa àwọn irúgbìn tó ní àrùn jù","Ṣàkóso eṣinṣin funfun pẹ̀lú imidacloprid tàbí thiamethoxam","Rọ́pò pẹ̀lú àwọn irúgbìn tó léwájú: TME 419","Ṣe roguing — yọ àwọn irúgbìn tó ní àrùn ní àkọ́kọ́","Má lo àwọn ẹ̀kúnrẹ́rẹ́ láti irúgbìn mosaic"],
            "prevention": "Àwọn irúgbìn tó léwájú dín ìpadánù èso dé 90%.",
        },
        "ha": {
            "full_name" : "Cutar Mosaic ta Rogo",
            "symptoms"  : "Rawayar mosaic, karkacewar ganye, ganye da aka murda, ƙananan shuke-shuke.",
            "cause"     : "Ƙwayar cuta: African Cassava Mosaic Virus (ACMV) ta farin tashi",
            "spread"    : "Farin tashi da yankan da aka kamu",
            "treatment" : ["Tumke da hallaka shuke-shuke masu cuta sosai","Sarrafa farar tashi da imidacloprid ko thiamethoxam","Maye gurbin ire-iren masu jurewa: TME 419","Yi roguing — cire shuke-shuke masu cuta da wuri","Kada a yi amfani da yankan daga shuke-shuke masu mosaic"],
            "prevention": "Ire-iren masu jurewa suna rage asarar amfanin gona zuwa 90%.",
        },
        "ee": {
            "full_name" : "Agbeli Mosaic Dɔ",
            "symptoms"  : "Gle ƒe dzidzi yeyiyi mosaic, gle gbɔgblɔ, ati ƒe dɔwɔwɔ dada.",
            "cause"     : "Fáírɔsì: African Cassava Mosaic Virus (ACMV) le zãzã tefe",
            "spread"    : "Zãzã kple dɔ ati",
            "treatment" : [
                "Tsɔ eye ku dɔ ati vevie katã",
                "Kpe zãzã ŋu le imidacloprid alo thiamethoxam",
                "Ɖo TME 419 ame ati",
                "Wɔ roguing — tsɔ dɔ ati ɣeyiɣi",
                "Megazã mosaic atiwo ƒe ati o",
            ],
            "prevention": "Ati siwo le CBSD ta dzɔdzɔe dze ɣesẽ 90% na agble.",
        },
        "kbp": {
            "full_name" : "Kasaafa Mosaic Kulusi",
            "symptoms"  : "Legʊ mosaic dzidzi, legʊ gbɔgblɔ, ateŋ dɔwɔ dada.",
            "cause"     : "Fáírɔsì: African Cassava Mosaic Virus (ACMV) zãzã tɔ̃ŋ",
            "spread"    : "Zãzã kɛ kulusi ateŋ",
            "treatment" : [
                "Tɔkɩ sʊ kulusi ateŋ vɛvɛ katã",
                "Kɔlɔ zãzã imidacloprid kɛ thiamethoxam tɔ̃ŋ",
                "Sʊ TME 419 ateŋ",
                "Wɔ roguing — sʊ kulusi ateŋ ɣeyiɣi",
                "Taa zã mosaic ateŋ",
            ],
            "prevention": "Kulusi wɛlɛ ateŋ dze ɣesẽ 90% agblɛ kɛ.",
        },
    },
    "Healthy": {
        "emoji" : "✅",
        "color" : "#16A34A",
        "severity": "NONE",
        "en": {
            "full_name" : "Healthy Cassava Plant",
            "symptoms"  : "No disease symptoms detected. Your plant looks great!",
            "cause"     : "N/A — Plant is healthy",
            "spread"    : "N/A",
            "treatment" : ["Continue regular weekly field monitoring","Maintain good field hygiene — remove weeds","Apply balanced NPK fertilizer at recommended rates","Ensure adequate plant spacing for air circulation","Keep field health records for future reference"],
            "prevention": "Maintain good agricultural practices to keep plants healthy.",
        },
        "fr": {
            "full_name" : "Plant de Manioc Sain",
            "symptoms"  : "Aucun symptôme de maladie détecté. Votre plante a l'air excellente !",
            "cause"     : "N/A — La plante est saine",
            "spread"    : "N/A",
            "treatment" : ["Continuer la surveillance hebdomadaire du champ","Maintenir une bonne hygiène du champ — enlever les mauvaises herbes","Appliquer un engrais NPK équilibré aux doses recommandées","Assurer un espacement adéquat pour la circulation de l'air","Tenir des registres de santé du champ"],
            "prevention": "Maintenir de bonnes pratiques agricoles pour garder les plantes saines.",
        },
        "sw": {
            "full_name" : "Mmea wa Muhogo Wenye Afya",
            "symptoms"  : "Hakuna dalili za ugonjwa. Mmea wako unaonekana vizuri sana!",
            "cause"     : "Hakuna — Mmea una afya",
            "spread"    : "Hakuna",
            "treatment" : ["Endelea na ufuatiliaji wa kila wiki wa shamba","Dumisha usafi mzuri wa shamba — ondoa magugu","Tumia mbolea ya NPK kwa kiwango kinachopendekezwa","Hakikisha nafasi ya kutosha ya mimea kwa mzunguko wa hewa","Weka rekodi za afya ya shamba kwa marejeleo ya baadaye"],
            "prevention": "Dumisha mazoea mazuri ya kilimo ili mimea ibaki na afya.",
        },
        "yo": {
            "full_name" : "Irúgbìn Igi Pàkì tó Làára",
            "symptoms"  : "Kò sí àmì àrùn kankan. Irúgbìn rẹ dára gan-an!",
            "cause"     : "Kò sí — Irúgbìn ní ìlera",
            "spread"    : "Kò sí",
            "treatment" : ["Tẹsíwájú ìbójútó pápá ní ọ̀sọ̀ọ̀sẹ̀","Ṣetọ́jú mímọ́ pápá dára — yọ koriko","Lo ajílẹ̀ NPK gẹ́gẹ́ bí a ti ṣe ìgbaniléèrò","Rii dájú ìpín tó péye fún àfẹ́fẹ́ ní àárín","Pa àwọn àkọsílẹ̀ ìlera pápá mọ́"],
            "prevention": "Ṣetọ́jú àwọn àṣà àgbẹ̀ tó dára láti jẹ́ kí irúgbìn ní ìlera.",
        },
        "ha": {
            "full_name" : "Shuka na Rogo Mai Lafiya",
            "symptoms"  : "Ba a gano alamun cuta. Shuka naka yana da kyau!",
            "cause"     : "Babu — Shuka yana da lafiya",
            "spread"    : "Babu",
            "treatment" : ["Ci gaba da sa ido na mako-mako a gonaki","Kula da tsafta mai kyau na gona — cire ciyawa","Yi amfani da takin NPK mai daidaito a adadin da aka ba da shawarar","Tabbatar da isasshen tazara tsakanin shuke-shuke","Riƙe rikodin lafiyar gona don amfanin gaba"],
            "prevention": "Kula da kyawawan ayyukan noma don kiyaye shuke-shuke lafiya.",
        },
        "ee": {
            "full_name" : "Agbeli Si Le Lɔlɔ̃",
            "symptoms"  : "Dɔ adesim aɖe mele o. Wò ati le ŋutɔ lɔlɔ̃!",
            "cause"     : "Aɖeke mele o — ati le lɔlɔ̃",
            "spread"    : "Aɖeke mele o",
            "treatment" : [
                "Tɔ agble kpɔkpɔ le ŋkeke siwo katã",
                "Ɖo agble dzi mímɔ — tsɔ dzɔ ƒe nuwo",
                "Zã NPK ɖoɖo alesi wòɖo",
                "Ɖo ati katã gã le ɣe ƒe ɖoɖo ta",
                "Ŋlɔ agble ƒe lɔlɔ̃ dzidzi",
            ],
            "prevention": "Ɖo agble dɔwɔwɔ nyuie na ati siwo le lɔlɔ̃ ŋu.",
        },
        "kbp": {
            "full_name" : "Kasaafa Ɩlɛ Kʊlʊ",
            "symptoms"  : "Kulusi ñɩmɩ wɛlɛ. Nɩ ateŋ ɩlɛ kʊlʊ!",
            "cause"     : "Wɛlɛ — ateŋ ɩlɛ kʊlʊ",
            "spread"    : "Wɛlɛ",
            "treatment" : [
                "Kɔlɔ agblɛ kɛ wɛld̀ʊ katã",
                "Ɖo agblɛ taa mímɔ — sʊ dzɔ",
                "Zã NPK laafɩ alɩwɛ",
                "Ɖo ateŋ katã gã helim tɔ̃ŋ ta",
                "Ŋlɔ agblɛ ɩlɛ kʊlʊ tɔ̃ŋ",
            ],
            "prevention": "Ɖo agblɛ dɔwɔ nyuie na ateŋ ɩlɛ kʊlʊ ŋu.",
        },
     }
}

# ══════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

# ══════════════════════════════════════════════════════════════
# IMAGE QUALITY CHECKS
# ══════════════════════════════════════════════════════════════
def check_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_score  = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness  = gray.mean()
    if blur_score < 80:   return False, "blurry"
    if brightness < 40:   return False, "too_dark"
    if brightness > 220:  return False, "too_bright"
    return True, "ok"

def check_confidence(probs):
    max_conf = float(np.max(probs)) * 100
    entropy  = -np.sum(probs * np.log(probs + 1e-9))
    if max_conf < 55.0:  return False, "not_cassava", max_conf
    if entropy  > 1.4:   return False, "unclear",     max_conf
    return True, "ok", max_conf

# ══════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════
def preprocess(image):
    img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def run_inference(image):
    arr = preprocess(image)
    interpreter.set_tensor(inp_details[0]['index'], arr)
    interpreter.invoke()
    return interpreter.get_tensor(out_details[0]['index'])[0]

# ══════════════════════════════════════════════════════════════
# REJECTION CARD
# ══════════════════════════════════════════════════════════════
def get_rejection_html(reason, lang_code="en", score=None):
    t = UI_TEXT[lang_code]
    REJECTION = {
        "blurry"      : ("📷", "#F59E0B", "Image Too Blurry / Image trop floue",
                         "Your photo is out of focus. Please retake a sharper image.",
                         ["Hold phone steady with both hands","Tap screen on the leaf to focus","Move closer to the leaf","Wait for autofocus before shooting"]),
        "too_dark"    : ("🌑", "#6366F1", "Image Too Dark / Trop Sombre",
                         "The photo is too dark for analysis. Please improve lighting.",
                         ["Move to a brighter area or go outside","Use natural daylight — avoid shadows","Turn on flashlight if necessary","Avoid shooting against bright backgrounds"]),
        "too_bright"  : ("☀️", "#EF4444", "Image Overexposed / Surexposée",
                         "The image is too bright. Please reduce the light source.",
                         ["Move away from direct sunlight","Take photo in partial shade","Adjust camera exposure manually","Try a different angle"]),
        "not_cassava" : ("🌿", "#10B981", "Not Recognized as Cassava / Non Reconnu",
                         f"AI confidence too low ({score:.1f}%). This may not be a cassava leaf.",
                         ["Make sure you photograph a cassava leaf","Fill frame with the leaf — avoid clutter","Use a clearly visible single leaf","Ensure leaf is flat and fully unfolded","Avoid extreme angles — shoot straight on"]),
        "unclear"     : ("❓", "#8B5CF6", "Unclear Image / Image Peu Claire",
                         "The AI is uncertain. The image may not show a clear cassava leaf.",
                         ["Ensure this is a cassava leaf close-up","Remove other plants from the frame","Try better lighting and sharp focus","Use a different leaf from the same plant"]),
    }
    icon, color, title, msg, tips = REJECTION.get(reason, REJECTION["not_cassava"])
    tips_html = "".join([f"""
        <div style="display:flex;gap:10px;align-items:flex-start;margin:6px 0">
            <div style="background:{color};color:white;border-radius:50%;
                        min-width:20px;height:20px;display:flex;align-items:center;
                        justify-content:center;font-size:11px;font-weight:700;
                        flex-shrink:0">✓</div>
            <span style="font-size:13px;color:#374151;line-height:1.5">{tip}</span>
        </div>""" for tip in tips])

    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:680px;margin:0 auto">
        <div style="background:linear-gradient(135deg,{color}18,{color}08);
                    border:2px solid {color};border-radius:20px;padding:32px;
                    text-align:center;margin-bottom:16px">
            <div style="font-size:60px;margin-bottom:12px">{icon}</div>
            <h2 style="color:{color};font-size:20px;font-weight:800;margin:0 0 10px">
                {title}
            </h2>
            <p style="color:#4B5563;font-size:14px;margin:0 0 20px;
                      max-width:400px;margin-left:auto;margin-right:auto;line-height:1.6">
                {msg}
            </p>
            <div style="background:{color};color:white;border-radius:999px;
                        display:inline-block;padding:10px 24px;
                        font-size:14px;font-weight:700;letter-spacing:0.3px">
                📸 {t['retake']}
            </div>
        </div>
        <div style="background:white;border:1px solid #E5E7EB;border-radius:16px;
                    padding:20px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <h3 style="margin:0 0 14px;font-size:14px;font-weight:700;
                       color:#111827">💡 {t['tips_title']}</h3>
            {tips_html}
        </div>
    </div>"""

# ══════════════════════════════════════════════════════════════
# MAIN PREDICT
# ══════════════════════════════════════════════════════════════
def predict(image, language):
    lang_code = LANGUAGES.get(language, "en")
    t         = UI_TEXT[lang_code]

    if image is None:
        return f"<p style='color:#9CA3AF;text-align:center;padding:40px;font-size:15px'>{t['waiting']}</p>"

    # Quality check
    is_good, quality_reason = check_image_quality(image)
    if not is_good:
        return get_rejection_html(quality_reason, lang_code)

    # Inference
    try:
        probs = run_inference(image)
    except Exception as e:
        return f"<p style='color:#EF4444;text-align:center;padding:20px'>Error during inference: {str(e)}</p>"

    # Confidence check
    is_cassava, conf_reason, max_conf = check_confidence(probs)
    if not is_cassava:
        return get_rejection_html(conf_reason, lang_code, score=max_conf)

    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    disease    = DISEASE_INFO[pred_class]
    info       = disease.get(lang_code, disease["en"])
    color      = disease["color"]
    emoji      = disease["emoji"]

    SEVERITY_MAP = {
        "CRITICAL": t["severity_crit"],
        "HIGH"    : t["severity_high"],
        "MODERATE": t["severity_mod"],
        "NONE"    : t["severity_none"],
    }
    SEVERITY_COLORS = {
        "CRITICAL": "#DC2626",
        "HIGH"    : "#EA580C",
        "MODERATE": "#D97706",
        "NONE"    : "#16A34A",
    }
    sev_label = SEVERITY_MAP[disease["severity"]]
    sev_color = SEVERITY_COLORS[disease["severity"]]

    # Confidence bars
    bars_html = ""
    sorted_scores = sorted(
        [(CLASS_NAMES[i], float(probs[i]) * 100) for i in range(5)],
        key=lambda x: x[1], reverse=True
    )
    for cls, score in sorted_scores:
        d        = DISEASE_INFO[cls]
        di       = d.get(lang_code, d["en"])
        is_top   = cls == pred_class
        bold     = "font-weight:800;" if is_top else "font-weight:400;"
        bg       = f"background:{d['color']}12;" if is_top else ""
        bars_html += f"""
        <div style="padding:8px 12px;border-radius:10px;margin:4px 0;{bg}">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:5px">
                <span style="font-size:13px;{bold}color:#111827">
                    {d['emoji']} {cls}
                    <span style="font-weight:400;color:#6B7280;font-size:12px">
                        — {di['full_name']}
                    </span>
                </span>
                <span style="font-size:13px;{bold}color:{d['color']};
                             background:{d['color']}18;padding:2px 8px;
                             border-radius:999px;white-space:nowrap">
                    {score:.1f}%
                </span>
            </div>
            <div style="background:#F3F4F6;border-radius:999px;height:8px;overflow:hidden">
                <div style="width:{max(score,0.5):.1f}%;background:linear-gradient(90deg,{d['color']},{d['color']}99);
                            border-radius:999px;height:8px;
                            transition:width 0.6s ease"></div>
            </div>
        </div>"""

    # Treatment steps
    treatment_html = ""
    for i, step in enumerate(info["treatment"], 1):
        treatment_html += f"""
        <div style="display:flex;gap:12px;margin:10px 0;align-items:flex-start">
            <div style="background:linear-gradient(135deg,{color},{color}aa);
                        color:white;border-radius:50%;min-width:28px;height:28px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:12px;font-weight:800;flex-shrink:0;
                        box-shadow:0 2px 4px {color}44">{i}</div>
            <span style="font-size:14px;color:#374151;line-height:1.6;
                         padding-top:4px">{step}</span>
        </div>"""

    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:700px;
                margin:0 auto;padding-bottom:8px">

        <!-- HERO RESULT CARD -->
        <div style="background:linear-gradient(135deg,{color}20,{color}08);
                    border:2px solid {color}60;border-radius:20px;
                    padding:28px 24px;margin-bottom:14px;text-align:center;
                    box-shadow:0 4px 20px {color}18">
            <div style="font-size:60px;margin-bottom:10px;
                        filter:drop-shadow(0 4px 8px {color}40)">{emoji}</div>
            <div style="font-size:28px;font-weight:900;color:{color};
                        margin-bottom:4px;letter-spacing:-0.5px">{pred_class}</div>
            <div style="font-size:15px;color:#4B5563;margin-bottom:18px;
                        font-weight:500">{info['full_name']}</div>
            <div style="display:inline-flex;gap:10px;flex-wrap:wrap;
                        justify-content:center">
                <div style="background:white;border-radius:999px;padding:8px 20px;
                             font-size:14px;font-weight:700;color:{color};
                             box-shadow:0 2px 8px rgba(0,0,0,0.08);
                             border:1px solid {color}30">
                    🎯 {confidence:.1f}% {t['confidence']}
                </div>
                <div style="background:linear-gradient(135deg,{sev_color},{sev_color}cc);
                             border-radius:999px;padding:8px 20px;
                             font-size:14px;font-weight:700;color:white;
                             box-shadow:0 2px 8px {sev_color}44">
                    ⚠️ {sev_label}
                </div>
            </div>
        </div>

        <!-- CONFIDENCE BREAKDOWN -->
        <div style="background:white;border:1px solid #E5E7EB;border-radius:16px;
                    padding:20px;margin-bottom:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <h3 style="margin:0 0 12px;font-size:14px;font-weight:800;
                       color:#111827;text-transform:uppercase;letter-spacing:0.5px">
                📊 {t['breakdown']}
            </h3>
            {bars_html}
        </div>

        <!-- DISEASE DETAILS -->
        <div style="background:white;border:1px solid #E5E7EB;border-radius:16px;
                    padding:20px;margin-bottom:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <h3 style="margin:0 0 14px;font-size:14px;font-weight:800;
                       color:#111827;text-transform:uppercase;letter-spacing:0.5px">
                🔬 {t['result_title']}
            </h3>
            <div style="display:grid;gap:8px">
                <div style="background:#F8FAFC;border-left:4px solid {color};
                            border-radius:0 10px 10px 0;padding:12px 14px">
                    <div style="font-size:11px;font-weight:700;color:#6B7280;
                                text-transform:uppercase;letter-spacing:0.5px;
                                margin-bottom:5px">{t['symptoms']}</div>
                    <div style="font-size:14px;color:#1F2937;
                                line-height:1.6">{info['symptoms']}</div>
                </div>
                <div style="background:#F8FAFC;border-left:4px solid #6366F1;
                            border-radius:0 10px 10px 0;padding:12px 14px">
                    <div style="font-size:11px;font-weight:700;color:#6B7280;
                                text-transform:uppercase;letter-spacing:0.5px;
                                margin-bottom:5px">{t['cause']}</div>
                    <div style="font-size:14px;color:#1F2937;
                                line-height:1.6">{info['cause']}</div>
                </div>
                <div style="background:#F8FAFC;border-left:4px solid #F59E0B;
                            border-radius:0 10px 10px 0;padding:12px 14px">
                    <div style="font-size:11px;font-weight:700;color:#6B7280;
                                text-transform:uppercase;letter-spacing:0.5px;
                                margin-bottom:5px">{t['spreads']}</div>
                    <div style="font-size:14px;color:#1F2937;
                                line-height:1.6">{info['spread']}</div>
                </div>
            </div>
        </div>

        <!-- TREATMENT -->
        <div style="background:white;border:1px solid #E5E7EB;border-radius:16px;
                    padding:20px;margin-bottom:12px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <h3 style="margin:0 0 14px;font-size:14px;font-weight:800;
                       color:#111827;text-transform:uppercase;letter-spacing:0.5px">
                💊 {t['treatment']}
            </h3>
            {treatment_html}
        </div>

        <!-- PREVENTION -->
        <div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);
                    border:1px solid #86EFAC;border-radius:16px;padding:18px;
                    box-shadow:0 2px 8px rgba(22,163,74,0.08)">
            <h3 style="margin:0 0 8px;font-size:14px;font-weight:800;
                       color:#166534;text-transform:uppercase;letter-spacing:0.5px">
                🛡️ {t['prevention']}
            </h3>
            <p style="margin:0;font-size:14px;color:#166534;
                      line-height:1.7">{info['prevention']}</p>
        </div>

        <!-- FOOTER -->
        <div style="text-align:center;margin-top:14px;
                    font-size:11px;color:#9CA3AF;letter-spacing:0.3px">
            Powered by <b style="color:#374151">Delight Entreprises AI</b> ·
            MobileNetV2 · TFLite INT8 · 2.76MB
        </div>
    </div>"""

# ══════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════
CSS = """
* { box-sizing: border-box; }
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
    background: #F8FAFC !important;
}
.gr-button-primary {
    background: linear-gradient(135deg,#15803D,#166534) !important;
    border: none !important; border-radius: 12px !important;
    font-size: 16px !important; font-weight: 800 !important;
    padding: 14px 32px !important; color: white !important;
    box-shadow: 0 4px 14px rgba(21,128,61,0.35) !important;
    transition: all 0.2s ease !important; letter-spacing: 0.3px !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(21,128,61,0.45) !important;
}
footer { display:none !important; }
"""

def build_hero(lang):
    t = UI_TEXT[LANGUAGES.get(lang, "en")]
    return f"""
    <div style="background:linear-gradient(135deg,#052e16 0%,#14532D 40%,#166534 70%,#15803D 100%);
                border-radius:24px;padding:40px 24px 32px;margin-bottom:24px;
                box-shadow:0 8px 40px rgba(5,46,22,0.4);position:relative;overflow:hidden">
        <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                    background:rgba(255,255,255,0.03);border-radius:50%"></div>
        <div style="position:absolute;bottom:-60px;left:-30px;width:250px;height:250px;
                    background:rgba(255,255,255,0.02);border-radius:50%"></div>
        <div style="position:relative;text-align:center">
            <div style="font-size:64px;margin-bottom:12px;
                        filter:drop-shadow(0 4px 12px rgba(0,0,0,0.3))">🌿</div>
            <h1 style="color:white;font-size:32px;font-weight:900;margin:0 0 10px;
                       letter-spacing:-0.5px;text-shadow:0 2px 8px rgba(0,0,0,0.2)">
                {t['title']}
            </h1>
            <p style="color:#BBF7D0;font-size:15px;margin:0 0 24px;
                      max-width:560px;margin-left:auto;margin-right:auto;
                      line-height:1.7;opacity:0.9">
                {t['subtitle']}
            </p>
            <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;
                        margin-bottom:24px">
                <div style="background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
                             color:white;border-radius:999px;padding:6px 16px;
                             font-size:13px;font-weight:600;border:1px solid rgba(255,255,255,0.15)">
                    🎯 73% {t['stats_acc']}
                </div>
                <div style="background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
                             color:white;border-radius:999px;padding:6px 16px;
                             font-size:13px;font-weight:600;border:1px solid rgba(255,255,255,0.15)">
                    ⚡ TFLite INT8
                </div>
                <div style="background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
                             color:white;border-radius:999px;padding:6px 16px;
                             font-size:13px;font-weight:600;border:1px solid rgba(255,255,255,0.15)">
                    🌍 5 {t['stats_classes']}
                </div>
                <div style="background:rgba(255,255,255,0.12);backdrop-filter:blur(8px);
                             color:white;border-radius:999px;padding:6px 16px;
                             font-size:13px;font-weight:600;border:1px solid rgba(255,255,255,0.15)">
                    📱 2.76MB {t['stats_size']}
                </div>
            </div>
            <!-- STATS ROW -->
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;
                        max-width:600px;margin:0 auto">
                <div style="background:rgba(255,255,255,0.08);border-radius:14px;
                             padding:14px 8px;border:1px solid rgba(255,255,255,0.1)">
                    <div style="font-size:22px;font-weight:900;color:white">21K+</div>
                    <div style="font-size:11px;color:#86EFAC;margin-top:2px">{t['stats_scans']}</div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:14px;
                             padding:14px 8px;border:1px solid rgba(255,255,255,0.1)">
                    <div style="font-size:22px;font-weight:900;color:white">73%</div>
                    <div style="font-size:11px;color:#86EFAC;margin-top:2px">{t['stats_acc']}</div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:14px;
                             padding:14px 8px;border:1px solid rgba(255,255,255,0.1)">
                    <div style="font-size:22px;font-weight:900;color:white">5</div>
                    <div style="font-size:11px;color:#86EFAC;margin-top:2px">{t['stats_classes']}</div>
                </div>
                <div style="background:rgba(255,255,255,0.08);border-radius:14px;
                             padding:14px 8px;border:1px solid rgba(255,255,255,0.1)">
                    <div style="font-size:22px;font-weight:900;color:white">2.76MB</div>
                    <div style="font-size:11px;color:#86EFAC;margin-top:2px">{t['stats_size']}</div>
                </div>
            </div>
        </div>
    </div>"""

def build_disease_legend(lang):
    t = UI_TEXT[LANGUAGES.get(lang, "en")]
    SEVERITY_COLORS = {"CRITICAL":"#DC2626","HIGH":"#EA580C","MODERATE":"#D97706","NONE":"#16A34A"}
    cards = ""
    for cls in CLASS_NAMES:
        d    = DISEASE_INFO[cls]
        di   = d.get(LANGUAGES.get(lang,"en"), d["en"])
        sc   = SEVERITY_COLORS[d["severity"]]
        sm   = {"CRITICAL":t["severity_crit"],"HIGH":t["severity_high"],
                "MODERATE":t["severity_mod"],"NONE":t["severity_none"]}
        cards += f"""
        <div style="background:white;border:1px solid #E5E7EB;border-radius:14px;
                    padding:14px;text-align:center;flex:1;min-width:90px;
                    box-shadow:0 2px 8px rgba(0,0,0,0.04);
                    transition:transform 0.2s;border-top:3px solid {d['color']}">
            <div style="font-size:26px;margin-bottom:6px">{d['emoji']}</div>
            <div style="font-size:13px;font-weight:800;color:{d['color']};
                        margin-bottom:2px">{cls}</div>
            <div style="font-size:10px;color:#6B7280;margin-bottom:6px;
                        line-height:1.3">{di['full_name']}</div>
            <div style="background:{sc};color:white;border-radius:999px;
                        padding:2px 8px;font-size:9px;font-weight:700;
                        display:inline-block">{sm[d['severity']]}</div>
        </div>"""
    return f"""
    <div style="margin-bottom:20px">
        <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center">
            {cards}
        </div>
    </div>"""

def build_how_it_works(lang):
    t     = UI_TEXT[LANGUAGES.get(lang, "en")]
    steps = [
        ("📸", t["step1"], t["step1_desc"], "#3B82F6"),
        ("🧠", t["step2"], t["step2_desc"], "#8B5CF6"),
        ("🔬", t["step3"], t["step3_desc"], "#EC4899"),
        ("💊", t["step4"], t["step4_desc"], "#10B981"),
    ]
    steps_html = ""
    for i, (icon, title, desc, color) in enumerate(steps):
        connector = f"""<div style="width:40px;height:2px;background:linear-gradient(90deg,{color},{steps[i+1][3] if i<3 else color});
                                    margin:auto;flex-shrink:0"></div>""" if i < 3 else ""
        steps_html += f"""
        <div style="text-align:center;flex:1;min-width:100px">
            <div style="background:linear-gradient(135deg,{color},{color}cc);
                        width:52px;height:52px;border-radius:16px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:24px;margin:0 auto 10px;
                        box-shadow:0 4px 12px {color}40">{icon}</div>
            <div style="font-weight:800;font-size:13px;color:#111827;
                        margin-bottom:4px">{i+1}. {title}</div>
            <div style="font-size:12px;color:#6B7280;line-height:1.5">{desc}</div>
        </div>
        {connector}"""
    return f"""
    <div style="background:white;border:1px solid #E5E7EB;border-radius:20px;
                padding:28px;margin-top:20px;
                box-shadow:0 2px 8px rgba(0,0,0,0.04)">
        <h3 style="text-align:center;margin:0 0 24px;font-size:16px;
                   font-weight:900;color:#111827;text-transform:uppercase;
                   letter-spacing:0.5px">⚙️ {t['how_title']}</h3>
        <div style="display:flex;align-items:flex-start;gap:0;flex-wrap:wrap;
                    justify-content:center;gap:8px">
            {steps_html}
        </div>
    </div>"""

def build_footer(lang):
    t = UI_TEXT[LANGUAGES.get(lang, "en")]
    return f"""
    <div style="text-align:center;margin-top:20px;padding:20px;
                color:#6B7280;font-size:13px;border-top:1px solid #E5E7EB">
        <div style="font-weight:700;color:#111827;font-size:14px;margin-bottom:4px">
            🌿 Cassava AI Detector
        </div>
        {t['footer']}
    </div>"""

# ── BUILD UI ───────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Cassava AI Detector 🌿") as demo:

    language = gr.Dropdown(
        choices=list(LANGUAGES.keys()),
        value="🇬🇧 English",
        label="🌍 Language / Langue / Lugha / Ede / Harshe / Gbe / Kabiyè",
        interactive=True,
        scale=1,
    )

    hero_html   = gr.HTML(build_hero("🇬🇧 English"))
    legend_html = gr.HTML(build_disease_legend("🇬🇧 English"))

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="numpy",
                label="📸 Upload Cassava Leaf Image",
                height=320,
                sources=["upload", "webcam"],
            )
            tips_html = gr.HTML("""
            <div style="background:linear-gradient(135deg,#F0FDF4,#DCFCE7);
                        border:1px solid #86EFAC;border-radius:14px;
                        padding:16px;margin-top:10px">
                <b style="color:#166534 !important;font-size:13px;
                          font-weight:800;display:block;margin-bottom:8px">
                    📷 Tips for Best Results
                </b>
                <ul style="margin:0;padding-left:18px;list-style:disc">
                    <li style="color:#166534 !important;font-size:13px;
                               line-height:2;margin:2px 0">
                        Use natural daylight — avoid shadows
                    </li>
                    <li style="color:#166534 !important;font-size:13px;
                               line-height:2;margin:2px 0">
                        Focus on a single cassava leaf
                    </li>
                    <li style="color:#166534 !important;font-size:13px;
                               line-height:2;margin:2px 0">
                        Fill the frame with the leaf
                    </li>
                    <li style="color:#166534 !important;font-size:13px;
                               line-height:2;margin:2px 0">
                        Keep camera steady and close
                    </li>
                </ul>
            </div>""")
            predict_btn = gr.Button(
                "🔍  Analyse Leaf",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            result_html = gr.HTML("""
            <div style="background:white;border:2px dashed #D1FAE5;
                        border-radius:20px;padding:56px 24px;text-align:center;
                        min-height:420px;display:flex;flex-direction:column;
                        align-items:center;justify-content:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.04)">
                <div style="font-size:64px;margin-bottom:16px;opacity:0.6">🌿</div>
                <h3 style="color:#374151;margin:0 0 8px;font-size:18px;
                           font-weight:700">Results will appear here</h3>
                <p style="color:#9CA3AF;font-size:14px;margin:0;line-height:1.7">
                    Upload a cassava leaf photo and click<br>
                    <b style="color:#16A34A">Analyse Leaf</b> to get instant diagnosis
                </p>
            </div>""")

    how_html    = gr.HTML(build_how_it_works("🇬🇧 English"))
    footer_html = gr.HTML(build_footer("🇬🇧 English"))

    # ── EVENTS ────────────────────────────────────────────────
    predict_btn.click(
        fn=predict,
        inputs=[image_input, language],
        outputs=[result_html]
    )

    def update_ui(lang):
        return (
            build_hero(lang),
            build_disease_legend(lang),
            build_how_it_works(lang),
            build_footer(lang),
        )

    language.change(
        fn=update_ui,
        inputs=[language],
        outputs=[hero_html, legend_html, how_html, footer_html]
    )

demo.launch()
