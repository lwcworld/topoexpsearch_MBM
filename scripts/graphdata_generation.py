from Utils_datagen import *

category_substrs = {
    'lab': ['lab', 'TEKNUTR', 'MIKROSK', 'LAB', 'ARB', 'EXPERIMENTHALL', 'ELEKTROKEM', 'FORSK', 'MÄTRUM', 'VÅGRUM',
            'STUDIO MOTION', 'MÖRKRUM', 'office', 'KANSLI', 'KONTOR', 'KONT', 'SEKRE-TERARE', 'PROF', 'RUM FÖR RO.ANL.',
            'RUM F DRAGPROV', 'PERSONALRUM', 'TEKN PERSONAL', 'INSTRUKTÖR'],
    'corridor': ['corridor', 'pass', 'KORRIDOR', 'KORR', 'PASS', 'TERMINAL', 'ENTRE', 'LÄGENHETSFÖRRÅD', 'stair', 'TR.',
                 'TRAPPHUS', 'TRAPP', 'H1', 'TRAPPA', 'TARPPA', 'HISS', 'J/F/U'],
    'storage': ['storage', 'STÄD', 'MELLANLAGR', 'INSTRUMENTFÖRRÅD', 'FRD', 'KAPPR', 'BATTERI', 'CYKELFÖRRÅD',
                'KARTFÖRRÅD', 'FÖRRÅD', 'OMKL', 'equipment', 'MÄSS', 'TRYCKLUFT', 'BAND-ROBOTRUM', 'BANDROBOT', 'DATA',
                'ANALYS', 'SERVER', 'INSTRUMENT FRD', 'SALTDIMMA', 'GJUTHALL', 'KOPIERING'],
    'toilet': ['toilet', 'KPR', 'TOALETT', 'TORKRUM', 'TVÄTT', 'URINOIR', 'WC', 'DUSCH'],
    'food': ['food', 'PRNTRY', 'KÄLLSORTERING', 'KOP', 'KÖK', 'DISP', 'CHEF', 'BAR', 'LUNCH', 'LUNSCH', 'UGNSRUM',
             'UNGSRUM', 'DISK', 'AUTOKLAV', 'PENTRY'],
    'share': ['share', 'ELEV-FACK', 'LÄSPL', 'UPPEHÅLLSRUM', 'LOGE', 'MÖTE', 'SAMMANTRÄDE', 'TELE', 'LÄROSAL', 'KQNTOR',
              'KURSLOKAL', 'IT-RUM', 'LÄSRUM', 'LOUNGE', 'GÄST-MATSAL', 'GEMENSAMHETSLOKAL', 'LEKTIONSSAL', 'VERKSTAD',
              'STÄD', 'GEMENSAM VERKSTAD', 'HALL', 'BIBL', 'SEM', 'DATORRUM', 'VIL', 'DAGRUM', 'STYRELSERUM TEKNOLOGER',
              'SAMMANTRÄDE', 'RECEPTION', 'FINMEK VERKSTAD', 'LÄSP', 'LÄROSAL', 'KONF', 'PROVNINGSHALL', 'HÖRSAL',
              'GRUPPRUM', 'LÄSESAL', 'comport', 'LOGE', 'GALLERI', 'TVAGNING', 'GYM', 'TRH', 'WS', 'SKYDDSRUM', 'MÖTE',
              'FOAJE', 'VÄNTRUM', 'BASTU', 'UPPEHÅLL', 'GODSMOTTANGING', 'PAUS', 'FÖRRUM', 'KAFFERUM', 'TERRASS',
              'ÖVNINGSSAL'],
    'maintenence': ['maintenence', 'FLÄKTRUM', 'AFM', 'UMV', 'NOD FA', 'UPS-RUM', 'GAMLA SPEX', 'VF', 'VVS', 'HM',
                    'FÖRSÖRJN', 'TÄTPACKNING', 'KK', 'VIND', 'MÅLERI', 'UC', 'KYLCENTRAL', 'SLUSS', 'INBRLARM',
                    'FISKODL', 'KORSKOPPLINGSRUM', 'FLÄKT', 'POST', 'NÖD UT', 'AVFALLSRUM', 'TRANSPORTGÅNG', 'TEKNIK',
                    'FLÄKTRUM', 'BYGGPLATS', 'AVFETTNING', 'UNDERCENTRAL', 'ÖVERVAKNING', 'SOP', 'VATTENRENING',
                    'STÄLLV', 'OMFORMARRUM', 'BLÄSTER', 'RESERVKRAFT', 'VINDFÅNG', 'SCHAKT', 'POST SORT.FAX',
                    'MATERIAL FRD', '']}

cmap_category = {'lab': (0, 0, 255), 'corridor': (0, 150, 255), 'share': (0, 255, 255), 'storage': (0, 255, 0),
                 'toilet': (255, 0, 0), 'maintenence': (0, 255, 0), 'food': (0, 255, 255)}
imap_category = {'lab': 1, 'corridor': 2, 'share': 3, 'storage': 4, 'toilet': 5, 'maintenence': 4, 'food': 3}

# selected map
map_name_list = ['0510032194_A_40_1_102',
                 '0510034839_A-40.1-101',
                 '50056457',
                 '50056458',
                 '50056459',
                 '50015850_PLAN4',
                 '50015850_PLAN5',
                 '0510045906_A_40_1_104',
                 '50045231',
                 '50045232',
                 '0510030968_A-40',
                 '50052751',
                 '50052753',
                 '50052754',
                 '50052752',
                 '0510032270_A_40_1_104',
                 '50055639',
                 '50055640',
                 '0510025537_Layout1_PLAN3',
                 '0510025537_Layout1_PLAN2']


dir_map = '../dataset/dataset_kth/*/'
