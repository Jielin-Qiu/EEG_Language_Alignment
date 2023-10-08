import torch

device = torch.device('cuda')
# device = torch.device('cpu')

d_model = 16
class_num = 3
d_inner = 32
dropout = 0.3
warm_steps = 2000
fea_num = 7
PAD = 0
KS = 3
d_k = 64
d_v = 64

Fea_PLUS = 2
EEG_LEN = 832
TEXT_LEN = 768
# SIG_LEN3 = 6

'''
# --- For K-EmoCon ternary
SIG_LEN = 768
SIG_LEN2 = 48
SIG_LEN3 = 6

# --- For K-Emocon binary
SIG_LEN = 768
SIG_LEN2 = 48
SIG_LEN3 = 4

# --- For ZuCo sentence SA
SIG_LEN = 768
SIG_LEN2 = 838
SIG_LEN3 = 6

# --- For ZuCo word SA
SIG_LEN = 768
SIG_LEN2 = 832
SIG_LEN3 = 6

# --- ZuCo word concat SA
SIG_LEN = 768
SIG_LEN2 = 1598
SIG_LEN3 = 6

# --- ZuCo sentence RD
SIG_LEN = 768
SIG_LEN2 = 832
SIG_LEN3 = 20

# --- ZuCo word RD
SIG_LEN = 768
SIG_LEN2 = 832
SIG_LEN3 = 20

# --- ZuCo word concat RD
SIG_LEN = 768
SIG_LEN2 = 1598
SIG_LEN3 = 6
'''

MAX_LEN = 32
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'

# --- For K-EmoCon, pick from [happy_trans, happy2_trans, angry_trans, 
#                               angry2_trans, sad_trans, sad2_trans, nervous_trans, nervous2_trans]
# if {emotion}_trans pick eeg with delta0
# if {emotion}2_trans pick eeg with delta0_2

emotion = 'sad_trans'
csv = 'df.csv'

eeg = [emotion, 'delta0', 'lowAlpha0', 'highAlpha0','lowBeta0','highBeta0', 'lowGamma0', 'middleGamma0', 'theta0',
         'delta1', 'lowAlpha1', 'highAlpha1', 'lowBeta1', 'highBeta1', 'lowGamma1', 'middleGamma1', 'theta1',
         'delta2', 'lowAlpha2', 'highAlpha2', 'lowBeta2', 'highBeta2', 'lowGamma2', 'middleGamma2', 'theta2',
         'delta3', 'lowAlpha3', 'highAlpha3', 'lowBeta3', 'highBeta3', 'lowGamma3', 'middleGamma3', 'theta3',
         'delta4', 'lowAlpha4', 'highAlpha4', 'lowBeta4', 'highBeta4', 'lowGamma4', 'middleGamma4', 'theta4',
         'delta5', 'lowAlpha5', 'highAlpha5', 'lowBeta5', 'highBeta5', 'lowGamma5', 'middleGamma5', 'theta5']

# eeg = [emotion, 'delta0_2', 'lowAlpha0_2', 'highAlpha0_2','lowBeta0_2','highBeta0_2', 'lowGamma0_2', 'middleGamma0_2', 'theta0_2',
#          'delta1_2', 'lowAlpha1_2', 'highAlpha1_2', 'lowBeta1_2', 'highBeta1_2', 'lowGamma1_2', 'middleGamma1_2', 'theta1_2',
#          'delta2_2', 'lowAlpha2_2', 'highAlpha2_2', 'lowBeta2_2', 'highBeta2_2', 'lowGamma2_2', 'middleGamma2_2', 'theta2_2',
#          'delta3_2', 'lowAlpha3_2', 'highAlpha3_2', 'lowBeta3_2', 'highBeta3_2', 'lowGamma3_2', 'middleGamma3_2', 'theta3_2',
#          'delta4_2', 'lowAlpha4_2', 'highAlpha4_2', 'lowBeta4_2', 'highBeta4_2', 'lowGamma4_2', 'middleGamma4_2', 'theta4_2',
#          'delta5_2', 'lowAlpha5_2', 'highAlpha5_2', 'lowBeta5_2', 'highBeta5_2', 'lowGamma5_2', 'middleGamma5_2', 'theta5_2']



# --- For ZuCo, if SA emotion = 'sentiment'
#               if RD emotion = '1'
# pick patient from ['ZAB', 'ZDM', 'ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKH', 'ZKW', 'ZMG']
patient = 'ZAB'
# emotion = 'sentiment'
# emotion = '1'


outdim_size = class_num
use_all_singular_values = False

torchload3 = 'Name of Model (.chkpt)'


